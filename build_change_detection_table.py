"""Build a dashboard-ready, long-format baseline/change-detection table.

This uses the same prior-night windows and thresholds as baseline_deviations.py,
while preserving inputs needed to audit each comparison. It exports one row per
participant, night, metric, and baseline window.

Run from the project root:
    python3 build_change_detection_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ALIGNED_DIR = PROJECT_DIR / "aligned_data"
DEFAULT_AWAKENING_FILE = PROJECT_DIR / "participant_nightly_awakening" / "nightly_awakenings.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "change_detection_results.csv"

# Each window counts prior observed nights, not calendar days.
WINDOWS = [
    ("1_day", 1, 1, "absolute_difference"),
    ("7_days", 7, 3, "absolute_z_score"),
    ("14_days", 14, 5, "absolute_z_score"),
    ("30_days", 30, 5, "absolute_z_score"),
]

METRIC_SOURCE_COLUMNS = {
    "total_sleep_min": ("night_total_sleep_seconds", 60.0, "min"),
    "deep_sleep_min": ("night_deep_sleep_seconds", 60.0, "min"),
    "rem_sleep_min": ("night_rem_sleep_seconds", 60.0, "min"),
    "light_sleep_min": ("night_light_sleep_seconds", 60.0, "min"),
    "awake_min": ("night_awake_seconds", 60.0, "min"),
    "temperature_delta_c": ("temperature_delta", 1.0, "°C"),
}
METRIC_LABELS = {
    "total_sleep_min": "Total sleep",
    "deep_sleep_min": "Deep sleep",
    "rem_sleep_min": "REM sleep",
    "light_sleep_min": "Light sleep",
    "awake_min": "Awake time",
    "observed_awakenings_count": "Observed awakenings",
    "sleeping_hr_avg_bpm": "Sleeping heart rate",
    "temperature_delta_c": "Temperature deviation",
}

OUTPUT_COLUMNS = [
    "participant_id",
    "night_date",
    "metric",
    "metric_label",
    "unit",
    "observed_value",
    "baseline_value",
    "baseline_std",
    "difference",
    "z_score",
    "baseline_window",
    "baseline_nights",
    "minimum_nights",
    "threshold_type",
    "threshold_value",
    "status",
    "message",
]


def _first_valid(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _nightly_values(aligned_dir: Path, awakening_file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(aligned_dir.rglob("par_*_aligned.csv")):
        participant_id = path.name.removesuffix("_aligned.csv")
        data = pd.read_csv(path)
        if "date" not in data:
            continue
        for night_date, group in data.groupby("date", dropna=True, sort=True):
            row: dict[str, object] = {
                "participant_id": participant_id,
                "night_date": pd.to_datetime(night_date, errors="coerce").strftime("%Y-%m-%d"),
            }
            for metric, (source, divisor, _unit) in METRIC_SOURCE_COLUMNS.items():
                value = _first_valid(group[source]) if source in group else None
                value = value / divisor if value is not None else None
                row[metric] = value

            if "heart_rate" in group and "hypnogram_class" in group:
                stages = group["hypnogram_class"].astype("string").str.strip().str.lower()
                asleep = stages.isin({"light", "deep", "rem", "sleep", "n1", "n2", "n3"})
                heart_rate = pd.to_numeric(group.loc[asleep, "heart_rate"], errors="coerce")
                heart_rate = heart_rate[heart_rate > 0]
                row["sleeping_hr_avg_bpm"] = float(heart_rate.mean()) if not heart_rate.empty else None
            else:
                row["sleeping_hr_avg_bpm"] = None
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No par_*_aligned.csv files found in {aligned_dir}")
    nights = pd.DataFrame(rows)

    if awakening_file.exists():
        awakenings = pd.read_csv(awakening_file)
        needed = {"participant_id", "date", "awakening_count"}
        if needed.issubset(awakenings.columns):
            awakenings = awakenings[["participant_id", "date", "awakening_count"]].copy()
            awakenings["participant_id"] = awakenings["participant_id"].astype(str)
            awakenings["night_date"] = pd.to_datetime(awakenings.pop("date"), errors="coerce").dt.strftime("%Y-%m-%d")
            awakenings = awakenings.rename(columns={"awakening_count": "observed_awakenings_count"})
            nights = nights.merge(awakenings, on=["participant_id", "night_date"], how="left")
        else:
            nights["observed_awakenings_count"] = None
    else:
        nights["observed_awakenings_count"] = None

    nights["night_date"] = pd.to_datetime(nights["night_date"], errors="coerce")
    return nights.sort_values(["participant_id", "night_date"]).reset_index(drop=True)


def _threshold(metric: str, window: str) -> tuple[str, float]:
    if window == "1_day":
        if metric == "sleeping_hr_avg_bpm":
            return "absolute_difference", 5.0
        if metric == "temperature_delta_c":
            return "absolute_difference", 0.5
        return "absolute_difference", 10.0
    return "absolute_z_score", 2.0


def build_change_table(
    aligned_dir: str | Path = DEFAULT_ALIGNED_DIR,
    awakening_file: str | Path = DEFAULT_AWAKENING_FILE,
) -> pd.DataFrame:
    """Calculate prior-night comparisons and retain each flag's evidence."""
    nights = _nightly_values(Path(aligned_dir), Path(awakening_file))
    metric_units = {metric: unit for metric, (_source, _divisor, unit) in METRIC_SOURCE_COLUMNS.items()}
    metric_units.update({"sleeping_hr_avg_bpm": "bpm", "observed_awakenings_count": "count"})
    metric_order = list(METRIC_LABELS)
    output: list[dict[str, object]] = []

    for participant_id, participant in nights.groupby("participant_id", sort=True):
        participant = participant.sort_values("night_date").reset_index(drop=True)
        for index, current in participant.iterrows():
            for metric in metric_order:
                observed = current.get(metric)
                for window, lookback, minimum, threshold_type in WINDOWS:
                    prior = pd.to_numeric(
                        participant.iloc[max(0, index - lookback):index][metric],
                        errors="coerce",
                    ).dropna()
                    count = int(prior.size)
                    baseline = float(prior.mean()) if count else None
                    baseline_std = float(prior.std()) if count > 1 else None
                    threshold_type, threshold_value = _threshold(metric, window)
                    difference = None
                    z_score = None

                    if count < minimum:
                        status = "insufficient_history"
                        message = f"Only {count} valid prior nights; {minimum} required."
                    elif pd.isna(observed):
                        status = "missing"
                        message = "Current night measurement is missing."
                    else:
                        difference = float(observed) - float(baseline)
                        if window == "1_day":
                            flagged = abs(difference) >= threshold_value
                            message = f"|difference| {'>=' if flagged else '<'} {threshold_value:g} {metric_units[metric]}."
                        else:
                            z_score = (difference / baseline_std) if baseline_std is not None and baseline_std > 0 else 0.0
                            flagged = abs(z_score) >= threshold_value
                            message = f"|z-score| {'>=' if flagged else '<'} {threshold_value:g}; baseline standard deviation={baseline_std if baseline_std is not None else 'undefined'}."
                        status = "change" if flagged else "normal"

                    output.append({
                        "participant_id": participant_id,
                        "night_date": current["night_date"].strftime("%Y-%m-%d"),
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "unit": metric_units[metric],
                        "observed_value": observed,
                        "baseline_value": baseline,
                        "baseline_std": baseline_std,
                        "difference": difference,
                        "z_score": z_score,
                        "baseline_window": window,
                        "baseline_nights": count,
                        "minimum_nights": minimum,
                        "threshold_type": threshold_type,
                        "threshold_value": threshold_value,
                        "status": status,
                        "message": message,
                    })
    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR)
    parser.add_argument("--awakening-file", type=Path, default=DEFAULT_AWAKENING_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table = build_change_table(args.aligned_dir, args.awakening_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(
        f"Wrote {len(table)} participant-night-metric-window comparisons "
        f"for {table['participant_id'].nunique()} participants to {args.output}"
    )
    print(table["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
