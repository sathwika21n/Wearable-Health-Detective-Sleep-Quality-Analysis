"""Calculate each participant's prior-10-night baselines.

Each row contains the sleep, heart-rate, and temperature baseline calculated
from up to the participant's 10 immediately preceding nights. Temperature is
the source nightly temperature delta, not an absolute skin-temperature
measurement.

Run from the project root:
    python participant_baselines.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SLEEP_FEATURES = PROJECT_DIR / "nightly_sleep_features.csv"
DEFAULT_PHYSIOLOGY = PROJECT_DIR / "model_ready_sleep_score.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "participant_baselines.csv"

BASELINE_METRICS = {
    "total_sleep_time_min": "typical_sleep_time_min",
    "average_heart_rate_bpm": "typical_heart_rate_bpm",
    "temperature_delta_c": "typical_temperature_delta_c",
}
LOOKBACK_NIGHTS = 10


def _write_csv(data: pd.DataFrame, output_path: Path) -> Path:
    """Write a CSV, using a unique sibling path if the destination is locked."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(output_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fallback_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )
        data.to_csv(fallback_path, index=False)
        print(
            f"Could not overwrite {output_path}; it may be open in another "
            f"program. Wrote the output to {fallback_path} instead."
        )
        return fallback_path
    return output_path


def _read_nights(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Load and validate one row per participant-night."""
    data = pd.read_csv(path)
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"{path}: missing required columns {sorted(missing_columns)}")

    data = data[list(required_columns)].copy()
    data["participant_id"] = data["participant_id"].astype("string").str.strip()
    data["sleep_date"] = pd.to_datetime(data["sleep_date"], errors="coerce")
    if data["participant_id"].isna().any() or data["participant_id"].eq("").any():
        raise ValueError(f"{path}: participant_id contains blank values")
    if data["sleep_date"].isna().any():
        raise ValueError(f"{path}: sleep_date contains missing or invalid values")
    data["sleep_date"] = data["sleep_date"].dt.normalize()
    duplicates = data.duplicated(["participant_id", "sleep_date"], keep=False)
    if duplicates.any():
        examples = data.loc[duplicates, ["participant_id", "sleep_date"]].head().to_dict("records")
        raise ValueError(f"{path}: duplicate participant-night rows, for example: {examples}")
    return data


def build_prior_night_baselines(
    sleep_features_path: str | Path = DEFAULT_SLEEP_FEATURES,
    physiology_path: str | Path = DEFAULT_PHYSIOLOGY,
) -> pd.DataFrame:
    """Build per-night baselines using only each participant's earlier nights."""
    sleep_features_path = Path(sleep_features_path)
    physiology_path = Path(physiology_path)

    sleep = _read_nights(
        sleep_features_path,
        {"participant_id", "sleep_date", "total_sleep_time_min"},
    )
    physiology = _read_nights(
        physiology_path,
        {"participant_id", "sleep_date", "average_heart_rate_bpm", "temperature_delta_c"},
    )
    for metric in BASELINE_METRICS:
        if metric in sleep.columns:
            sleep[metric] = pd.to_numeric(sleep[metric], errors="coerce")
    for metric in ("average_heart_rate_bpm", "temperature_delta_c"):
        physiology[metric] = pd.to_numeric(physiology[metric], errors="coerce")

    nights = sleep.merge(
        physiology,
        on=["participant_id", "sleep_date"],
        how="left",
        validate="one_to_one",
    )
    nights = nights.sort_values(["participant_id", "sleep_date"]).reset_index(drop=True)
    nights["prior_nights_count"] = nights.groupby("participant_id").cumcount().clip(
        upper=LOOKBACK_NIGHTS
    )

    for metric, baseline_column in BASELINE_METRICS.items():
        nights[baseline_column] = nights.groupby("participant_id", sort=False)[metric].transform(
            lambda values: values.shift().rolling(
                window=LOOKBACK_NIGHTS, min_periods=1
            ).mean()
        )

    nights["sleep_date"] = nights["sleep_date"].dt.strftime("%Y-%m-%d")
    return nights[
        [
            "participant_id",
            "sleep_date",
            "prior_nights_count",
            "typical_sleep_time_min",
            "typical_heart_rate_bpm",
            "typical_temperature_delta_c",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-features", type=Path, default=DEFAULT_SLEEP_FEATURES)
    parser.add_argument("--physiology", type=Path, default=DEFAULT_PHYSIOLOGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    baselines = build_prior_night_baselines(args.sleep_features, args.physiology)
    output_path = _write_csv(baselines, args.output)
    print(
        f"Wrote prior-night baselines for {len(baselines)} participant-nights "
        f"across {baselines['participant_id'].nunique()} participants"
    )
    print(
        f"Baselines use up to the {LOOKBACK_NIGHTS} immediately preceding nights; "
        "the first night per participant has no baseline."
    )


if __name__ == "__main__":
    main()