"""Build a nightly Oura sleep-score dataset with leakage-safe features.

The target is Oura's overall ``score`` from each participant's sleep.csv.
Only a fixed whitelist of physiological measurements is used as model input.
Oura score components and sleep metrics that contribute directly to its Sleep
Score (duration, stages, efficiency, latency, restfulness, and timing) are
intentionally excluded.

Run from the project root:
    python3 build_sleep_score_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "ifh_affect"
DEFAULT_OUTPUT = PROJECT_DIR / "model_ready_sleep_score.csv"

TARGET_COLUMN = "target_sleep_score"
# Conservative feature whitelist: physiological measurements not used as
# Oura Sleep Score components. Never add score_* fields or sleep-score inputs.
FEATURE_COLUMNS = {
    "hr_average": "average_heart_rate_bpm",
    "hr_lowest": "lowest_heart_rate_bpm",
    "rmssd": "hrv_rmssd_ms",
    "breath_average": "average_breath_rate_bpm",
    "temperature_delta": "temperature_delta_c",
}
OUTPUT_COLUMNS = [
    "participant_id",
    "sleep_date",
    *FEATURE_COLUMNS.values(),
    TARGET_COLUMN,
]


def build_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Read participant Oura sleep files and return one labeled row per night."""
    data_dir = Path(data_dir)
    sleep_files = sorted(data_dir.glob("par_*/oura/sleep.csv"))
    if not sleep_files:
        raise FileNotFoundError(f"No par_*/oura/sleep.csv files found in {data_dir}")

    rows: list[pd.DataFrame] = []
    for path in sleep_files:
        participant_id = path.parents[1].name
        source = pd.read_csv(path)
        required = {"date", "score", *FEATURE_COLUMNS}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")

        # Preserve only identifiers, the whitelisted inputs, and the target.
        # This prevents score components or other source fields from leaking
        # into the model matrix through an unrestricted column selection.
        frame = pd.DataFrame(index=source.index)
        frame["participant_id"] = participant_id
        frame["sleep_date"] = pd.to_datetime(source["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for source_column, output_column in FEATURE_COLUMNS.items():
            frame[output_column] = pd.to_numeric(source[source_column], errors="coerce")
        frame[TARGET_COLUMN] = pd.to_numeric(source["score"], errors="coerce")
        rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True)[OUTPUT_COLUMNS]
    dataset = dataset.dropna(subset=["sleep_date", TARGET_COLUMN]).copy()
    duplicate_nights = dataset.duplicated(["participant_id", "sleep_date"], keep=False)
    if duplicate_nights.any():
        examples = dataset.loc[duplicate_nights, ["participant_id", "sleep_date"]].head().to_dict("records")
        raise ValueError(f"Duplicate participant-night rows found, for example: {examples}")
    if not dataset[TARGET_COLUMN].between(0, 100).all():
        raise ValueError("Overall sleep score contains values outside the expected 0-100 range")

    return dataset.sort_values(["participant_id", "sleep_date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = build_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(
        f"Wrote {len(dataset)} participant-nights for "
        f"{dataset['participant_id'].nunique()} participants to {args.output}"
    )
    print(f"Target: {TARGET_COLUMN}; model inputs: {', '.join(FEATURE_COLUMNS.values())}")
    print("Missing values by column:")
    print(dataset.isna().sum().to_string())


if __name__ == "__main__":
    main()
