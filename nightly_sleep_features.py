"""Create one row of sleep features per participant sleep window.

The aligned CSVs repeat nightly summary values on each signal row. This module
reduces those rows to one record per sleep window and converts duration fields
from seconds to minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "sleep_start_utc",
    "sleep_end_utc",
    "night_total_sleep_seconds",
    "night_deep_sleep_seconds",
    "night_rem_sleep_seconds",
    "night_light_sleep_seconds",
    "night_awake_seconds",
    "night_observed_awakenings",
    "night_observed_stage_changes",
}

DURATION_COLUMNS = {
    "total_sleep_time_min": "night_total_sleep_seconds",
    "deep_sleep_min": "night_deep_sleep_seconds",
    "rem_sleep_min": "night_rem_sleep_seconds",
    "light_sleep_min": "night_light_sleep_seconds",
    "awake_min": "night_awake_seconds",
}

COUNT_COLUMNS = {
    "observed_awakenings_count": "night_observed_awakenings",
    "observed_stage_changes_count": "night_observed_stage_changes",
}

OUTPUT_COLUMNS = [
    "participant_id",
    "sleep_date",
    "sleep_start_utc",
    "sleep_end_utc",
    *DURATION_COLUMNS,
    *COUNT_COLUMNS,
]


def _one_value(rows: pd.DataFrame, column: str, participant_id: str) -> object:
    """Return a nightly value, rejecting conflicting repeated summaries."""
    values = rows[column].dropna().drop_duplicates()
    if len(values) > 1:
        raise ValueError(
            f"{participant_id}: conflicting {column} values within one sleep window: "
            f"{values.tolist()}"
        )
    return values.iloc[0] if not values.empty else pd.NA


def extract_participant_features(csv_path: Path) -> list[dict[str, object]]:
    """Extract nightly feature rows from one ``par_*_aligned.csv`` file."""
    participant_id = csv_path.name.removesuffix("_aligned.csv")
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    keys = ["date", "sleep_start_utc", "sleep_end_utc"]
    features: list[dict[str, object]] = []
    for key, night in data.groupby(keys, dropna=False, sort=True):
        sleep_date, start_utc, end_utc = key
        row: dict[str, object] = {
            "participant_id": participant_id,
            "sleep_date": sleep_date if pd.notna(sleep_date) else pd.NA,
            "sleep_start_utc": start_utc if pd.notna(start_utc) else pd.NA,
            "sleep_end_utc": end_utc if pd.notna(end_utc) else pd.NA,
        }
        for output_column, source_column in DURATION_COLUMNS.items():
            seconds = _one_value(night, source_column, participant_id)
            row[output_column] = (
                float(seconds) / 60.0 if pd.notna(seconds) else pd.NA
            )
        for output_column, source_column in COUNT_COLUMNS.items():
            row[output_column] = _one_value(night, source_column, participant_id)
        features.append(row)
    return features


def build_nightly_features(
    aligned_dir: str | Path = "aligned_data",
) -> pd.DataFrame:
    """Load all participant aligned files and return a combined nightly table."""
    aligned_dir = Path(aligned_dir)
    input_files = sorted(aligned_dir.glob("par_*_aligned.csv"))
    if not input_files:
        raise FileNotFoundError(f"No par_*_aligned.csv files found in {aligned_dir}")

    rows: list[dict[str, object]] = []
    for csv_path in input_files:
        rows.extend(extract_participant_features(csv_path))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        ["participant_id", "sleep_date", "sleep_start_utc"], na_position="last"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one combined nightly sleep feature table from aligned CSVs."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "aligned_data",
        help="Folder containing par_*_aligned.csv files (default: ./aligned_data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "nightly_sleep_features.csv",
        help="Output CSV path (default: ./nightly_sleep_features.csv)",
    )
    args = parser.parse_args()

    table = build_nightly_features(args.aligned_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(
        f"Wrote {len(table)} nights for {table['participant_id'].nunique()} "
        f"participants to {args.output}"
    )


if __name__ == "__main__":
    main()
