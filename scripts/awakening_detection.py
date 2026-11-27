"""Shared stage normalization and sleep-to-awake transition logic."""

from pathlib import Path

import pandas as pd


EXPECTED_INTERVAL = pd.Timedelta(minutes=5)
VALID_STAGES = {"awake", "light", "deep", "rem"}
SLEEP_STAGES = VALID_STAGES - {"awake"}


def normalize_stages(stages: pd.Series) -> pd.Series:
    """Normalize labels so source capitalization does not change stage meaning."""
    return stages.astype("string").str.strip().str.casefold()


def sleep_to_awake_mask(
    stages: pd.Series,
    timestamps: pd.Series,
) -> pd.Series:
    """Return transitions from a sleep stage to awake at adjacent 5-minute epochs."""
    normalized = normalize_stages(stages)
    previous = normalized.shift()
    adjacent = timestamps.diff().eq(EXPECTED_INTERVAL)
    return adjacent & normalized.eq("awake") & previous.isin(SLEEP_STAGES)


def build_nightly_awakenings(aligned_dir: str | Path) -> pd.DataFrame:
    """Count awakenings and awake minutes from aligned participant-night files."""
    aligned_dir = Path(aligned_dir)
    csv_files = sorted(aligned_dir.glob("par_*_aligned.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No par_*_aligned.csv files found in {aligned_dir}")

    rows = []
    required = {
        "date",
        "datetime_utc",
        "sleep_start_utc",
        "sleep_end_utc",
        "hypnogram_class",
    }
    for csv_path in csv_files:
        participant_id = csv_path.name.removesuffix("_aligned.csv")
        data = pd.read_csv(csv_path)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
        data["datetime_utc"] = pd.to_datetime(
            data["datetime_utc"], utc=True, errors="coerce"
        )
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

        for (night_date, _, _), night in data.groupby(
            ["date", "sleep_start_utc", "sleep_end_utc"], dropna=False, sort=True
        ):
            night = night.sort_values("datetime_utc")
            stages = normalize_stages(night["hypnogram_class"])
            valid = stages.isin(VALID_STAGES)
            if valid.any():
                awakening_count = int(
                    sleep_to_awake_mask(
                        night["hypnogram_class"], night["datetime_utc"]
                    ).sum()
                )
                awake_minutes = int(stages.eq("awake").sum() * 5)
            else:
                awakening_count = pd.NA
                awake_minutes = pd.NA
            rows.append(
                {
                    "participant_id": participant_id,
                    "date": (
                        night_date.strftime("%Y-%m-%d")
                        if pd.notna(night_date)
                        else pd.NA
                    ),
                    "awakening_count": awakening_count,
                    "total_awake_minutes": awake_minutes,
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "participant_id",
            "date",
            "awakening_count",
            "total_awake_minutes",
        ],
    )
    result["_participant_number"] = (
        result["participant_id"].str.extract(r"(\d+)").astype(int)
    )
    return (
        result.sort_values(["_participant_number", "date"])
        .drop(columns="_participant_number")
        .reset_index(drop=True)
    )
