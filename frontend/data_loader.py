"""Load and validate the dashboard's long-format results table.

Kept separate from app.py so it can be tested without starting Streamlit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SAMPLE_PATH = APP_DIR / "sample_sleep_dashboard.csv"
GENERATED_PATH = PROJECT_DIR / "outputs" / "change_detection_results.csv"
REQUIRED_COLUMNS = {
    "participant_id", "night_date", "metric", "unit", "observed_value",
    "baseline_value", "difference", "baseline_window", "baseline_nights",
    "minimum_nights", "status", "message",
}
VALID_STATUSES = {"change", "normal", "missing", "insufficient_history"}


def load_dashboard_data(
    uploaded_file=None,
    generated_path: Path = GENERATED_PATH,
    sample_path: Path = SAMPLE_PATH,
) -> tuple[pd.DataFrame, str]:
    """Load upload, generated project results, or the sample fallback, in that order."""
    if uploaded_file is not None:
        source = uploaded_file
        source_name = uploaded_file.name
    elif generated_path.exists():
        source = generated_path
        source_name = f"{generated_path.name} (generated project results)"
    else:
        source = sample_path
        source_name = f"{sample_path.name} (sample fallback)"
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype("string")
    frame["night_date"] = pd.to_datetime(frame["night_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["metric"] = frame["metric"].astype(str)
    frame["unit"] = frame["unit"].astype("string").fillna("").astype(str)
    frame["status"] = frame["status"].fillna("").astype(str).str.lower()
    for column in ["observed_value", "baseline_value", "difference", "baseline_nights", "minimum_nights"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["message"] = frame["message"].astype("string").fillna("").astype(str)
    for column in ["baseline_std", "z_score", "threshold_value"]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "threshold_type" not in frame:
        frame["threshold_type"] = ""
    invalid_statuses = set(frame["status"].dropna().unique()) - VALID_STATUSES
    if invalid_statuses:
        raise ValueError(f"Unsupported status values: {', '.join(sorted(invalid_statuses))}")
    missing_observation = frame["observed_value"].isna() & ~frame["status"].eq("insufficient_history")
    frame.loc[missing_observation, "status"] = "missing"
    frame = frame.dropna(subset=["participant_id", "night_date"])
    return frame, source_name
