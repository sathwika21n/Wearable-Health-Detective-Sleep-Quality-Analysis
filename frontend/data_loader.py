"""Load and validate the dashboard's long-format results table.

Kept separate from app.py so it can be tested without starting Streamlit.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SAMPLE_PATH = APP_DIR / "sample_sleep_dashboard.csv"
GENERATED_PATH = PROJECT_DIR / "outputs" / "change_detection_results.csv"
OURA_SOURCE = "Oura / public dataset"
APPLE_SOURCE = "Apple Health / Apple Watch"
APPLE_OUTPUT_DIR_ENV = "APPLE_HEALTH_OUTPUT_DIR"
APPLE_NIGHTLY_METRICS = ("hrv_sdnn_ms", "wrist_temperature_c")
REQUIRED_COLUMNS = {
    "participant_id", "night_date", "metric", "unit", "observed_value",
    "baseline_value", "difference", "baseline_window", "baseline_nights",
    "minimum_nights", "status", "message",
}
VALID_STATUSES = {"change", "normal", "missing", "insufficient_history"}


def apple_output_directory() -> Path | None:
    configured = os.getenv(APPLE_OUTPUT_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else None


def load_apple_nightly_features(output_dir: Path) -> pd.DataFrame:
    """Read optional Apple-only measurements keyed by participant and local wake date."""
    path = output_dir / "nightly_sleep_features.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["participant_id", "sleep_date", *APPLE_NIGHTLY_METRICS])
    keys = ["participant_id", "sleep_date"]
    frame = pd.read_csv(path, usecols=lambda column: column in (*keys, *APPLE_NIGHTLY_METRICS))
    if not set(keys).issubset(frame):
        raise ValueError("Apple nightly features need participant_id and sleep_date columns.")
    frame["participant_id"] = frame["participant_id"].astype("string")
    frame["sleep_date"] = pd.to_datetime(frame["sleep_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for metric in APPLE_NIGHTLY_METRICS:
        frame[metric] = pd.to_numeric(frame.get(metric), errors="coerce")
    frame = frame.dropna(subset=keys)
    if frame.duplicated(keys).any():
        raise ValueError("Apple nightly features need one row per participant and sleep date.")
    return frame


def load_dashboard_data(
    uploaded_file=None,
    generated_path: Path = GENERATED_PATH,
    sample_path: Path = SAMPLE_PATH,
    *,
    apple_output_dir: Path | None = None,
    data_source: str = OURA_SOURCE,
) -> tuple[pd.DataFrame, str]:
    """Load Apple results or use the upload, generated Oura, and sample precedence."""
    if data_source == APPLE_SOURCE:
        source = apple_output_dir / "change_detection_results.csv" if apple_output_dir is not None else None
        if source is None or not source.is_file():
            return pd.DataFrame(), "Apple Health results unavailable"
        source_name = "Apple Health results"
    elif uploaded_file is not None:
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
    if data_source == APPLE_SOURCE:
        frame = frame.loc[~frame["metric"].isin({"hrv_rmssd_ms", "temperature_delta_c"})]
    return frame, source_name
