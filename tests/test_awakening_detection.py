"""Tests for case-insensitive sleep-to-awake transition detection."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from awakening_hr_windows import (  # noqa: E402
    DEFAULT_OUTPUT,
    extract_awakening_hr_windows,
    write_output,
)
from awakening_detection import (  # noqa: E402
    build_nightly_awakenings,
    sleep_to_awake_mask,
)


def test_sleep_to_awake_mask_accepts_uppercase_rem_and_ignores_gaps():
    stages = pd.Series(["REM", "awake", "awake", "Deep", "awake", "Light", "awake"])
    timestamps = pd.Series(pd.to_datetime([
        "2026-01-01 00:00",
        "2026-01-01 00:05",
        "2026-01-01 00:10",
        "2026-01-01 00:15",
        "2026-01-01 00:20",
        "2026-01-01 00:25",
        "2026-01-01 00:35",
    ], utc=True))

    mask = sleep_to_awake_mask(stages, timestamps)

    assert mask.tolist() == [False, True, False, False, True, False, False]


def test_hr_window_extraction_includes_rem_to_awake(tmp_path):
    timestamps = pd.to_datetime([
        "2026-01-01 00:00",
        "2026-01-01 00:05",
        "2026-01-01 00:10",
        "2026-01-01 00:15",
        "2026-01-01 00:20",
        "2026-01-01 00:25",
        "2026-01-01 00:35",
        "2026-01-01 00:40",
        "2026-01-01 00:45",
    ], utc=True)
    aligned = pd.DataFrame({
        "date": ["2026-01-01"] * len(timestamps),
        "datetime_utc": timestamps,
        "hypnogram_class": ["REM", "awake", "awake", "Deep", "awake", "Light", "awake", "REM", "awake"],
        "heart_rate": [60, 59, 58, 57, 56, 55, 54, 53, 52],
    })
    aligned.to_csv(tmp_path / "par_1_aligned.csv", index=False)

    windows = extract_awakening_hr_windows(tmp_path)

    assert windows["awakening_time_utc"].tolist() == [timestamps[1], timestamps[4], timestamps[8]]
    assert windows["previous_sleep_stage"].tolist() == ["REM", "Deep", "REM"]


def test_nightly_export_counts_uppercase_rem_transitions(tmp_path):
    aligned = pd.DataFrame({
        "date": ["2026-01-01"] * 4,
        "datetime_utc": pd.date_range("2026-01-01", periods=4, freq="5min"),
        "sleep_start_utc": ["2026-01-01 00:00:00"] * 4,
        "sleep_end_utc": ["2026-01-01 00:20:00"] * 4,
        "hypnogram_class": ["REM", "awake", "light", "awake"],
    })
    aligned.to_csv(tmp_path / "par_2_aligned.csv", index=False)

    nightly = build_nightly_awakenings(tmp_path)

    assert nightly.loc[0, "awakening_count"] == 2
    assert nightly.loc[0, "total_awake_minutes"] == 10


def test_locked_output_uses_rem_fallback(tmp_path, monkeypatch):
    output = tmp_path / "awakening_hr_windows.csv"
    original_to_csv = pd.DataFrame.to_csv

    def fail_for_primary_path(self, path_or_buf=None, *args, **kwargs):
        if Path(path_or_buf) == output:
            raise PermissionError("file is locked")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_for_primary_path)
    written = write_output(pd.DataFrame({"previous_sleep_stage": ["REM"]}), output)

    assert written == tmp_path / "awakening_hr_windows_rem.csv"
    assert written.exists()


def test_default_output_uses_rem_filename():
    assert DEFAULT_OUTPUT.name == "awakening_hr_windows_rem.csv"
