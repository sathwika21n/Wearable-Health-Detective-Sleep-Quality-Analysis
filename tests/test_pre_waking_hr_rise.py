"""Tests for the pre-waking heart-rate rise in scripts/pre_waking_hr_rise.py.

Uses small made-up nights, never participant data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pre_waking_hr_rise import (  # noqa: E402
    awakening_rows,
    awakening_times,
    build_tables,
    nightly_summary,
    prepare_signals,
    rise_metrics,
)


def night(stages, heart_rates, start="2026-01-01 23:00:00", date="2026-01-01", step_min=5):
    times = pd.date_range(start, periods=len(stages), freq=f"{step_min}min")
    return pd.DataFrame({
        "date": date,
        "datetime_utc": times.strftime("%Y-%m-%d %H:%M:%S"),
        "hypnogram_class": stages,
        "heart_rate": heart_rates,
    })


def full_window(*values):
    """Heart rate at 30, 25, 20, 15, 10 and 5 minutes before waking."""
    return dict(zip([30, 25, 20, 15, 10, 5], values))


def test_capital_rem_counts_as_sleep_and_zero_heart_rate_is_missing():
    signals = prepare_signals(night(["REM", "awake"], [0, 60]))
    assert list(signals["stage"]) == ["rem", "awake"]
    assert pd.isna(signals["heart_rate"].iloc[0])


def test_awakening_needs_sleep_then_awake_exactly_five_minutes_apart():
    signals = prepare_signals(night(["light", "awake", "awake", "REM", "awake"], [50] * 5))
    assert len(awakening_times(signals)) == 2
    gapped = prepare_signals(night(["light", "awake"], [50, 60], step_min=10))
    assert awakening_times(gapped) == []


def test_steady_heart_rate_is_no_rise():
    result = rise_metrics(full_window(50, 50, 51, 50, 50, 51), sleeping_hr=50)
    assert result["status"] == "no_rise"
    assert result["rise_vs_sleeping_bpm"] == pytest.approx(1.0)
    assert result["rise_start_min_before"] is None


def test_rise_start_is_the_earliest_sample_of_the_run_ending_at_waking():
    # Above 50 + 3 from 15 minutes before waking onward; the 25-min spike doesn't count.
    result = rise_metrics(full_window(50, 56, 50, 54, 57, 60), sleeping_hr=50)
    assert result["status"] == "rise"
    assert result["rise_vs_sleeping_bpm"] == pytest.approx(10.0)
    assert result["rise_vs_30min_before_bpm"] == pytest.approx(10.0)
    assert result["rise_start_min_before"] == 15


def test_a_missing_sample_ends_the_rise_run():
    result = rise_metrics(full_window(55, 55, 55, None, 56, 58), sleeping_hr=50)
    assert result["rise_start_min_before"] == 10


def test_threshold_is_inclusive_and_adjustable():
    window = full_window(50, 50, 50, 50, 50, 53)
    assert rise_metrics(window, sleeping_hr=50)["status"] == "rise"
    assert rise_metrics(window, sleeping_hr=50, threshold_bpm=5)["status"] == "no_rise"


@pytest.mark.parametrize("window, sleeping_hr", [
    (full_window(50, 50, 50, 50, 50, None), 50),   # no sample just before waking
    (full_window(None, None, None, None, 50, 60), 50),  # too few samples
    (full_window(50, 50, 50, 50, 50, 60), None),   # no sleeping heart rate that night
])
def test_too_little_data_is_reported_not_guessed(window, sleeping_hr):
    result = rise_metrics(window, sleeping_hr)
    assert result["status"] == "insufficient_data"
    assert result["rise_vs_sleeping_bpm"] is None


def test_awakening_rows_mark_the_final_wake_and_use_that_nights_sleeping_hr():
    stages = ["light"] * 7 + ["awake"] + ["deep"] * 6 + ["awake"]
    rates = [50] * 6 + [58, 70] + [50] * 5 + [60, 72]
    rows = awakening_rows("par_9", prepare_signals(night(stages, rates)))
    assert [row["is_final_wake"] for row in rows] == [False, True]
    first = rows[0]
    # 13 sleep epochs: 6 x 50, 58, 5 x 50, 60 -> 668 / 13 = 51.4
    assert first["night_sleeping_hr_bpm"] == pytest.approx(51.4)
    assert first["hr_minus_5m"] == 58
    assert first["status"] == "rise"
    assert first["rise_start_min_before"] == 5


def test_nightly_summary_counts_rises_and_reports_the_final_wake():
    awakenings = pd.DataFrame([
        {"participant_id": "par_9", "night_date": "2026-01-01", "is_final_wake": False,
         "status": "no_rise", "rise_vs_sleeping_bpm": 1.0, "rise_start_min_before": None},
        {"participant_id": "par_9", "night_date": "2026-01-01", "is_final_wake": True,
         "status": "rise", "rise_vs_sleeping_bpm": 9.0, "rise_start_min_before": 15},
        {"participant_id": "par_9", "night_date": "2026-01-02", "is_final_wake": True,
         "status": "insufficient_data", "rise_vs_sleeping_bpm": None, "rise_start_min_before": None},
    ])
    summary = nightly_summary(awakenings).set_index("night_date")
    first = summary.loc["2026-01-01"]
    assert (first["awakenings"], first["awakenings_judged"], first["awakenings_with_rise"]) == (2, 2, 1)
    assert first["mean_rise_vs_sleeping_bpm"] == pytest.approx(5.0)
    assert first["final_wake_status"] == "rise"
    assert first["final_wake_rise_start_min_before"] == 15
    assert summary.loc["2026-01-02", "awakenings_judged"] == 0


def test_build_tables_reads_aligned_files(tmp_path):
    night(["light"] * 7 + ["awake"], [50] * 6 + [55, 70]).to_csv(tmp_path / "par_9_aligned.csv", index=False)
    awakenings, nightly = build_tables(tmp_path)
    assert len(awakenings) == 1 and len(nightly) == 1
    assert awakenings.iloc[0]["participant_id"] == "par_9"


def test_no_aligned_files_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_tables(tmp_path)
