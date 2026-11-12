"""Tests for pointing out flagged changes on the night timeline (frontend/charts.py)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from charts import timeline_chart, timeline_highlights  # noqa: E402


def flagged(*rows):
    columns = ["metric", "metric_label", "unit", "observed_value", "baseline_value", "baseline_window"]
    return pd.DataFrame([dict(zip(columns, row)) for row in rows])


def test_nothing_flagged_means_no_highlights():
    for value in (None, flagged()):
        result = timeline_highlights(value)
        assert result["stages"].empty
        assert result["heart_rate"] is None
        assert result["elsewhere"] == []


def test_stage_metrics_highlight_their_row():
    result = timeline_highlights(flagged(
        ("deep_sleep_min", "Deep sleep", "min", 48, 73, "7_days"),
        ("rem_sleep_min", "REM sleep", "min", 120, 95, "7_days"),
    ))
    notes = dict(zip(result["stages"]["stage"], result["stages"]["note"]))
    assert notes == {
        "Deep": "Deep sleep: 48 min vs usual 73 min",
        "REM": "REM sleep: 120 min vs usual 95 min",
    }


def test_awake_time_and_awakenings_share_the_awake_row():
    result = timeline_highlights(flagged(
        ("awake_min", "Awake time", "min", 70, 40, "7_days"),
        ("observed_awakenings_count", "Observed awakenings", "count", 6, 2.5, "7_days"),
    ))
    assert list(result["stages"]["stage"]) == ["Awake"]
    assert result["stages"]["note"].iloc[0] == "Awake time: 70 min vs usual 40 min; Observed awakenings: 6 vs usual 2.5"


def test_heart_rate_change_draws_the_usual_level():
    result = timeline_highlights(flagged(("sleeping_hr_avg_bpm", "Sleeping heart rate", "bpm", 66, 58.4, "7_days")))
    assert result["heart_rate"] == {"heart_rate": 58.4, "note": "Sleeping heart rate: 66 bpm vs usual 58 bpm"}


def test_whole_night_metrics_are_listed_instead():
    result = timeline_highlights(flagged(
        ("total_sleep_min", "Total sleep", "min", 300, 420, "7_days"),
        ("temperature_delta_c", "Temperature deviation", "°C", 0.9, 0.1, "7_days"),
    ))
    assert result["stages"].empty
    assert result["elsewhere"] == ["Total sleep", "Temperature deviation"]


def test_chart_builds_with_highlights():
    start = pd.Timestamp("2026-01-01 23:00")
    night = pd.DataFrame({
        "start": [start, start + pd.Timedelta(minutes=5)],
        "end": [start + pd.Timedelta(minutes=5), start + pd.Timedelta(minutes=10)],
        "stage": ["Deep", "REM"],
        "heart_rate": [58.0, 60.0],
        "segment": [0, 0],
    })
    highlights = timeline_highlights(flagged(
        ("deep_sleep_min", "Deep sleep", "min", 48, 73, "7_days"),
        ("sleeping_hr_avg_bpm", "Sleeping heart rate", "bpm", 66, 58, "7_days"),
    ))
    spec = timeline_chart(night, highlights).to_dict()
    assert len(spec["layer"]) == 2
