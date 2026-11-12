"""Tests for the baseline windows and change flags in scripts/build_change_detection_table.py.

Uses small made-up nights, never participant data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_change_detection_table import build_change_table  # noqa: E402


def write_participant(folder, participant_id, nights):
    """Write one aligned CSV with two 5-minute rows per night.

    Each night is a dict with total sleep minutes ("total"), optional
    "deep", "hr" (bpm while asleep) and "temp" (temperature delta).
    """
    rows = []
    for index, night in enumerate(nights):
        date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
        for _ in range(2):
            rows.append({
                "date": date,
                "night_total_sleep_seconds": night["total"] * 60,
                "night_deep_sleep_seconds": night.get("deep", 60) * 60,
                "night_rem_sleep_seconds": 90 * 60,
                "night_light_sleep_seconds": 200 * 60,
                "night_awake_seconds": 30 * 60,
                "temperature_delta": night.get("temp", 0.0),
                "heart_rate": night.get("hr", 60),
                "hypnogram_class": "deep",
            })
    pd.DataFrame(rows).to_csv(folder / f"{participant_id}_aligned.csv", index=False)


def build(tmp_path, nights, participant_id="par_1"):
    write_participant(tmp_path, participant_id, nights)
    return build_change_table(tmp_path, tmp_path / "no_awakenings_file.csv")


def pick(table, metric, window, night_index):
    date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=night_index)).strftime("%Y-%m-%d")
    rows = table[
        table["metric"].eq(metric)
        & table["baseline_window"].eq(window)
        & table["night_date"].eq(date)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


TOTALS = [400, 415, 405, 420, 410, 480]


def test_first_night_has_no_history_in_any_window(tmp_path):
    table = build(tmp_path, [{"total": t} for t in TOTALS])
    for window in ["1_day", "7_days", "14_days", "30_days"]:
        row = pick(table, "total_sleep_min", window, 0)
        assert row["status"] == "insufficient_history"
        assert row["baseline_nights"] == 0


def test_previous_night_flags_a_difference_of_ten_minutes_or_more(tmp_path):
    table = build(tmp_path, [{"total": t} for t in TOTALS])
    assert pick(table, "total_sleep_min", "1_day", 1)["status"] == "change"   # +15 min
    row = pick(table, "total_sleep_min", "1_day", 2)                         # -10 min, on the line
    assert row["status"] == "change"
    assert row["difference"] == pytest.approx(-10.0)


def test_small_previous_night_difference_is_normal(tmp_path):
    table = build(tmp_path, [{"total": 400}, {"total": 405}])
    row = pick(table, "total_sleep_min", "1_day", 1)
    assert row["status"] == "normal"
    assert row["difference"] == pytest.approx(5.0)


def test_seven_night_window_needs_three_prior_nights(tmp_path):
    table = build(tmp_path, [{"total": t} for t in TOTALS])
    assert pick(table, "total_sleep_min", "7_days", 2)["status"] == "insufficient_history"
    assert pick(table, "total_sleep_min", "7_days", 3)["status"] != "insufficient_history"


def test_baseline_uses_only_earlier_nights(tmp_path):
    table = build(tmp_path, [{"total": t} for t in TOTALS])
    row = pick(table, "total_sleep_min", "7_days", 5)
    assert row["baseline_nights"] == 5
    assert row["baseline_value"] == pytest.approx(sum(TOTALS[:5]) / 5)
    assert row["observed_value"] == pytest.approx(480)


def test_z_score_of_two_or_more_is_a_change(tmp_path):
    table = build(tmp_path, [{"total": t} for t in TOTALS])
    unusual = pick(table, "total_sleep_min", "7_days", 5)
    assert unusual["status"] == "change"
    assert abs(unusual["z_score"]) >= 2
    typical = pick(table, "total_sleep_min", "7_days", 4)
    assert typical["status"] == "normal"
    assert abs(typical["z_score"]) < 2


def test_heart_rate_and_temperature_use_their_own_thresholds(tmp_path):
    nights = [
        {"total": 400, "hr": 60, "temp": 0.0},
        {"total": 400, "hr": 66, "temp": 0.4},
        {"total": 400, "hr": 68, "temp": 1.0},
    ]
    table = build(tmp_path, nights)
    assert pick(table, "sleeping_hr_avg_bpm", "1_day", 1)["status"] == "change"   # +6 bpm >= 5
    assert pick(table, "sleeping_hr_avg_bpm", "1_day", 2)["status"] == "normal"   # +2 bpm
    assert pick(table, "temperature_delta_c", "1_day", 1)["status"] == "normal"   # +0.4 °C < 0.5
    assert pick(table, "temperature_delta_c", "1_day", 2)["status"] == "change"   # +0.6 °C


def test_missing_current_measurement_is_reported_as_missing(tmp_path):
    nights = [{"total": 400, "temp": 0.1}, {"total": 400, "temp": float("nan")}]
    table = build(tmp_path, nights)
    row = pick(table, "temperature_delta_c", "1_day", 1)
    assert row["status"] == "missing"
    assert pd.isna(row["difference"])


def test_participants_are_compared_only_with_themselves(tmp_path):
    write_participant(tmp_path, "par_1", [{"total": 400}, {"total": 400}])
    write_participant(tmp_path, "par_2", [{"total": 600}, {"total": 600}])
    table = build_change_table(tmp_path, tmp_path / "no_awakenings_file.csv")
    for participant in ["par_1", "par_2"]:
        rows = table[
            table["participant_id"].eq(participant)
            & table["metric"].eq("total_sleep_min")
            & table["baseline_window"].eq("1_day")
            & table["night_date"].eq("2026-01-02")
        ]
        assert rows.iloc[0]["status"] == "normal"
        assert rows.iloc[0]["difference"] == pytest.approx(0.0)


def test_no_aligned_files_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_change_table(tmp_path, tmp_path / "no_awakenings_file.csv")
