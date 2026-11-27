"""Tests for the history view helpers in frontend/history.py (#74, #80)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from history import history_rows, recent_nights_summary, threshold_band, trend_chart  # noqa: E402


def row(night, metric="total_sleep_min", status="normal", observed=420.0, baseline=410.0, std=10.0,
        window="7_days", threshold_type="absolute_z_score", threshold=2.0, participant="par_1"):
    return {
        "participant_id": participant, "night_date": night, "metric": metric, "metric_label": None,
        "unit": "min", "observed_value": observed, "baseline_value": baseline, "baseline_std": std,
        "baseline_window": window, "threshold_type": threshold_type, "threshold_value": threshold,
        "status": status,
    }


def nights(n, start="2026-01-01"):
    return [(pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


LABELS = {"total_sleep_min": "Total sleep", "deep_sleep_min": "Deep sleep", "awake_min": "Awake time"}


def test_band_matches_the_flag_rules():
    rows = pd.DataFrame([
        row("2026-01-01", threshold_type="absolute_z_score", threshold=2.0, baseline=410, std=10),
        row("2026-01-02", threshold_type="absolute_difference", threshold=10.0, baseline=400, std=None),
        row("2026-01-03", status="insufficient_history"),
        row("2026-01-04", std=0.0),  # no spread: never flagged, so no band
    ])
    lower, upper = threshold_band(rows)
    assert (lower[0], upper[0]) == (390, 430)
    assert (lower[1], upper[1]) == (390, 410)
    assert pd.isna(lower[2]) and pd.isna(upper[2])
    assert pd.isna(lower[3])


def test_history_rows_filter_sort_and_mark_flags():
    data = pd.DataFrame([
        row("2026-01-03", status="change"),
        row("2026-01-01"),
        row("2026-01-02", status="insufficient_history"),
        row("2026-01-02", window="14_days"),
        row("2026-01-02", participant="par_2"),
        row("2026-01-02", metric="deep_sleep_min"),
    ])
    rows = history_rows(data, "par_1", "7_days", "total_sleep_min")
    assert list(rows["night_date"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert list(rows["flagged"]) == [False, False, True]
    assert pd.isna(rows.loc[1, "baseline_shown"])  # no comparison on a short-history night
    assert list(rows["status_name"]) == ["Within range", "Not enough history", "Flagged change"]


def test_history_rows_respect_the_date_range():
    data = pd.DataFrame([row(night) for night in nights(10)])
    rows = history_rows(data, "par_1", "7_days", "total_sleep_min", start="2026-01-03", end="2026-01-05")
    assert list(rows["night_date"]) == ["2026-01-03", "2026-01-04", "2026-01-05"]


def test_chart_builds():
    data = pd.DataFrame([row(night, status=s) for night, s in zip(nights(3), ["normal", "change", "insufficient_history"])])
    spec = trend_chart(history_rows(data, "par_1", "7_days", "total_sleep_min"), "Total sleep", "min").to_dict()
    assert len(spec["layer"]) == 4


def test_summary_counts_repeated_flags_in_the_last_seven_nights():
    days = nights(9)
    data = pd.DataFrame(
        [row(d) for d in days]
        + [row(d, metric="deep_sleep_min", status="change" if i in (3, 5, 8) else "normal") for i, d in enumerate(days)]
        + [row(d, metric="awake_min", status="change" if i == 7 else "normal") for i, d in enumerate(days)]
        # A flag before the 7-night span must not count.
        + [row(days[0], metric="awake_min", status="change")]
    )
    summary = recent_nights_summary(data, "par_1", "7_days", days[-1], LABELS)
    assert summary.headline == "Over your last 7 nights, Deep sleep stood out more than once compared with your previous 7 nights."
    assert summary.details[0] == "Deep sleep was flagged on 3 of the last 7 nights."
    assert summary.details[1] == "Awake time was flagged on 1 of the last 7 nights."


def test_summary_ends_at_the_chosen_night():
    days = nights(9)
    data = pd.DataFrame([row(d, status="change" if i == 8 else "normal") for i, d in enumerate(days)])
    summary = recent_nights_summary(data, "par_1", "7_days", days[7], LABELS)
    assert summary.headline == "Over your last 7 nights, nothing was flagged compared with your previous 7 nights."


def test_summary_reports_short_history_and_missing_nights():
    days = nights(4)
    statuses = ["insufficient_history", "insufficient_history", "missing", "normal"]
    data = pd.DataFrame([row(d, status=s) for d, s in zip(days, statuses)])
    summary = recent_nights_summary(data, "par_1", "7_days", days[-1], LABELS)
    assert summary.headline.startswith("Over your last 4 nights, nothing was flagged")
    assert "2 of these nights didn't have enough earlier nights to compare yet." in summary.details
    assert "Some measurements were missing on 1 night." in summary.details


@pytest.mark.parametrize("statuses, expected", [
    (["insufficient_history"] * 3, "Over your last 3 nights, there wasn't enough history to compare yet."),
    (["normal", "change", "normal"], "Over your last 3 nights, 1 measurement stood out on one night each compared with your previous 7 nights."),
])
def test_summary_headline_variants(statuses, expected):
    data = pd.DataFrame([row(d, status=s) for d, s in zip(nights(3), statuses)])
    assert recent_nights_summary(data, "par_1", "7_days", nights(3)[-1], LABELS).headline == expected


def test_summary_with_no_nights():
    data = pd.DataFrame([row("2026-01-05")])
    assert recent_nights_summary(data, "par_1", "7_days", "2026-01-01").headline == "No nights to summarize yet."
