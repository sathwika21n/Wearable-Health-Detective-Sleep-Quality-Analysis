"""Tests for the rule-based night summaries in frontend/insights.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from insights import DISCLAIMER, describe, format_amount, night_facts, summarize  # noqa: E402


def row(metric, status, observed, baseline=None, difference=None, unit="min", z_score=None,
        window="7_days", threshold_type="absolute_z_score", threshold_value=2.0,
        baseline_nights=7, minimum_nights=3, label=None):
    return {
        "metric": metric, "metric_label": label or metric, "unit": unit, "status": status,
        "baseline_window": window, "observed_value": observed, "baseline_value": baseline,
        "difference": difference, "z_score": z_score, "threshold_type": threshold_type,
        "threshold_value": threshold_value, "baseline_nights": baseline_nights,
        "minimum_nights": minimum_nights,
    }


LABELS = {
    "deep_sleep_min": "Deep sleep",
    "rem_sleep_min": "REM sleep",
    "total_sleep_min": "Total sleep",
    "sleeping_hr_avg_bpm": "Sleeping heart rate",
    "temperature_delta_c": "Temperature deviation",
    "observed_awakenings_count": "Observed awakenings",
}


def facts_for(*rows):
    return night_facts(pd.DataFrame(list(rows)), LABELS)


def test_format_amount_units():
    assert format_amount(25, "min") == "25 min"
    assert format_amount(85, "min") == "1 h 25 min"
    assert format_amount(120, "min") == "2 h"
    assert format_amount(4.25, "bpm") == "4.2 bpm"
    assert format_amount(0.62, "°C") == "+0.62 °C"
    assert format_amount(-0.001, "°C") == "0.00 °C"
    assert format_amount(3.0, "count") == "3"


def test_describe_change_uses_direction_and_window():
    fact = facts_for(row("deep_sleep_min", "change", 48, 73, -25, z_score=-2.4))[0]
    assert describe(fact) == (
        "Deep sleep: 25 min shorter than your average over the previous 7 nights (48 min vs 1 h 13 min)."
    )


def test_describe_one_night_comparison_and_counts():
    fact = facts_for(row(
        "observed_awakenings_count", "change", 5, 2, 3, unit="count", window="1_day",
        threshold_type="absolute_difference", threshold_value=10.0,
    ))[0]
    assert describe(fact) == "Observed awakenings: 3 more than the night before (5 vs 2)."


def test_describe_other_statuses():
    normal, missing, short = facts_for(
        row("total_sleep_min", "normal", 432, 420, 12, z_score=0.5),
        row("sleeping_hr_avg_bpm", "missing", None, unit="bpm"),
        row("rem_sleep_min", "insufficient_history", 90, baseline_nights=2),
    )
    assert describe(normal) == "Total sleep: within your usual range (7 h 12 min)."
    assert describe(missing) == "Sleeping heart rate: not recorded this night."
    assert describe(short) == "REM sleep: not enough history yet to compare (2 of 3 prior nights)."


def test_summary_ranks_strongest_change_first_and_notes_same_night_association():
    summary = summarize(facts_for(
        row("rem_sleep_min", "change", 60, 95, -35, z_score=-2.1),
        row("sleeping_hr_avg_bpm", "change", 62, 55, 7, unit="bpm", z_score=3.2),
        row("total_sleep_min", "normal", 430, 425, 5, z_score=0.3),
    ))
    assert summary.headline == (
        "2 measurements stood out compared with your previous 7 nights: Sleeping heart rate and REM sleep."
    )
    assert summary.details[0].startswith("Sleeping heart rate: 7 bpm higher")
    association = summary.details[-1]
    assert "REM sleep happened on the same night as the change in sleeping heart rate" in association
    assert "does not show that one caused the other" in association


def test_summary_limits_highlights_and_counts_the_rest():
    rows = [row(f"m{i}", "change", 10 + i, 5, 5 + i, unit="ms", z_score=2.0 + i, label=f"Metric {i}") for i in range(5)]
    summary = summarize(facts_for(*rows))
    assert summary.headline.endswith("Metric 4, Metric 3, Metric 2 and 2 more.")
    assert len(summary.details) == 3


def test_summary_for_quiet_night_and_unavailable_data():
    quiet = summarize(facts_for(
        row("total_sleep_min", "normal", 430, 425, 5, z_score=0.3),
        row("rem_sleep_min", "insufficient_history", 90, baseline_nights=1),
    ))
    assert quiet.headline.startswith("Nothing stood out")
    assert quiet.details == ["1 measurement could not be compared (missing data or not enough prior nights)."]

    no_history = summarize(facts_for(row("rem_sleep_min", "insufficient_history", 90, baseline_nights=0)))
    assert no_history.headline == "There is not enough data to compare this night yet."


def test_empty_night_and_disclaimer_wording():
    assert summarize([]).headline == "No measurements are available for this night."
    assert "not medical advice" in DISCLAIMER
