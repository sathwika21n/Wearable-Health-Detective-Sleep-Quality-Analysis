"""Tests for dashboard comparisons and temperature baselines."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import charts  # noqa: E402


def test_comparison_pairs_excludes_missing_values_and_keeps_same_night():
    rows = pd.DataFrame([
        ("2026-01-01", "total_sleep_min", 400),
        ("2026-01-01", "temperature_delta_c", 0.1),
        ("2026-01-02", "total_sleep_min", 420),
        ("2026-01-02", "temperature_delta_c", None),
        ("2026-01-03", "total_sleep_min", float("inf")),
        ("2026-01-03", "temperature_delta_c", 0.3),
    ], columns=["night_date", "metric", "observed_value"])

    pairs = charts.comparison_pairs(rows, "total_sleep_min", "temperature_delta_c")

    assert pairs.to_dict("records") == [
        {"night_date": "2026-01-01", "total_sleep_min": 400.0, "temperature_delta_c": 0.1}
    ]
    assert charts.comparison_pairs(rows, "total_sleep_min", "sleeping_hr_avg_bpm").empty


def test_comparison_pairs_rejects_duplicate_metric_nights():
    rows = pd.DataFrame([
        ("2026-01-01", "total_sleep_min", 400),
        ("2026-01-01", "total_sleep_min", 410),
    ], columns=["night_date", "metric", "observed_value"])
    with pytest.raises(ValueError, match="Duplicate metric"):
        charts.comparison_pairs(rows, "total_sleep_min", "temperature_delta_c")


def test_temperature_baselines_load_from_outputs(monkeypatch):
    expected = charts.PROJECT_DIR / "outputs" / "participant_baselines.csv"
    monkeypatch.setattr(Path, "exists", lambda path: path == expected)
    monkeypatch.setattr(charts.pd, "read_csv", lambda path: pd.DataFrame({
        "participant_id": ["par_1"],
        "sleep_date": ["2026-01-01"],
        "typical_temperature_delta_c": [0.2],
    }))

    result = charts.load_temperature_baselines()

    assert result is not None
    assert result.loc[0, "typical_temperature_delta_c"] == 0.2


def test_dashboard_comparison_selects_metric_and_participant():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    tabs = {tab.label: tab for tab in app.get("tab")}
    assert set(tabs) == {"Overview", "Metric details", "Night timeline", "Temperature trend", "Compare metrics"}
    assert any("What stood out" in item.value for item in tabs["Overview"].markdown)
    assert any("Changes to review" in item.value for item in tabs["Metric details"].markdown)
    assert any("Night timeline" in item.value for item in tabs["Night timeline"].markdown)
    assert any("Overall nightly temperature delta" in item.value for item in tabs["Temperature trend"].markdown)
    assert any("Compare nightly metrics" in item.value for item in tabs["Compare metrics"].markdown)
    comparison = next(widget for widget in app.selectbox if widget.label == "Choose two measurements to compare")
    comparison.select("Total sleep vs. temperature deviation").run(timeout=30)
    assert not app.exception
    assert any("Paired nights:" in caption.value for caption in app.caption)
    assert any("association" in caption.value and "cause" in caption.value for caption in app.caption)
    participant = next(widget for widget in app.selectbox if widget.label == "Participant")
    if len(participant.options) > 1:
        participant.select(participant.options[1]).run(timeout=30)
        assert not app.exception
        assert any("Paired nights:" in caption.value for caption in app.caption)


def test_comparison_shows_paired_count_and_correlation():
    app = AppTest.from_string("""
import pandas as pd
from charts import show_metric_comparison

data = pd.DataFrame([
    ("2026-01-01", "total_sleep_min", 400),
    ("2026-01-01", "sleeping_hr_avg_bpm", 50),
    ("2026-01-02", "total_sleep_min", 420),
    ("2026-01-02", "sleeping_hr_avg_bpm", 55),
    ("2026-01-03", "total_sleep_min", 440),
    ("2026-01-03", "sleeping_hr_avg_bpm", 60),
    ("2026-01-04", "total_sleep_min", 460),
    ("2026-01-04", "sleeping_hr_avg_bpm", None),
], columns=["night_date", "metric", "observed_value"])
show_metric_comparison(data, "2026-01-03")
""")
    app.run(timeout=30)
    assert not app.exception
    assert any("Paired nights: 3" in caption.value for caption in app.caption)
    assert any("Pearson correlation r = 1.00" in caption.value for caption in app.caption)
    assert any("orange diamond marks the selected night" in caption.value for caption in app.caption)
    assert any("dashed line" in caption.value for caption in app.caption)
    assert any("tends to be higher" in markdown.value for markdown in app.markdown)


def test_comparison_explains_when_selected_night_is_unavailable():
    app = AppTest.from_string("""
import pandas as pd
from charts import show_metric_comparison

data = pd.DataFrame([
    ("2026-01-01", "total_sleep_min", 400),
    ("2026-01-01", "sleeping_hr_avg_bpm", 50),
    ("2026-01-02", "total_sleep_min", 420),
], columns=["night_date", "metric", "observed_value"])
show_metric_comparison(data, "2026-01-02")
""")
    app.run(timeout=30)
    assert not app.exception
    assert any("selected night is not shown" in caption.value for caption in app.caption)
    assert any("at least 3 paired nights" in info.value for info in app.info)


def test_temperature_trend_marks_supplied_change_even_inside_reference_band():
    nights = pd.DataFrame([
        ("2026-01-01", 0.1, 0.0, "change"),
        ("2026-01-02", 0.7, 0.0, "normal"),
    ], columns=["night_date", "observed_value", "baseline_value", "status"])
    spec = charts.temperature_trend_chart(nights, 0.5).to_dict()

    assert spec["layer"][-1]["transform"][0]["filter"] == "(datum.status === 'change')"
    records = next(iter(spec["datasets"].values()))
    assert [(row["observed_value"], row["status"]) for row in records] == [
        (0.1, "change"), (0.7, "normal")
    ]


@pytest.mark.parametrize(
    ("status", "element", "expected"),
    [
        ("change", "warning", "Temperature change detected"),
        ("normal", "success", "No temperature change was detected"),
        ("missing", "info", "measurement missing"),
        ("insufficient_history", "info", "not enough baseline history"),
    ],
)
def test_selected_temperature_status(status, element, expected):
    app = AppTest.from_string(f"""
import pandas as pd
from charts import show_selected_temperature_status
row = pd.Series({{
    "status": {status!r}, "observed_value": 0.1, "baseline_value": 0.0,
    "difference": 0.1, "threshold_type": "absolute_z_score",
    "threshold_value": 2.0, "z_score": 2.5,
}})
show_selected_temperature_status(row)
""")
    app.run(timeout=30)
    assert not app.exception
    message = getattr(app, element)[0].value
    assert expected in message
    if status == "change":
        assert "selected-window baseline +0.00 °C" in message
        assert "z-score +2.50" in message
        assert "flag rule |z-score| ≥ 2" in message


def test_temperature_trend_explains_selected_night_and_counts_flagged_nights():
    app = AppTest.from_string("""
import pandas as pd
from charts import show_temperature_trend
data = pd.DataFrame([
    ("2026-01-01", 0.1, 0.0, 0.1, "change", 2.5),
    ("2026-01-02", 0.7, 0.0, 0.7, "normal", 0.5),
    ("2026-01-03", 0.2, 0.0, 0.2, "change", 3.0),
], columns=["night_date", "observed_value", "baseline_value", "difference", "status", "z_score"])
data["participant_id"] = "p1"
data["metric"] = "temperature_delta_c"
data["baseline_window"] = "7_days"
data["threshold_type"] = "absolute_z_score"
data["threshold_value"] = 2.0
show_temperature_trend(data, "p1", "7_days", "2026-01-03", False)
""")
    app.run(timeout=30)
    assert not app.exception
    assert "Temperature change detected" in app.warning[0].value
    assert "difference +0.20 °C" in app.warning[0].value
    assert any("Red diamonds mark 2 night(s)" in caption.value for caption in app.caption)
    assert any("not necessarily the selected window's flag rule" in caption.value for caption in app.caption)
