"""Exercise the Apple importer with invented Health export records."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_change_detection_table import build_change_table
from import_apple_health import HEART_RATE, SDNN, SLEEP_TYPE, TEMPERATURE, import_export
from nightly_sleep_features import build_nightly_features


@pytest.fixture
def apple_export(tmp_path):
    root = ET.Element("HealthData")

    def record(kind, start, end, value, unit=None):
        attrs = {
            "type": kind, "startDate": f"2025-01-{start}:00 -0600",
            "endDate": f"2025-01-{end}:00 -0600", "value": str(value),
        }
        if unit:
            attrs["unit"] = unit
        ET.SubElement(root, "Record", attrs)

    def sleep(start, end, stage):
        record(SLEEP_TYPE, start, end, f"HKCategoryValueSleepAnalysis{stage}")

    sleep("10 23:55", "11 00:40", "InBed")
    sleep("11 00:00", "11 00:30", "AsleepUnspecified")
    sleep("11 00:00", "11 00:10", "AsleepCore")
    sleep("11 00:00", "11 00:10", "AsleepCore")
    sleep("11 00:10", "11 00:15", "Awake")
    sleep("11 00:15", "11 00:20", "AsleepDeep")
    sleep("11 00:20", "11 00:25", "AsleepREM")
    sleep("11 00:35", "11 00:40", "Awake")
    record(HEART_RATE, "11 00:01", "11 00:01", 60, "count/min")
    record(HEART_RATE, "11 00:04", "11 00:04", 64, "count/min")
    record(HEART_RATE, "11 00:12", "11 00:12", 90, "count/min")
    record(HEART_RATE, "11 00:41", "11 00:41", 99, "count/min")
    record(SDNN, "11 00:02", "11 00:02", 0.04, "s")
    record(TEMPERATURE, "10 23:50", "11 00:30", 96.8, "degF")
    sleep("11 13:00", "11 13:15", "AsleepCore")  # A nap loses to the main session.
    sleep("12 00:00", "12 00:20", "AsleepUnspecified")
    sleep("13 00:00", "13 00:10", "AsleepCore")
    sleep("13 00:00", "13 00:05", "AsleepDeep")  # Conflicting detailed stages.
    path = tmp_path / "export.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_cli_output_and_downstream_readers(apple_export, tmp_path):
    output = tmp_path / "aligned"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/import_apple_health.py"),
        str(apple_export), "--output-dir", str(output),
    ], check=True)
    aligned = pd.read_csv(output / "par_apple_aligned.csv")
    awakenings_path = output / "nightly_awakenings.csv"
    awakenings = pd.read_csv(awakenings_path).set_index("date")

    first = aligned[aligned["date"].eq("2025-01-11")].reset_index(drop=True)
    assert first["datetime_utc"].iloc[0] == "2025-01-11 05:55:00"
    assert first["sleep_end_utc"].iloc[0] == "2025-01-11 06:40:00"
    assert pd.to_datetime(first["datetime_utc"]).diff().dropna().eq(pd.Timedelta(minutes=5)).all()
    assert first["hypnogram_class"].fillna("").tolist() == [
        "", "light", "light", "awake", "deep", "REM", "", "", "awake",
    ]
    assert first["heart_rate"].dropna().tolist() == [62, 90]
    assert first["heart_sdnn_ms"].dropna().tolist() == [40]
    assert first["wrist_temperature_c"].tolist() == pytest.approx([36] * len(first))
    assert first["night_hr_average"].iloc[0] == 62
    assert first["night_total_sleep_seconds"].iloc[0] == 1500
    assert first["night_time_in_bed_seconds"].iloc[0] == 2700
    assert aligned[["heart_rmssd", "night_hrv_rmssd", "temperature_delta"]].isna().all().all()
    assert awakenings.loc["2025-01-11", "awakening_count"] == 1
    assert awakenings.loc["2025-01-11", "total_awake_minutes"] == 10

    features = build_nightly_features(output).set_index("sleep_date")
    assert features.index.tolist() == ["2025-01-11", "2025-01-12", "2025-01-13"]
    assert features.loc["2025-01-11", "total_sleep_time_min"] == 25
    assert features.loc["2025-01-11", "observed_awakenings_count"] == 1
    assert features.loc["2025-01-11", "hrv_sdnn_ms"] == 40
    assert features.loc["2025-01-11", "wrist_temperature_c"] == pytest.approx(36)
    assert features.loc["2025-01-12", "total_sleep_time_min"] == 20
    assert pd.isna(features.loc["2025-01-12", "deep_sleep_min"])
    assert pd.isna(awakenings.loc["2025-01-12", "awakening_count"])
    assert features.loc["2025-01-13", "total_sleep_time_min"] == 5

    changes = build_change_table(output, awakenings_path)
    rows = changes[(changes["night_date"] == "2025-01-11") & (changes["baseline_window"] == "1_day")]
    values = rows.set_index("metric")["observed_value"]
    assert set(changes["participant_id"]) == {"par_apple"}
    assert values["total_sleep_min"] == 25
    assert values["observed_awakenings_count"] == 1
    assert values["sleeping_hr_avg_bpm"] == 62
    assert pd.isna(values["temperature_delta_c"])
    assert aligned.loc[aligned["date"].eq("2025-01-12"),
                       ["heart_rate", "heart_sdnn_ms", "wrist_temperature_c"]].isna().all().all()


def test_empty_export_fails_without_producing_files(tmp_path):
    export = tmp_path / "export.xml"
    export.write_text("<HealthData/>", encoding="utf-8")
    output = tmp_path / "aligned"
    with pytest.raises(ValueError, match="No usable sleep sessions"):
        import_export(export, output)
    assert not output.exists()


def test_wake_date_uses_export_offset_and_tied_epochs_stay_unlabeled(tmp_path):
    source = tmp_path / "export.xml"
    source.write_text(
        f"""<HealthData>
          <Record type="{SLEEP_TYPE}" value="HKCategoryValueSleepAnalysisAsleepCore"
                  startDate="2025-01-11 00:00:00 +0900" endDate="2025-01-11 00:02:30 +0900"/>
          <Record type="{SLEEP_TYPE}" value="HKCategoryValueSleepAnalysisAsleepDeep"
                  startDate="2025-01-11 00:02:30 +0900" endDate="2025-01-11 00:05:00 +0900"/>
          <Record type="{HEART_RATE}" unit="count/min" value="malformed"
                  startDate="2025-01-11 00:00:00 +0900" endDate="2025-01-11 00:00:00 +0900"/>
        </HealthData>""",
        encoding="utf-8",
    )
    aligned_path, _ = import_export(source, tmp_path / "aligned")
    row = pd.read_csv(aligned_path).iloc[0]
    assert row["date"] == "2025-01-11"
    assert row["datetime_utc"] == "2025-01-10 15:00:00"
    assert row["night_total_sleep_seconds"] == 300
    assert row["night_deep_sleep_seconds"] == 150
    assert pd.isna(row["hypnogram_class"])
    assert pd.isna(row["heart_rate"])


@pytest.mark.parametrize("has_temperature", [False, True])
def test_dashboard_apple_source(apple_export, tmp_path, monkeypatch, has_temperature):
    monkeypatch.syspath_prepend(str(ROOT / "frontend"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    output = tmp_path / "aligned"
    monkeypatch.setenv("APPLE_HEALTH_OUTPUT_DIR", str(output))
    _, awakening_path = import_export(apple_export, output)
    features = build_nightly_features(output)
    if not has_temperature:
        features["wrist_temperature_c"] = None
    features.to_csv(output / "nightly_sleep_features.csv", index=False)
    build_change_table(output, awakening_path).to_csv(output / "change_detection_results.csv", index=False)
    app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()

    def widget(label):
        return next(item for item in app.selectbox if item.label == label)

    widget("Data source").select("Apple Health / Apple Watch").run()
    widget("Night").select("2025-01-11").run()
    assert not app.exception
    assert widget("Participant").options == ["par_apple"]
    assert any(metric.label == "Heart rate variability (SDNN)" and metric.value == "40 ms" for metric in app.metric)
    assert not any("RMSSD" in metric.label for metric in app.metric)
    assert all("temperature deviation" not in name for name in widget("Choose two measurements to compare").options)
    tabs = {tab.label: tab for tab in app.get("tab")}
    assert tabs["Night timeline"].get("arrow_vega_lite_chart")
    if has_temperature:
        assert tabs["Temperature trend"].get("arrow_vega_lite_chart")
    else:
        assert any("Wrist temperature is unavailable" in info.value for info in tabs["Temperature trend"].info)
    assert any("Rule-based summary" in item.value for item in app.caption)
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert any("Rule-based (AI unavailable)" in item.value for item in app.caption)
    widget("Data source").select("Oura / public dataset").run()
    assert not app.exception
    assert "par_apple" not in widget("Participant").options


def test_dashboard_missing_apple_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "frontend"))
    monkeypatch.setenv("APPLE_HEALTH_OUTPUT_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "frontend/app.py"), default_timeout=30).run()
    source = next(item for item in app.selectbox if item.label == "Data source")
    source.select("Apple Health / Apple Watch").run()
    assert not app.exception
    assert any("Apple Health results are unavailable" in info.value for info in app.info)
