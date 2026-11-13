"""Tests for the dashboard's results loader in frontend/data_loader.py."""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from data_loader import SAMPLE_PATH, load_dashboard_data  # noqa: E402


def results_row(**overrides):
    row = {
        "participant_id": "par_1", "night_date": "2026-01-05", "metric": "total_sleep_min",
        "unit": "min", "observed_value": 420, "baseline_value": 410, "difference": 10,
        "baseline_window": "7_days", "baseline_nights": 4, "minimum_nights": 3,
        "status": "normal", "message": "",
    }
    row.update(overrides)
    return row


def write_results(path, *rows):
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def test_generated_results_are_used_when_present(tmp_path):
    generated = write_results(tmp_path / "results.csv", results_row())
    frame, source = load_dashboard_data(generated_path=generated)
    assert "generated project results" in source
    assert len(frame) == 1


def test_sample_data_is_used_when_results_are_missing(tmp_path):
    frame, source = load_dashboard_data(generated_path=tmp_path / "not_built_yet.csv")
    assert "sample fallback" in source
    assert SAMPLE_PATH.name in source
    assert not frame.empty


def test_uploaded_file_takes_priority(tmp_path):
    generated = write_results(tmp_path / "results.csv", results_row())
    upload = io.BytesIO(pd.DataFrame([results_row(), results_row(metric="deep_sleep_min")]).to_csv(index=False).encode())
    upload.name = "my_upload.csv"
    frame, source = load_dashboard_data(upload, generated_path=generated)
    assert source == "my_upload.csv"
    assert len(frame) == 2


def test_missing_required_column_is_reported(tmp_path):
    row = results_row()
    del row["baseline_window"]
    generated = write_results(tmp_path / "results.csv", row)
    with pytest.raises(ValueError, match="baseline_window"):
        load_dashboard_data(generated_path=generated)


def test_unknown_status_is_reported(tmp_path):
    generated = write_results(tmp_path / "results.csv", results_row(status="alarm"))
    with pytest.raises(ValueError, match="alarm"):
        load_dashboard_data(generated_path=generated)


def test_blank_measurement_is_shown_as_missing_unless_history_is_short(tmp_path):
    generated = write_results(
        tmp_path / "results.csv",
        results_row(observed_value=None, status="normal"),
        results_row(metric="deep_sleep_min", observed_value=None, status="insufficient_history"),
    )
    frame, _ = load_dashboard_data(generated_path=generated)
    statuses = dict(zip(frame["metric"], frame["status"]))
    assert statuses == {"total_sleep_min": "missing", "deep_sleep_min": "insufficient_history"}
