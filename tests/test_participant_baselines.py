"""Tests for the prior-10-night baselines in scripts/participant_baselines.py.

Uses small made-up nights, never participant data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from participant_baselines import LOOKBACK_NIGHTS, build_prior_night_baselines  # noqa: E402


def write_inputs(folder, nights_by_participant):
    """nights_by_participant: {participant_id: [(total_sleep_min, heart_rate, temperature), ...]}."""
    sleep_rows, physiology_rows = [], []
    for participant_id, nights in nights_by_participant.items():
        for index, (total, heart_rate, temperature) in enumerate(nights):
            date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)).strftime("%Y-%m-%d")
            sleep_rows.append({"participant_id": participant_id, "sleep_date": date, "total_sleep_time_min": total})
            physiology_rows.append({
                "participant_id": participant_id, "sleep_date": date,
                "average_heart_rate_bpm": heart_rate, "temperature_delta_c": temperature,
            })
    sleep_path = folder / "sleep.csv"
    physiology_path = folder / "physiology.csv"
    pd.DataFrame(sleep_rows).to_csv(sleep_path, index=False)
    pd.DataFrame(physiology_rows).to_csv(physiology_path, index=False)
    return sleep_path, physiology_path


def test_first_night_has_no_baseline(tmp_path):
    paths = write_inputs(tmp_path, {"par_1": [(400, 60, 0.1), (420, 62, 0.2)]})
    baselines = build_prior_night_baselines(*paths)
    first = baselines.iloc[0]
    assert first["prior_nights_count"] == 0
    assert pd.isna(first["typical_sleep_time_min"])


def test_baseline_is_the_mean_of_earlier_nights_only(tmp_path):
    paths = write_inputs(tmp_path, {"par_1": [(400, 60, 0.0), (420, 64, 0.2), (500, 90, 1.0)]})
    third = build_prior_night_baselines(*paths).iloc[2]
    assert third["prior_nights_count"] == 2
    assert third["typical_sleep_time_min"] == pytest.approx(410)
    assert third["typical_heart_rate_bpm"] == pytest.approx(62)
    assert third["typical_temperature_delta_c"] == pytest.approx(0.1)


def test_baseline_looks_back_at_most_ten_nights(tmp_path):
    totals = list(range(400, 400 + 12 * 10, 10))  # 12 nights: 400, 410, ... 510
    paths = write_inputs(tmp_path, {"par_1": [(total, 60, 0.0) for total in totals]})
    last = build_prior_night_baselines(*paths).iloc[-1]
    assert last["prior_nights_count"] == LOOKBACK_NIGHTS == 10
    assert last["typical_sleep_time_min"] == pytest.approx(sum(totals[1:11]) / 10)


def test_participants_do_not_share_baselines(tmp_path):
    paths = write_inputs(tmp_path, {
        "par_1": [(400, 60, 0.0), (400, 60, 0.0)],
        "par_2": [(600, 80, 0.5), (600, 80, 0.5)],
    })
    baselines = build_prior_night_baselines(*paths).set_index(["participant_id", "sleep_date"])
    assert baselines.loc[("par_2", "2026-01-02"), "typical_sleep_time_min"] == pytest.approx(600)
    assert pd.isna(baselines.loc[("par_2", "2026-01-01"), "typical_sleep_time_min"])


def test_duplicate_nights_are_rejected(tmp_path):
    sleep_path, physiology_path = write_inputs(tmp_path, {"par_1": [(400, 60, 0.0)]})
    sleep = pd.read_csv(sleep_path)
    pd.concat([sleep, sleep]).to_csv(sleep_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        build_prior_night_baselines(sleep_path, physiology_path)


def test_missing_columns_are_rejected(tmp_path):
    sleep_path, physiology_path = write_inputs(tmp_path, {"par_1": [(400, 60, 0.0)]})
    pd.read_csv(sleep_path).drop(columns="total_sleep_time_min").to_csv(sleep_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        build_prior_night_baselines(sleep_path, physiology_path)
