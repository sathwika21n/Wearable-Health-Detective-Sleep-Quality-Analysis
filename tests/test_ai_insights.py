"""Tests for the optional AI feedback in frontend/ai_insights.py.

A fake OpenAI client stands in for the API, so these run in CI without a key
and never make a network call.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

import ai_insights  # noqa: E402
from ai_insights import (  # noqa: E402
    FEEDBACK_SCHEMA,
    HABIT_TIP_NOTE,
    HABIT_TIPS,
    AIFeedback,
    FeedbackError,
    build_feedback_payload,
    get_summary,
    parse_feedback,
    validate_feedback,
)
from insights import night_facts, summarize  # noqa: E402


LABELS = {
    "total_sleep_min": "Total sleep",
    "deep_sleep_min": "Deep sleep",
    "sleeping_hr_avg_bpm": "Sleeping heart rate",
}
NIGHT = "2026-04-24"


def row(metric, night_date, observed, status="normal", baseline=None, difference=None, unit="min",
        z_score=None, participant="par_1"):
    return {
        "participant_id": participant, "night_date": night_date, "metric": metric, "unit": unit,
        "status": status, "baseline_window": "7_days", "observed_value": observed,
        "baseline_value": baseline, "difference": difference, "z_score": z_score,
        "threshold_type": "absolute_z_score", "threshold_value": 2.0,
        "baseline_nights": 7, "minimum_nights": 3,
    }


def selected_night():
    return pd.DataFrame([
        row("total_sleep_min", NIGHT, 430, "change", 470, -40, z_score=-2.4),
        row("deep_sleep_min", NIGHT, 92, "change", 170, -78, z_score=-3.1),
        row("sleeping_hr_avg_bpm", NIGHT, 60, "change", 50, 10, unit="bpm", z_score=2.6),
    ])


def history():
    rows = []
    for day, (total, deep, hr) in enumerate([(480, 175, 50), (465, 168, 51), (470, 172, 49),
                                             (455, 160, 50), (475, 171, 50), (468, 169, 51),
                                             (472, 170, 49), (466, 165, 50)], start=15):
        date = f"2026-04-{day:02d}"
        rows += [row("total_sleep_min", date, total), row("deep_sleep_min", date, deep),
                 row("sleeping_hr_avg_bpm", date, hr, unit="bpm")]
    # A later night must never leak into the "previous nights".
    rows.append(row("total_sleep_min", "2026-04-25", 999))
    return pd.concat([pd.DataFrame(rows), selected_night()], ignore_index=True)


def facts():
    return night_facts(selected_night(), LABELS)


GOOD_REPLY = {
    "this_night": [
        "Deep sleep was 92 min, well below your usual 170 min.",
        "On the same night your sleeping heart rate was 60 bpm instead of about 50 bpm.",
    ],
    "recent_nights": ["Your previous 7 nights had steady deep sleep, so this night stands out."],
    "habit_tips": [{"tip_id": "wind_down", "reason": "It may help on nights after a busy day."}],
}


class FakeClient:
    """Records every call and returns a canned reply (or raises)."""

    def __init__(self, reply=None, error=None):
        self.calls = []
        self.reply = reply
        self.error = error
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return SimpleNamespace(output_text=text)


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    ai_insights.clear_cache()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    yield
    ai_insights.clear_cache()


# ---------------------------------------------------------
# get_summary: the four outcomes app.py shows
# ---------------------------------------------------------

def test_ai_off_uses_rules_without_calling_the_api():
    client = FakeClient(GOOD_REPLY)
    summary, source = get_summary(facts(), use_ai=False, client=client)
    assert source == "Rule-based"
    assert summary == summarize(facts())
    assert client.calls == []


def test_no_key_uses_rules(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = FakeClient(GOOD_REPLY)
    _, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "Rule-based (AI unavailable)"
    assert client.calls == []


def test_accepted_reply_shows_all_three_kinds_of_feedback():
    client = FakeClient(GOOD_REPLY)
    summary, source = get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    assert source == "AI-generated"
    assert summary.headline == summarize(facts()).headline
    assert summary.details[:2] == GOOD_REPLY["this_night"]
    assert summary.details[2] == "Recent nights: " + GOOD_REPLY["recent_nights"][0]
    assert summary.details[3].startswith("Habit to try: " + HABIT_TIPS["wind_down"])
    assert summary.details[-1] == HABIT_TIP_NOTE


def test_request_uses_strict_json_schema_and_the_configured_model():
    client = FakeClient(GOOD_REPLY)
    get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    assert call["text"]["format"]["schema"] == FEEDBACK_SCHEMA
    sent = call["input"]
    assert "par_1" not in sent and "2026-04" not in sent


def test_works_without_history_like_the_current_app():
    reply = dict(GOOD_REPLY, recent_nights=[])
    summary, source = get_summary(facts(), use_ai=True, client=FakeClient(reply))
    assert source == "AI-generated"
    assert not any(line.startswith("Recent nights:") for line in summary.details)


@pytest.mark.parametrize("client", [
    FakeClient(error=TimeoutError("timed out")),
    FakeClient(reply="not json at all"),
    FakeClient(reply=""),
    FakeClient(reply={"this_night": [], "recent_nights": [], "habit_tips": []}),
    FakeClient(reply=dict(GOOD_REPLY, this_night=["Deep sleep fell by 46% to 92 min."])),
    FakeClient(reply=dict(GOOD_REPLY, this_night=["This could be a sign of insomnia."])),
    FakeClient(reply=dict(GOOD_REPLY, this_night=["Your heart rate rose because of short deep sleep."])),
    FakeClient(reply=dict(GOOD_REPLY, habit_tips=[{"tip_id": "take_melatonin", "reason": "x"}])),
])
def test_bad_or_failed_replies_fall_back_to_rules(client):
    summary, source = get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    assert source == "Rule-based (AI fallback)"
    assert summary == summarize(facts())


def test_recent_night_claims_need_history():
    # GOOD_REPLY talks about recent nights, but no history was sent.
    _, source = get_summary(facts(), use_ai=True, client=FakeClient(GOOD_REPLY))
    assert source == "Rule-based (AI fallback)"


def test_same_night_is_cached_across_reruns():
    client = FakeClient(GOOD_REPLY)
    for _ in range(3):
        get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    assert len(client.calls) == 1


def test_failed_replies_are_not_cached():
    failing = FakeClient(error=RuntimeError("down"))
    get_summary(facts(), use_ai=True, client=failing)
    working = FakeClient(dict(GOOD_REPLY, recent_nights=[]))
    _, source = get_summary(facts(), use_ai=True, client=working)
    assert source == "AI-generated"
    assert len(working.calls) == 1


# ---------------------------------------------------------
# The input the model sees
# ---------------------------------------------------------

def test_payload_has_this_night_without_ids_or_dates():
    payload = build_feedback_payload(facts())
    assert payload["compared_with"] == "your average over the previous 7 nights"
    assert payload["this_night"][1] == {
        "metric": "Deep sleep", "unit": "min", "status": "change",
        "observed": 92, "baseline": 170, "difference": -78, "z_score": -3.1,
    }
    assert "recent_nights" not in payload
    text = json.dumps(payload)
    assert "par_1" not in text and "2026" not in text


def test_payload_recent_nights_are_the_previous_seven_oldest_first():
    payload = build_feedback_payload(facts(), history(), NIGHT)
    recent = payload["recent_nights"]
    assert recent["nights_included"] == 7
    total = recent["metrics"][0]
    assert total["metric"] == "Total sleep"
    assert total["previous_values"] == [465, 470, 455, 475, 468, 472, 466]
    assert total["previous_average"] == 467
    assert total["this_night"] == 430
    assert 999 not in total["previous_values"]


def test_payload_skips_history_without_a_selected_night():
    assert "recent_nights" not in build_feedback_payload(facts(), history(), None)


# ---------------------------------------------------------
# Parsing and checks
# ---------------------------------------------------------

def test_parse_feedback_reads_the_schema_shape():
    feedback = parse_feedback(json.dumps(GOOD_REPLY))
    assert feedback.this_night == GOOD_REPLY["this_night"]
    assert feedback.habit_tips == [("wind_down", "It may help on nights after a busy day.")]


@pytest.mark.parametrize("text", ["[]", '{"this_night": "one string"}', '{"habit_tips": [{"reason": "x"}]}'])
def test_parse_feedback_rejects_wrong_shapes(text):
    with pytest.raises(FeedbackError):
        parse_feedback(text)


def test_minutes_may_be_written_as_hours_and_minutes():
    payload = build_feedback_payload(facts())
    feedback = AIFeedback(this_night=["You slept 7 h 10 min, about 40 min less than usual."])
    assert validate_feedback(feedback, payload) == []


def test_small_counts_are_allowed_in_prose():
    payload = build_feedback_payload(facts())
    feedback = AIFeedback(this_night=["3 measurements stood out compared with your previous 7 nights."])
    assert validate_feedback(feedback, payload) == []


def test_invented_numbers_are_reported():
    payload = build_feedback_payload(facts())
    problems = validate_feedback(AIFeedback(this_night=["Deep sleep was 120 min."]), payload)
    assert problems == ["numbers not in the data: 120"]


def test_too_many_tips_are_rejected():
    payload = build_feedback_payload(facts())
    tips = [(tip, "") for tip in list(HABIT_TIPS)[:4]]
    problems = validate_feedback(AIFeedback(this_night=["Deep sleep was low."], habit_tips=tips), payload)
    assert "too many habit tips" in problems
