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
        "Deep sleep was {deep_sleep_min.observed}, {deep_sleep_min.change} than your usual {deep_sleep_min.usual}.",
        "On the same night your sleeping heart rate was {sleeping_hr_avg_bpm.change} than usual.",
    ],
    "recent_nights": [
        "Over your last {recent.nights} nights, deep sleep stayed between "
        "{deep_sleep_min.recent_lowest} and {deep_sleep_min.recent_highest}, so this night stands out."
    ],
    "habit_tips": [{"tip_id": "wind_down", "reason": "it may help on nights with shorter deep sleep."}],
}
# What the dashboard shows for GOOD_REPLY: the placeholders filled in from the data.
SHOWN_THIS_NIGHT = [
    "Deep sleep was 1 h 32 min, 1 h 18 min shorter than your usual 2 h 50 min.",
    "On the same night your sleeping heart rate was 10 bpm higher than usual.",
]
SHOWN_RECENT = "Over your last 7 nights, deep sleep stayed between 2 h 40 min and 2 h 52 min, so this night stands out."


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
    # A developer's own .env must not change what the tests see.
    for name in ("OPENAI_FALLBACK_MODELS", "OPENAI_MAX_OUTPUT_TOKENS", "OPENAI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
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
    assert summary.details[:2] == SHOWN_THIS_NIGHT
    assert summary.details[2] == "Recent nights: " + SHOWN_RECENT
    assert summary.details[3] == (
        "Habit to try: " + HABIT_TIPS["wind_down"] + " It may help on nights with shorter deep sleep."
    )
    assert summary.details[-1] == HABIT_TIP_NOTE
    assert not any("{" in line for line in summary.details)


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
    # From Youngjun's review: a health claim and a number from another measurement.
    FakeClient(reply=dict(GOOD_REPLY, this_night=["You have hypertension."])),
    FakeClient(reply=dict(GOOD_REPLY, this_night=["Deep sleep was {sleeping_hr_avg_bpm.observed}."])),
    FakeClient(reply=dict(GOOD_REPLY, this_night=["Deep sleep was 60 bpm."])),
    # Incomplete or extra keys don't match the schema.
    FakeClient(reply={"this_night": GOOD_REPLY["this_night"]}),
    FakeClient(reply=dict(GOOD_REPLY, note="extra")),
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
        "metric": "Deep sleep", "ref": "deep_sleep_min", "unit": "min", "status": "change",
        "observed": 92, "baseline": 170, "difference": -78, "z_score": -3.1,
        "placeholders": ["{deep_sleep_min.observed}", "{deep_sleep_min.usual}", "{deep_sleep_min.change}"],
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
    assert (total["previous_lowest"], total["previous_highest"]) == (455, 475)
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
    assert feedback.habit_tips == [("wind_down", "it may help on nights with shorter deep sleep.")]


@pytest.mark.parametrize("reply", [
    [],
    {"this_night": "one string", "recent_nights": [], "habit_tips": []},
    {"this_night": ["Deep sleep was low."]},  # missing keys
    {"this_night": ["Deep sleep was low."], "recent_nights": [], "habit_tips": [], "extra": 1},
    {"this_night": ["Deep sleep was low."], "recent_nights": [], "habit_tips": [{"reason": "x"}]},
    {"this_night": ["Deep sleep was low."], "recent_nights": [], "habit_tips": [{"tip_id": "wind_down"}]},
    {"this_night": ["Deep sleep was low."], "recent_nights": [],
     "habit_tips": [{"tip_id": "wind_down", "reason": "x", "extra": "y"}]},
])
def test_parse_feedback_enforces_the_full_schema(reply):
    with pytest.raises(FeedbackError):
        parse_feedback(json.dumps(reply))


def problems_for(*sentences, history_given=False):
    payload = build_feedback_payload(facts(), history() if history_given else None, NIGHT if history_given else None)
    return validate_feedback(AIFeedback(this_night=list(sentences)), payload)


def test_placeholders_are_filled_in_from_the_data():
    payload = build_feedback_payload(facts(), history(), NIGHT)
    values = ai_insights.placeholder_values(payload)
    assert values["deep_sleep_min.observed"] == "1 h 32 min"
    assert values["deep_sleep_min.change"] == "1 h 18 min shorter"
    assert values["sleeping_hr_avg_bpm.change"] == "10 bpm higher"
    assert values["total_sleep_min.recent_lowest"] == "7 h 35 min"
    assert values["recent.nights"] == "7"
    assert ai_insights.render("Deep sleep was {deep_sleep_min.observed}.", values) == "Deep sleep was 1 h 32 min."


@pytest.mark.parametrize("sentence", [
    "Deep sleep was 120 min.",           # invented
    "Deep sleep was 92 min.",            # right number, but written out instead of a placeholder
    "Deep sleep was 60 bpm.",            # a number from another measurement
    "Deep sleep fell by 46%.",
    "Deep sleep was about twenty minutes shorter.",
    "Deep sleep was half of your usual.",
])
def test_numbers_must_come_from_placeholders(sentence):
    assert problems_for(sentence)


@pytest.mark.parametrize("sentence", [
    "Deep sleep was {sleeping_hr_avg_bpm.observed}.",
    "Your sleeping heart rate was {deep_sleep_min.change} than usual.",
    "Total sleep was {total_sleep_min.observed} while deep sleep was {total_sleep_min.usual}.",
    "You slept {total_sleep_min.observed}.",  # no measurement named for the number
])
def test_placeholders_must_sit_with_their_own_measurement(sentence):
    assert any("own measurement" in problem for problem in problems_for(sentence))


def test_unknown_placeholders_are_rejected():
    assert problems_for("Deep sleep was {deep_sleep_min.minimum}.") == [
        "unknown placeholder: {deep_sleep_min.minimum}"
    ]
    # Recent-night placeholders only exist when recent nights were sent.
    assert problems_for("Deep sleep averaged {deep_sleep_min.recent_average}.")


@pytest.mark.parametrize("sentence", [
    "Deep sleep rose to {deep_sleep_min.observed}.",
    "Your sleeping heart rate dropped to {sleeping_hr_avg_bpm.observed}.",
    "Deep sleep went up to {deep_sleep_min.observed} from {deep_sleep_min.usual}.",
])
def test_direction_words_must_match_the_data(sentence):
    assert any("contradict" in problem for problem in problems_for(sentence))


@pytest.mark.parametrize("sentence", [
    "Deep sleep was {deep_sleep_min.change} than usual, at {deep_sleep_min.observed}.",
    "Your sleeping heart rate rose to {sleeping_hr_avg_bpm.observed}.",
    "You got {deep_sleep_min.observed} of deep sleep.",
    "Total sleep was {total_sleep_min.observed} and deep sleep was {deep_sleep_min.observed}.",
    "Deep sleep and sleeping heart rate both changed on the same night.",
])
def test_placeholders_next_to_their_measurement_pass(sentence):
    assert problems_for(sentence) == []


@pytest.mark.parametrize("sentence", [
    "You have hypertension.",
    "You may have a sleep disorder.",
    "Your sleeping heart rate suggests stress.",
    "This could be a sign of an infection.",
    "Your deep sleep is a warning sign.",
    "Your sleeping heart rate points to a heart condition.",
    "You are probably sick.",
    "Your body is recovering from something.",
])
def test_health_claims_are_rejected(sentence):
    assert problems_for(sentence)


def test_every_sentence_must_be_about_a_measurement():
    assert problems_for("Things look different tonight.") == ["a sentence isn't about any of the measurements"]
    assert problems_for("All measurements were in your usual range.") == []


def test_habit_tip_reasons_are_checked_too():
    payload = build_feedback_payload(facts())
    for reason in ["It may help on nights after a busy day.", "You have hypertension.", "Deep sleep was 92 min."]:
        feedback = AIFeedback(this_night=["Deep sleep was low."], habit_tips=[("wind_down", reason)])
        assert validate_feedback(feedback, payload), reason
    ok = AIFeedback(this_night=["Deep sleep was low."], habit_tips=[("wind_down", "deep sleep was shorter than usual")])
    assert validate_feedback(ok, payload) == []


def test_too_many_tips_are_rejected():
    payload = build_feedback_payload(facts())
    tips = [(tip, "") for tip in list(HABIT_TIPS)[:4]]
    problems = validate_feedback(AIFeedback(this_night=["Deep sleep was low."], habit_tips=tips), payload)
    assert "too many habit tips" in problems


# ---------------------------------------------------------
# Output budget and timeout (reasoning models need room to think)
# ---------------------------------------------------------

def test_default_output_budget_leaves_room_for_reasoning(monkeypatch):
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)
    client = FakeClient(dict(GOOD_REPLY, recent_nights=[]))
    get_summary(facts(), use_ai=True, client=client)
    assert client.calls[0]["max_output_tokens"] == ai_insights.MAX_OUTPUT_TOKENS >= 4000


def test_output_budget_and_timeout_can_be_set_in_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "20")
    client = FakeClient(dict(GOOD_REPLY, recent_nights=[]))
    get_summary(facts(), use_ai=True, client=client)
    assert client.calls[0]["max_output_tokens"] == 1500
    assert ai_insights.request_timeout() == 20.0


@pytest.mark.parametrize("value", ["", "lots", "0", "-5"])
def test_bad_env_values_fall_back_to_defaults(monkeypatch, value):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", value)
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)
    assert ai_insights.output_token_limit() == ai_insights.MAX_OUTPUT_TOKENS
    assert ai_insights.request_timeout() == ai_insights.DEFAULT_TIMEOUT_SECONDS


# ---------------------------------------------------------
# Backup models (OPENAI_FALLBACK_MODELS)
# ---------------------------------------------------------

class ModelFakeClient:
    """Gives each model its own canned reply or error, and records the order tried."""

    def __init__(self, by_model):
        self.by_model = by_model
        self.models_tried = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        model = kwargs["model"]
        self.models_tried.append(model)
        outcome = self.by_model[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(output_text=outcome if isinstance(outcome, str) else json.dumps(outcome))


class AuthenticationError(Exception):
    """Same name as the OpenAI SDK's error, so the code treats it the same way."""


NO_HISTORY_REPLY = dict(GOOD_REPLY, recent_nights=[])


def test_model_chain_reads_env_in_order_without_repeats(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", " backup-a , ,backup-b,test-model, backup-a ")
    assert ai_insights.model_chain() == ["test-model", "backup-a", "backup-b"]


def test_no_fallbacks_means_only_the_main_model(monkeypatch):
    assert ai_insights.model_chain() == ["test-model"]


def test_backup_model_is_used_when_the_main_one_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a,backup-b")
    client = ModelFakeClient({"test-model": TimeoutError("busy"), "backup-a": "not json", "backup-b": NO_HISTORY_REPLY})
    summary, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "AI-generated"
    assert client.models_tried == ["test-model", "backup-a", "backup-b"]
    assert summary.details[:2] == SHOWN_THIS_NIGHT


def test_backups_are_not_tried_once_one_model_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": NO_HISTORY_REPLY, "backup-a": NO_HISTORY_REPLY})
    get_summary(facts(), use_ai=True, client=client)
    assert client.models_tried == ["test-model"]


def test_reply_that_fails_checks_moves_to_the_backup(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    bad = dict(NO_HISTORY_REPLY, this_night=["This could be a sign of insomnia."])
    client = ModelFakeClient({"test-model": bad, "backup-a": NO_HISTORY_REPLY})
    _, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "AI-generated"
    assert client.models_tried == ["test-model", "backup-a"]


def test_rule_based_when_every_model_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": RuntimeError("down"), "backup-a": TimeoutError("busy")})
    summary, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "Rule-based (AI fallback)"
    assert summary == summarize(facts())
    assert client.models_tried == ["test-model", "backup-a"]


def test_key_errors_skip_the_backups(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": AuthenticationError("bad key"), "backup-a": NO_HISTORY_REPLY})
    _, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "Rule-based (AI fallback)"
    assert client.models_tried == ["test-model"]


class PermissionDeniedError(Exception):
    """Same name as the OpenAI SDK's 403 error."""


def test_a_403_on_one_model_moves_to_the_backup(monkeypatch):
    # A 403 can be about one model (e.g. not allowed for this key), unlike a bad key.
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": PermissionDeniedError("model not allowed"), "backup-a": NO_HISTORY_REPLY})
    _, source = get_summary(facts(), use_ai=True, client=client)
    assert source == "AI-generated"
    assert client.models_tried == ["test-model", "backup-a"]


def test_backup_reply_is_cached_too(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": RuntimeError("down"), "backup-a": NO_HISTORY_REPLY})
    get_summary(facts(), use_ai=True, client=client)
    get_summary(facts(), use_ai=True, client=client)
    # Second run (a Streamlit rerun): the backup's accepted reply comes from the cache,
    # and the main model that failed is not asked again.
    assert client.models_tried == ["test-model", "backup-a"]
