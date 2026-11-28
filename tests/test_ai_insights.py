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
    HABIT_TIP_NOTE,
    HABIT_TIPS,
    AIFeedback,
    FeedbackError,
    build_feedback_payload,
    feedback_schema,
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


def claim(metric, direction):
    return {"metric": metric, "direction": direction}


GOOD_REPLY = {
    "this_night": [claim("deep_sleep_min", "lower"), claim("sleeping_hr_avg_bpm", "higher")],
    "same_night": ["deep_sleep_min", "sleeping_hr_avg_bpm"],
    "recent_nights": [{"metric": "deep_sleep_min", "pattern": "lowest_of_recent"}],
    "habit_tips": [{"tip_id": "wind_down", "metric": "deep_sleep_min"}],
}
NO_HISTORY_REPLY = dict(GOOD_REPLY, recent_nights=[])

# Everything the dashboard shows for GOOD_REPLY is written from templates and the data.
SHOWN = [
    "Deep sleep was 1 h 18 min shorter than your average over the previous 7 nights (1 h 32 min vs 2 h 50 min).",
    "Sleeping heart rate was 10 bpm higher than your average over the previous 7 nights (60 bpm vs 50 bpm).",
    "Deep sleep and sleeping heart rate changed on the same night; that doesn't mean one caused the other.",
    "Recent nights: Deep sleep (1 h 32 min) was shorter than on any of your previous 7 nights (2 h 40 min to 2 h 52 min).",
    "Habit to try: " + HABIT_TIPS["wind_down"] + " It fits the change in deep sleep this night.",
    HABIT_TIP_NOTE,
]


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


def ai_result(reply, with_history=True):
    kwargs = {"history": history(), "night": NIGHT} if with_history else {}
    return get_summary(facts(), use_ai=True, client=FakeClient(reply), **kwargs)


def problems_for(reply, with_history=True):
    payload = build_feedback_payload(facts(), history() if with_history else None, NIGHT if with_history else None)
    return validate_feedback(parse_feedback(json.dumps(reply)), payload)


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


def test_accepted_claims_are_written_from_templates():
    summary, source = ai_result(GOOD_REPLY)
    assert source == "AI-generated"
    assert summary.headline == summarize(facts()).headline
    assert summary.details == SHOWN


def test_request_uses_strict_json_schema_limited_to_the_measurements_sent():
    client = FakeClient(GOOD_REPLY)
    get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    schema = call["text"]["format"]["schema"]
    assert schema == feedback_schema(["total_sleep_min", "deep_sleep_min", "sleeping_hr_avg_bpm"])
    claim_fields = schema["properties"]["this_night"]["items"]["properties"]
    assert set(claim_fields) == {"metric", "direction"}  # no free-text field anywhere
    sent = call["input"]
    assert "par_1" not in sent and "2026-04" not in sent


def test_works_without_history_like_the_current_app():
    summary, source = ai_result(NO_HISTORY_REPLY, with_history=False)
    assert source == "AI-generated"
    assert not any(line.startswith("Recent nights:") for line in summary.details)


@pytest.mark.parametrize("client", [
    FakeClient(error=TimeoutError("timed out")),
    FakeClient(reply="not json at all"),
    FakeClient(reply=""),
    FakeClient(reply={"this_night": [], "same_night": [], "recent_nights": [], "habit_tips": []}),
    FakeClient(reply=dict(GOOD_REPLY, habit_tips=[{"tip_id": "take_melatonin", "metric": "deep_sleep_min"}])),
    FakeClient(reply={"this_night": GOOD_REPLY["this_night"]}),  # incomplete
    FakeClient(reply=dict(GOOD_REPLY, note="extra")),
])
def test_bad_or_failed_replies_fall_back_to_rules(client):
    summary, source = get_summary(facts(), use_ai=True, history=history(), night=NIGHT, client=client)
    assert source == "Rule-based (AI fallback)"
    assert summary == summarize(facts())


# ---------------------------------------------------------
# Youngjun's three cases (PR #92 review)
# ---------------------------------------------------------

@pytest.mark.parametrize("reply", [
    # Free text in place of a claim
    dict(GOOD_REPLY, this_night=["Deep sleep confirms anemia."]),
    # Free text riding along on an otherwise valid claim
    dict(GOOD_REPLY, this_night=[{"metric": "deep_sleep_min", "direction": "lower", "text": "Deep sleep confirms anemia."}]),
    dict(GOOD_REPLY, habit_tips=[{"tip_id": "wind_down", "metric": "deep_sleep_min", "reason": "It confirms anemia."}]),
])
def test_model_text_like_deep_sleep_confirms_anemia_can_never_be_shown(reply):
    summary, source = ai_result(reply)
    assert source == "Rule-based (AI fallback)"
    assert not any("anemia" in line for line in summary.details)


def test_wrong_direction_like_deep_sleep_above_usual_is_rejected():
    # Deep sleep was 1 h 32 min against a usual 2 h 50 min, so "higher" is wrong.
    reply = dict(GOOD_REPLY, this_night=[claim("deep_sleep_min", "higher")])
    assert problems_for(reply) == ["deep_sleep_min is lower this night, not higher"]
    _, source = ai_result(reply)
    assert source == "Rule-based (AI fallback)"


def test_recent_night_claims_without_history_are_rejected():
    reply = dict(NO_HISTORY_REPLY, recent_nights=[{"metric": "deep_sleep_min", "pattern": "lowest_of_recent"}])
    assert "recent-night claims without recent-night data" in problems_for(reply, with_history=False)
    _, source = ai_result(reply, with_history=False)
    assert source == "Rule-based (AI fallback)"


# ---------------------------------------------------------
# Checking each kind of claim against the data
# ---------------------------------------------------------

@pytest.mark.parametrize("this_night, problem", [
    ([claim("total_sleep_min", "usual")], "total_sleep_min is lower this night, not usual"),
    ([claim("sleeping_hr_avg_bpm", "lower")], "sleeping_hr_avg_bpm is higher this night, not lower"),
    ([claim("rem_sleep_min", "lower")], "unknown measurement: rem_sleep_min"),
    ([claim("deep_sleep_min", "lower"), claim("deep_sleep_min", "lower")], "a measurement is mentioned twice"),
    ([claim("deep_sleep_min", "sideways")], "deep_sleep_min is lower this night, not sideways"),
])
def test_this_night_claims_must_match_status_and_sign(this_night, problem):
    assert problem in problems_for(dict(GOOD_REPLY, this_night=this_night))


def test_direction_follows_each_status():
    rows = pd.DataFrame([
        row("total_sleep_min", NIGHT, 470, "normal", 468, 2),
        row("deep_sleep_min", NIGHT, 150, "insufficient_history"),
        row("sleeping_hr_avg_bpm", NIGHT, None, "missing", unit="bpm"),
    ])
    rows.loc[rows["status"].eq("insufficient_history"), "baseline_nights"] = 2
    payload = build_feedback_payload(night_facts(rows, LABELS))
    good = AIFeedback(this_night=[("total_sleep_min", "usual"), ("deep_sleep_min", "not_enough_history"),
                                  ("sleeping_hr_avg_bpm", "missing")])
    assert validate_feedback(good, payload) == []
    assert ai_insights.to_night_summary(good, "H", payload).details == [
        "Total sleep was within your usual range (7 h 50 min).",
        "Deep sleep: not enough earlier nights to compare yet (2 of 3 earlier nights).",
        "Sleeping heart rate was not recorded this night.",
    ]
    wrong = AIFeedback(this_night=[("total_sleep_min", "higher")])
    assert validate_feedback(wrong, payload) == ["total_sleep_min is usual this night, not higher"]


def test_same_night_claims_need_two_changed_measurements():
    assert problems_for(dict(GOOD_REPLY, same_night=["deep_sleep_min"])) == [
        "a same-night claim needs two or more different measurements"
    ]
    rows = pd.concat([selected_night(), pd.DataFrame([row("rem_sleep_min", NIGHT, 100, "normal", 98, 2)])])
    payload = build_feedback_payload(night_facts(rows, LABELS))
    feedback = AIFeedback(this_night=[("deep_sleep_min", "lower")], same_night=["deep_sleep_min", "rem_sleep_min"])
    assert validate_feedback(feedback, payload) == ["rem_sleep_min didn't change this night"]


@pytest.mark.parametrize("pattern, holds", [
    ("lowest_of_recent", True),        # 92 is below every previous value (160 to 172)
    ("highest_of_recent", False),
    ("within_recent_range", False),
    ("below_recent_average", True),
    ("above_recent_average", False),
    ("flagged_before", False),         # none of the previous nights were flagged
])
def test_recent_patterns_are_checked_against_previous_values(pattern, holds):
    reply = dict(GOOD_REPLY, recent_nights=[{"metric": "deep_sleep_min", "pattern": pattern}])
    assert (problems_for(reply) == []) is holds


def test_recent_patterns_are_worded_from_the_data():
    payload = build_feedback_payload(facts(), history(), NIGHT)
    feedback = AIFeedback(
        this_night=[("deep_sleep_min", "lower")],
        recent_nights=[("deep_sleep_min", "below_recent_average"), ("total_sleep_min", "lowest_of_recent")],
    )
    assert validate_feedback(feedback, payload) == []
    assert ai_insights.to_night_summary(feedback, "H", payload).details[1:] == [
        "Recent nights: Deep sleep (1 h 32 min) was shorter than your average over your previous 7 nights (2 h 48 min).",
        "Recent nights: Total sleep (7 h 10 min) was shorter than on any of your previous 7 nights (7 h 35 min to 7 h 55 min).",
    ]


def test_habit_tips_must_be_tied_to_a_measurement_that_changed():
    rows = pd.concat([selected_night(), pd.DataFrame([row("rem_sleep_min", NIGHT, 100, "normal", 98, 2)])])
    payload = build_feedback_payload(night_facts(rows, LABELS))
    feedback = AIFeedback(this_night=[("deep_sleep_min", "lower")], habit_tips=[("wind_down", "rem_sleep_min")])
    assert validate_feedback(feedback, payload) == [
        "habit tip wind_down is tied to rem_sleep_min, which didn't change this night"
    ]


def test_too_many_tips_are_rejected():
    tips = [{"tip_id": tip, "metric": "deep_sleep_min"} for tip in list(HABIT_TIPS)[:4]]
    assert "too many habit tips" in problems_for(dict(GOOD_REPLY, habit_tips=tips))


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
    working = FakeClient(NO_HISTORY_REPLY)
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
# Parsing
# ---------------------------------------------------------

def test_parse_feedback_reads_the_claims():
    feedback = parse_feedback(json.dumps(GOOD_REPLY))
    assert feedback.this_night == [("deep_sleep_min", "lower"), ("sleeping_hr_avg_bpm", "higher")]
    assert feedback.same_night == ["deep_sleep_min", "sleeping_hr_avg_bpm"]
    assert feedback.recent_nights == [("deep_sleep_min", "lowest_of_recent")]
    assert feedback.habit_tips == [("wind_down", "deep_sleep_min")]


@pytest.mark.parametrize("reply", [
    [],
    {"this_night": "deep_sleep_min", "same_night": [], "recent_nights": [], "habit_tips": []},
    {"this_night": [claim("deep_sleep_min", "lower")]},  # missing keys
    dict(GOOD_REPLY, extra=1),
    dict(GOOD_REPLY, this_night=[{"metric": "deep_sleep_min"}]),
    dict(GOOD_REPLY, this_night=[{"metric": "deep_sleep_min", "direction": 1}]),
    dict(GOOD_REPLY, same_night="deep_sleep_min"),
    dict(GOOD_REPLY, habit_tips=[{"tip_id": "wind_down"}]),
])
def test_parse_feedback_enforces_the_full_schema(reply):
    with pytest.raises(FeedbackError):
        parse_feedback(json.dumps(reply))


# ---------------------------------------------------------
# Output budget and timeout (reasoning models need room to think)
# ---------------------------------------------------------

def test_default_output_budget_leaves_room_for_reasoning(monkeypatch):
    monkeypatch.delenv("OPENAI_MAX_OUTPUT_TOKENS", raising=False)
    client = FakeClient(NO_HISTORY_REPLY)
    get_summary(facts(), use_ai=True, client=client)
    assert client.calls[0]["max_output_tokens"] == ai_insights.MAX_OUTPUT_TOKENS >= 4000


def test_output_budget_and_timeout_can_be_set_in_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "1500")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "20")
    client = FakeClient(NO_HISTORY_REPLY)
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


class PermissionDeniedError(Exception):
    """Same name as the OpenAI SDK's 403 error."""


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
    assert summary.details[:2] == SHOWN[:2]


def test_backups_are_not_tried_once_one_model_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    client = ModelFakeClient({"test-model": NO_HISTORY_REPLY, "backup-a": NO_HISTORY_REPLY})
    get_summary(facts(), use_ai=True, client=client)
    assert client.models_tried == ["test-model"]


def test_reply_with_a_wrong_claim_moves_to_the_backup(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODELS", "backup-a")
    bad = dict(NO_HISTORY_REPLY, this_night=[claim("deep_sleep_min", "higher")])
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
