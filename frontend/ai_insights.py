"""Optional AI-generated sleep feedback with a rule-based fallback.

The AI chooses what to say; this module decides how it is said. The model replies
with structured claims only, so no text it writes ever reaches the screen:

1. This night: which measurements to mention, and the direction each moved
   compared with the person's own baseline, plus which changes happened on the
   same night.
2. Recent nights: how this night fits the participant's previous nights, as one
   of a few fixed patterns (for example "lower than on any previous night").
3. Habit tips: general, non-medical sleep habits from the fixed list in
   ``HABIT_TIPS``, each tied to a measurement that changed this night.

Every claim is checked against the data before anything is shown: the
measurement has to be in the input, the direction has to match its status and
the sign of its difference, a recent-nights pattern has to hold for the
previous values that were sent, and recent-night claims need history. The
sentences are then written from the templates in this file with the data's own
values, so the wording can't add numbers, diagnoses, health claims or
cause-and-effect claims. A reply with any wrong claim, a timeout or an error
falls back to the rule-based summary, so the dashboard never breaks.

``get_summary(facts, use_ai)`` keeps the signature app.py already uses. Passing
``history`` (the participant's rows for the selected baseline window) and
``night`` turns on the recent-nights feedback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from insights import DIRECTION_WORDS, WINDOW_PHRASES, Fact, NightSummary, format_amount, summarize


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_MODEL = "gpt-6-luna"
# Reasoning models (including OpenRouter's free ones) spend part of the output
# budget thinking before they write the JSON, and free models can be slow, so
# both limits leave room for that. OPENAI_MAX_OUTPUT_TOKENS and
# OPENAI_TIMEOUT_SECONDS in .env override them.
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_TOKENS = 8000
MAX_HISTORY_NIGHTS = 7
CACHE_SIZE = 64

MAX_THIS_NIGHT = 4
MAX_RECENT_NIGHTS = 3
MAX_HABIT_TIPS = 3

# General sleep habits only. The model picks ids from this list; it never writes
# its own advice, so the dashboard can't show medical or treatment suggestions.
HABIT_TIPS = {
    "consistent_schedule": "Keep a similar bedtime and wake-up time every day, weekends included.",
    "wind_down": "Give yourself 30 to 60 minutes to wind down before bed, away from bright screens.",
    "caffeine_timing": "Skip caffeine in the afternoon and evening.",
    "alcohol_timing": "Avoid alcohol in the few hours before bed.",
    "late_meals": "Avoid large meals close to bedtime.",
    "bedroom_environment": "Keep the bedroom cool, dark and quiet.",
    "daylight_and_activity": "Get daylight and some physical activity earlier in the day.",
    "time_in_bed": "Leave enough time in bed for the amount of sleep you get on your good nights.",
}
HABIT_TIP_NOTE = "Habit ideas are general sleep habits, not medical advice."

# What the model may say about a measurement this night. Each one must match the
# measurement's status, and for a change, the sign of its difference.
DIRECTIONS = ["higher", "lower", "usual", "not_enough_history", "missing"]
EXPECTED_DIRECTION = {"normal": "usual", "insufficient_history": "not_enough_history", "missing": "missing"}

# How this night may relate to the previous nights. Each is checked against the
# previous values that were sent to the model.
RECENT_PATTERNS = {
    "highest_of_recent": "this night is above every previous value",
    "lowest_of_recent": "this night is below every previous value",
    "within_recent_range": "this night is between the lowest and highest previous values",
    "above_recent_average": "this night is above previous_average",
    "below_recent_average": "this night is below previous_average",
    "flagged_before": "previous_nights_flagged is at least 1",
}
FEEDBACK_KEYS = ("this_night", "same_night", "recent_nights", "habit_tips")


def feedback_schema(refs: list[str]) -> dict:
    """The strict JSON schema for a reply; measurement refs are limited to the ones sent."""
    metric = {"type": "string", "enum": refs}

    def claims(description: str, **fields) -> dict:
        return {
            "type": "array",
            "description": description,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": fields,
                "required": list(fields),
            },
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "this_night": claims(
                "1 to 4 measurements to mention, most notable first, each with its direction.",
                metric=metric, direction={"type": "string", "enum": DIRECTIONS},
            ),
            "same_night": {
                "type": "array",
                "description": "Two or more measurements with status change that changed on this same night, or empty.",
                "items": metric,
            },
            "recent_nights": claims(
                "0 to 3 claims about how this night fits the previous nights. Empty if no recent nights are given.",
                metric=metric, pattern={"type": "string", "enum": list(RECENT_PATTERNS)},
            ),
            "habit_tips": claims(
                "0 to 3 habit tips, each tied to a measurement with status change.",
                tip_id={"type": "string", "enum": sorted(HABIT_TIPS)}, metric=metric,
            ),
        },
        "required": list(FEEDBACK_KEYS),
    }


INSTRUCTIONS = f"""
You choose what to tell one person about their wearable sleep results. You never
write sentences: you reply with structured claims, and the dashboard writes the
wording from them. Use ONLY the JSON data you are given. "this_night" compares
one night with the person's own baseline; "recent_nights", when present, lists
their previous nights (oldest first). Name measurements by their "ref".

Reply with:
1. this_night: 1 to 4 measurements, most notable first, each with a direction:
   "higher" or "lower" for status "change" (the sign of "difference"), "usual" for
   status "normal", "not_enough_history" for status "insufficient_history", and
   "missing" for status "missing".
2. same_night: the refs of the measurements with status "change" if two or more
   changed this night, otherwise an empty list.
3. recent_nights: 0 to 3 claims, each a ref and one pattern:
{json.dumps(RECENT_PATTERNS, indent=2)}
   Leave it empty when recent_nights is missing from the data.
4. habit_tips: 0 to 3 tip ids from this list, each tied to the ref of a
   measurement with status "change". Pick none if nothing changed.
{json.dumps(HABIT_TIPS, indent=2)}

Every claim is checked against the data, and a reply with any wrong claim is
thrown away, so only make claims the numbers support.
""".strip()


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

@dataclass
class AIFeedback:
    """The model's claims after parsing."""

    this_night: list[tuple[str, str]] = field(default_factory=list)  # (ref, direction)
    same_night: list[str] = field(default_factory=list)  # refs
    recent_nights: list[tuple[str, str]] = field(default_factory=list)  # (ref, pattern)
    habit_tips: list[tuple[str, str]] = field(default_factory=list)  # (tip_id, ref)


class FeedbackError(ValueError):
    """The AI reply can't be shown; the caller falls back to rule-based text."""


def ai_available() -> bool:
    """Return True when an OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# Input for the model
# ---------------------------------------------------------

def _round(value: float | None, unit: str) -> float | int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if unit in {"min", "count"}:
        return int(round(value))
    if unit == "°C":
        return round(float(value), 2)
    return round(float(value), 1)


def _fact_entry(fact: Fact) -> dict:
    entry = {
        "metric": fact.label,
        "ref": fact.metric,
        "unit": fact.unit,
        "status": fact.status,
        "observed": _round(fact.observed, fact.unit),
        "baseline": _round(fact.baseline, fact.unit),
        "difference": _round(fact.difference, fact.unit),
    }
    if fact.z_score is not None:
        entry["z_score"] = round(fact.z_score, 1)
    if fact.status == "insufficient_history":
        entry["prior_nights"] = fact.baseline_nights
        entry["nights_needed"] = fact.minimum_nights
    return {key: value for key, value in entry.items() if value is not None and value != ""}


def _recent_nights(facts: list[Fact], history: pd.DataFrame | None, night: str | None, max_nights: int) -> dict | None:
    """Previous nights for each metric on this night, oldest first, without dates or ids."""
    if history is None or history.empty or night is None:
        return None
    frame = history.copy()
    frame["night_date"] = pd.to_datetime(frame["night_date"], errors="coerce")
    selected = pd.Timestamp(night)
    prior = frame.loc[frame["night_date"].lt(selected)]
    nights = sorted(prior["night_date"].dropna().unique())[-max_nights:]
    if not nights:
        return None
    prior = prior.loc[prior["night_date"].isin(nights)]

    metrics = []
    for fact in facts:
        rows = prior.loc[prior["metric"].astype(str).eq(fact.metric)].sort_values("night_date")
        values = [_round(v, fact.unit) for v in pd.to_numeric(rows["observed_value"], errors="coerce")]
        present = [v for v in values if v is not None]
        if not present:
            continue
        metrics.append({
            "metric": fact.label,
            "ref": fact.metric,
            "unit": fact.unit,
            "previous_values": values,
            "previous_average": _round(sum(present) / len(present), fact.unit),
            "previous_lowest": min(present),
            "previous_highest": max(present),
            "this_night": _round(fact.observed, fact.unit),
            "previous_nights_flagged": int(rows["status"].astype(str).eq("change").sum()),
        })
    if not metrics:
        return None
    return {"nights_included": len(nights), "metrics": metrics}


def build_feedback_payload(
    facts: list[Fact],
    history: pd.DataFrame | None = None,
    night: str | None = None,
    max_nights: int = MAX_HISTORY_NIGHTS,
) -> dict:
    """The compact JSON the model sees. Holds no participant id and no dates."""
    window = facts[0].window if facts else ""
    payload = {
        "compared_with": WINDOW_PHRASES.get(window, "recent nights"),
        "rule_based_headline": summarize(facts).headline,
        "this_night": [_fact_entry(fact) for fact in facts],
    }
    recent = _recent_nights(facts, history, night, max_nights)
    if recent:
        payload["recent_nights"] = recent
    return payload


# ---------------------------------------------------------
# Checking the reply
# ---------------------------------------------------------

def expected_direction(entry: dict) -> str | None:
    """The only direction the data supports for a this-night measurement."""
    if entry.get("status") == "change":
        difference = entry.get("difference")
        if not difference:
            return None
        return "higher" if difference > 0 else "lower"
    return EXPECTED_DIRECTION.get(entry.get("status"))


def pattern_holds(metric: dict, pattern: str) -> bool:
    """Whether a recent-nights pattern is true for the values that were sent."""
    if pattern == "flagged_before":
        return metric.get("previous_nights_flagged", 0) >= 1
    now = metric.get("this_night")
    previous = [value for value in metric.get("previous_values", []) if value is not None]
    if now is None or not previous:
        return False
    average = metric.get("previous_average")
    return {
        "highest_of_recent": now > max(previous),
        "lowest_of_recent": now < min(previous),
        "within_recent_range": min(previous) <= now <= max(previous),
        "above_recent_average": average is not None and now > average,
        "below_recent_average": average is not None and now < average,
    }.get(pattern, False)


def validate_feedback(feedback: AIFeedback, payload: dict) -> list[str]:
    """Claims that the data doesn't support; an empty list means the reply can be shown."""
    problems = []
    night = {entry["ref"]: entry for entry in payload.get("this_night", [])}

    if not feedback.this_night:
        problems.append("no claims about this night")
    if len(feedback.this_night) > MAX_THIS_NIGHT:
        problems.append("too many claims about this night")
    mentioned = [ref for ref, _ in feedback.this_night]
    if len(set(mentioned)) != len(mentioned):
        problems.append("a measurement is mentioned twice")
    for ref, direction in feedback.this_night:
        entry = night.get(ref)
        if entry is None:
            problems.append(f"unknown measurement: {ref}")
            continue
        expected = expected_direction(entry)
        if direction != expected:
            problems.append(f"{ref} is {expected or 'not comparable'} this night, not {direction}")

    if feedback.same_night:
        if len(set(feedback.same_night)) < 2 or len(set(feedback.same_night)) != len(feedback.same_night):
            problems.append("a same-night claim needs two or more different measurements")
        for ref in feedback.same_night:
            if night.get(ref, {}).get("status") != "change":
                problems.append(f"{ref} didn't change this night")

    recent = payload.get("recent_nights")
    if feedback.recent_nights and not recent:
        problems.append("recent-night claims without recent-night data")
    elif feedback.recent_nights:
        if len(feedback.recent_nights) > MAX_RECENT_NIGHTS:
            problems.append("too many claims about recent nights")
        by_ref = {metric["ref"]: metric for metric in recent["metrics"]}
        for ref, pattern in feedback.recent_nights:
            if pattern not in RECENT_PATTERNS:
                problems.append(f"unknown recent-nights pattern: {pattern}")
            elif ref not in by_ref:
                problems.append(f"no recent nights were sent for {ref}")
            elif not pattern_holds(by_ref[ref], pattern):
                problems.append(f"{pattern} isn't true for {ref}")

    if len(feedback.habit_tips) > MAX_HABIT_TIPS:
        problems.append("too many habit tips")
    for tip_id, ref in feedback.habit_tips:
        if tip_id not in HABIT_TIPS:
            problems.append(f"unknown habit tip id: {tip_id}")
        if night.get(ref, {}).get("status") != "change":
            problems.append(f"habit tip {tip_id} is tied to {ref}, which didn't change this night")
    return list(dict.fromkeys(problems))


def parse_feedback(text: str) -> AIFeedback:
    """Turn the model's JSON text into AIFeedback, or raise FeedbackError.

    The reply must match the schema exactly: all four keys and nothing else, and
    claims with exactly the expected fields, all strings. Free text anywhere
    (a sentence in place of a claim, or an extra field) is rejected.
    """
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FeedbackError(f"reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise FeedbackError("reply is not a JSON object")
    if set(data) != set(FEEDBACK_KEYS):
        missing, extra = set(FEEDBACK_KEYS) - set(data), set(data) - set(FEEDBACK_KEYS)
        raise FeedbackError(f"reply keys don't match the schema (missing {sorted(missing)}, extra {sorted(extra)})")

    def claims(key: str, fields: tuple[str, str]) -> list[tuple[str, str]]:
        value = data[key]
        if not isinstance(value, list):
            raise FeedbackError(f"{key} must be a list")
        parsed = []
        for item in value:
            if not isinstance(item, dict) or set(item) != set(fields) or not all(isinstance(item[f], str) for f in fields):
                raise FeedbackError(f"each {key} claim needs exactly {' and '.join(fields)}, as strings")
            parsed.append((item[fields[0]], item[fields[1]]))
        return parsed

    same_night = data["same_night"]
    if not isinstance(same_night, list) or not all(isinstance(ref, str) for ref in same_night):
        raise FeedbackError("same_night must be a list of measurement refs")
    return AIFeedback(
        this_night=claims("this_night", ("metric", "direction")),
        same_night=list(same_night),
        recent_nights=claims("recent_nights", ("metric", "pattern")),
        habit_tips=claims("habit_tips", ("tip_id", "metric")),
    )


# ---------------------------------------------------------
# Wording: every sentence comes from these templates
# ---------------------------------------------------------

SUPERLATIVE_UP = {"min": "longer", "count": "more"}
SUPERLATIVE_DOWN = {"min": "shorter", "count": "fewer"}


def _change_text(difference: float, unit: str) -> str:
    """'1 h 18 min shorter', '10 bpm higher' or '2 more'."""
    up_word, down_word = DIRECTION_WORDS.get(unit, ("higher", "lower"))
    amount = format_amount(abs(difference), unit).lstrip("+")
    return f"{amount} {up_word if difference > 0 else down_word}"


def _mid_sentence(label: str) -> str:
    """'Deep sleep' -> 'deep sleep', but 'REM sleep' stays as it is."""
    return label if label[1:2].isupper() else label[:1].lower() + label[1:]


def _verb(label: str) -> str:
    return "were" if label.endswith("s") else "was"


def _join(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _this_night_sentence(entry: dict, direction: str, compared_with: str) -> str:
    label, unit = entry["metric"], entry.get("unit", "")
    verb = _verb(label)
    observed = format_amount(entry["observed"], unit) if "observed" in entry else None
    if direction in {"higher", "lower"}:
        usual = format_amount(entry["baseline"], unit)
        return f"{label} {verb} {_change_text(entry['difference'], unit)} than {compared_with} ({observed} vs {usual})."
    if direction == "usual":
        return f"{label} {verb} within your usual range ({observed})." if observed else f"{label} {verb} within your usual range."
    if direction == "not_enough_history":
        prior, needed = entry.get("prior_nights"), entry.get("nights_needed")
        detail = f" ({prior} of {needed} earlier nights)" if prior is not None and needed is not None else ""
        return f"{label}: not enough earlier nights to compare yet{detail}."
    return f"{label} {verb} not recorded this night."


def _recent_sentence(metric: dict, pattern: str) -> str:
    label, unit = metric["metric"], metric.get("unit", "")
    verb = _verb(label)
    previous = [value for value in metric["previous_values"] if value is not None]
    nights = f"your previous {len(previous)} night{'s' if len(previous) != 1 else ''}"
    now = format_amount(metric["this_night"], unit) if metric.get("this_night") is not None else ""
    low, high = format_amount(min(previous), unit), format_amount(max(previous), unit)
    up_word, down_word = DIRECTION_WORDS.get(unit, ("higher", "lower"))
    if pattern == "highest_of_recent":
        return f"{label} ({now}) {verb} {SUPERLATIVE_UP.get(unit, 'higher')} than on any of {nights} ({low} to {high})."
    if pattern == "lowest_of_recent":
        return f"{label} ({now}) {verb} {SUPERLATIVE_DOWN.get(unit, 'lower')} than on any of {nights} ({low} to {high})."
    if pattern == "within_recent_range":
        return f"{label} ({now}) {verb} within the range of {nights} ({low} to {high})."
    if pattern in {"above_recent_average", "below_recent_average"}:
        word = up_word if pattern == "above_recent_average" else down_word
        average = format_amount(metric["previous_average"], unit)
        return f"{label} ({now}) {verb} {word} than your average over {nights} ({average})."
    flagged = metric["previous_nights_flagged"]
    return f"{label} {verb} also flagged on {flagged} of {nights}."


def to_night_summary(feedback: AIFeedback, headline: str, payload: dict) -> NightSummary:
    """Write the checked claims as the panel's headline plus bullets."""
    night = {entry["ref"]: entry for entry in payload.get("this_night", [])}
    compared_with = payload.get("compared_with", "your recent nights")
    details = [_this_night_sentence(night[ref], direction, compared_with) for ref, direction in feedback.this_night]
    if feedback.same_night:
        labels = [night[ref]["metric"] for ref in feedback.same_night]
        labels = [labels[0]] + [_mid_sentence(label) for label in labels[1:]]
        details.append(f"{_join(labels)} changed on the same night; that doesn't mean one caused the other.")
    if feedback.recent_nights:
        by_ref = {metric["ref"]: metric for metric in payload["recent_nights"]["metrics"]}
        details += [f"Recent nights: {_recent_sentence(by_ref[ref], pattern)}" for ref, pattern in feedback.recent_nights]
    for tip_id, ref in feedback.habit_tips:
        label = _mid_sentence(night[ref]["metric"])
        details.append(f"Habit to try: {HABIT_TIPS[tip_id]} It fits the change in {label} this night.")
    if feedback.habit_tips:
        details.append(HABIT_TIP_NOTE)
    return NightSummary(headline=headline, details=details)


# ---------------------------------------------------------
# Calling OpenAI (cached)
# ---------------------------------------------------------

_cache: "OrderedDict[str, AIFeedback]" = OrderedDict()


def clear_cache() -> None:
    _cache.clear()


def _cache_key(payload: dict, model: str) -> str:
    raw = json.dumps({"model": model, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _env_number(name: str, default: float, cast=float):
    """A positive number from the environment, or the default if unset or invalid."""
    raw = os.getenv(name, "").strip()
    try:
        value = cast(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def output_token_limit() -> int:
    return _env_number("OPENAI_MAX_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS, int)


def request_timeout() -> float:
    return _env_number("OPENAI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, float)


def _make_client(timeout_seconds: float):
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available.")
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


def generate_ai_feedback(
    payload: dict,
    client=None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> AIFeedback:
    """Ask the model for feedback on the payload. Raises on any failure.

    Replies that pass the checks are cached in this process, so Streamlit reruns
    (every widget change) don't call the API again for the same night.
    """
    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    key = _cache_key(payload, model)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    client = client or _make_client(timeout_seconds or request_timeout())
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "sleep_feedback",
                "schema": feedback_schema([entry["ref"] for entry in payload["this_night"]]),
                "strict": True,
            }
        },
        max_output_tokens=output_token_limit(),
    )
    feedback = parse_feedback((getattr(response, "output_text", "") or "").strip())
    problems = validate_feedback(feedback, payload)
    if problems:
        raise FeedbackError("; ".join(problems))

    _cache[key] = feedback
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return feedback


# A bad or missing key fails the same way on every model, so trying a backup model
# would only waste time. A 403 (PermissionDeniedError) can be specific to one model,
# such as a model the key or region isn't allowed to use, so it moves on to the backup.
_NO_RETRY_ERRORS = {"AuthenticationError"}


def model_chain() -> list[str]:
    """OPENAI_MODEL first, then any OPENAI_FALLBACK_MODELS (comma-separated), without repeats."""
    models = [os.getenv("OPENAI_MODEL") or DEFAULT_MODEL]
    models += [name.strip() for name in os.getenv("OPENAI_FALLBACK_MODELS", "").split(",")]
    return list(dict.fromkeys(name for name in models if name))


def generate_with_fallbacks(
    payload: dict,
    client=None,
    timeout_seconds: float | None = None,
) -> AIFeedback:
    """Try each model in model_chain() until one gives a reply that passes the checks.

    A model that errors, times out, is busy or gives a reply that fails a check is
    skipped for the next one. Raises the last error if every model fails, so the
    caller can fall back to the rule-based summary.
    """
    models = model_chain()
    # A reply any model already gave for this night wins, so a main model that
    # failed isn't asked again on every Streamlit rerun.
    for model in models:
        cached = _cache.get(_cache_key(payload, model))
        if cached is not None:
            return cached
    client = client or _make_client(timeout_seconds or request_timeout())
    last_error: Exception | None = None
    for position, model in enumerate(models, start=1):
        try:
            return generate_ai_feedback(payload, client=client, model=model, timeout_seconds=timeout_seconds)
        except Exception as exc:  # any failure moves on to the next model
            last_error = exc
            if type(exc).__name__ in _NO_RETRY_ERRORS:
                raise
            if position < len(models):
                logger.warning("AI model %s failed (%s: %s); trying %s", model, type(exc).__name__, exc, models[position])
    raise last_error


def generate_ai_summary(
    facts: list[Fact],
    rule_summary: NightSummary,
    timeout_seconds: float | None = None,
    history: pd.DataFrame | None = None,
    night: str | None = None,
    client=None,
) -> NightSummary:
    """AI feedback as a NightSummary. Raises if every model fails."""
    payload = build_feedback_payload(facts, history, night)
    feedback = generate_with_fallbacks(payload, client=client, timeout_seconds=timeout_seconds)
    return to_night_summary(feedback, rule_summary.headline, payload)


# ---------------------------------------------------------
# Main function used by app.py
# ---------------------------------------------------------

def get_summary(
    facts: list[Fact],
    use_ai: bool = False,
    history: pd.DataFrame | None = None,
    night: str | None = None,
    client=None,
) -> tuple[NightSummary, str]:
    """Return either an AI-generated or rule-based summary, and which one it is.

    AI off:                      Rule-based
    AI on, no key:               Rule-based (AI unavailable)
    AI on, reply accepted:       AI-generated
    AI on, call or checks fail:  Rule-based (AI fallback)
    """
    rule_summary = summarize(facts)
    if not use_ai:
        return rule_summary, "Rule-based"
    if not ai_available():
        return rule_summary, "Rule-based (AI unavailable)"
    if not facts:
        return rule_summary, "Rule-based (AI fallback)"
    try:
        summary = generate_ai_summary(facts, rule_summary, history=history, night=night, client=client)
        return summary, "AI-generated"
    except Exception as exc:  # keep the dashboard running whatever the API does
        logger.warning("AI insight fell back to rule-based text: %s: %s", type(exc).__name__, exc)
        return rule_summary, "Rule-based (AI fallback)"
