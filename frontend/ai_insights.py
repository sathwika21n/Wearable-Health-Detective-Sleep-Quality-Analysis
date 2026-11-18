"""Optional AI-generated sleep feedback with a rule-based fallback.

The AI writes three kinds of feedback, all from results the dashboard has already
computed:

1. This night: what changed compared with the person's own baseline, and which
   changes happened on the same night.
2. Recent nights: how this night fits the participant's previous nights.
3. Habit tips: general, non-medical sleep habits, picked by the model from the
   fixed list in ``HABIT_TIPS``. The model only chooses tip ids and gives a short
   reason; the tip wording itself always comes from this file.

The reply is structured JSON. It is checked before it is shown (no numbers that
are not in the input, no diagnosis or cause-and-effect wording, no recent-night
claims without history, known tip ids only). Anything that fails a check, times
out or errors falls back to the rule-based summary, so the dashboard never breaks.

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
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from insights import WINDOW_PHRASES, Fact, NightSummary, summarize


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
MAX_SENTENCE_CHARS = 320

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

# Words that would make the feedback sound like a diagnosis, treatment or a
# cause-and-effect claim. Matching is case-insensitive on word starts.
BANNED_PATTERNS = [
    r"\bdiagnos", r"\bdisorder", r"\binsomnia", r"\bapnea", r"\bapnoea", r"\bdisease",
    r"\billness", r"\bsymptom", r"\bmedicat", r"\bprescri", r"\btreat", r"\bcure",
    r"\bsupplement", r"\bmelatonin", r"\bcaused\b", r"\bcauses\b", r"\bbecause of\b",
    r"\bdue to\b", r"\bled to\b", r"\bleads to\b", r"\bresult(?:s|ed)? in\b",
]
_BANNED_RE = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d\s*(?:%|percent\b)", re.IGNORECASE)

# Numbers this small are fine in prose ("2 of 7 nights", "3 measurements").
FREE_NUMBERS = set(range(0, 11))

FEEDBACK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "this_night": {
            "type": "array",
            "description": "1 to 4 short sentences about the selected night compared with the baseline.",
            "items": {"type": "string"},
        },
        "recent_nights": {
            "type": "array",
            "description": "0 to 3 short sentences about how this night fits the recent nights. Empty if no recent nights are given.",
            "items": {"type": "string"},
        },
        "habit_tips": {
            "type": "array",
            "description": "0 to 3 habit tips chosen from the allowed tip ids.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tip_id": {"type": "string", "enum": sorted(HABIT_TIPS)},
                    "reason": {
                        "type": "string",
                        "description": "One short clause linking the tip to this night's pattern.",
                    },
                },
                "required": ["tip_id", "reason"],
            },
        },
    },
    "required": ["this_night", "recent_nights", "habit_tips"],
}

INSTRUCTIONS = f"""
You explain one person's wearable sleep results in plain language.
Use ONLY the JSON data you are given. It compares one night with the person's own
baseline and, when present, lists their previous nights (oldest first).

Write three parts:
1. this_night: 1 to 4 short sentences on what stood out this night and which
   changes happened on the same night. If nothing changed, say so briefly.
2. recent_nights: 0 to 3 short sentences on how this night fits the previous
   nights (for example a run of short nights, or a value back near its usual
   level). Leave it empty when recent_nights is missing or empty in the data.
3. habit_tips: 0 to 3 tip ids from this list that fit the pattern, each with a
   short reason. Pick none if nothing stood out.
{json.dumps(HABIT_TIPS, indent=2)}

Rules:
- Never diagnose, name a condition, or mention medicine, supplements or treatment.
- Never say one measurement caused another; say they happened on the same night.
- Only use numbers that appear in the data. Do not calculate new numbers,
  percentages or averages. Minutes may be written as hours and minutes.
- Speak to the person as "you". Keep every sentence under 40 words.
""".strip()


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

@dataclass
class AIFeedback:
    """The model's reply after parsing."""

    this_night: list[str] = field(default_factory=list)
    recent_nights: list[str] = field(default_factory=list)
    habit_tips: list[tuple[str, str]] = field(default_factory=list)


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
            "unit": fact.unit,
            "previous_values": values,
            "previous_average": _round(sum(present) / len(present), fact.unit),
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

def _payload_numbers(payload) -> set[float]:
    """Every number in the payload, plus the forms prose may use for it."""
    numbers: set[float] = set()

    def add(value: float, unit: str | None) -> None:
        numbers.update({value, abs(value)})
        if unit == "min":
            minutes = int(round(abs(value)))
            hours, rest = divmod(minutes, 60)
            numbers.update({hours, rest, round(minutes / 60, 1)})

    def walk(node, unit=None) -> None:
        if isinstance(node, dict):
            unit = node.get("unit", unit)
            for value in node.values():
                walk(value, unit)
        elif isinstance(node, list):
            for value in node:
                walk(value, unit)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            add(float(node), unit)
        elif isinstance(node, str):
            for match in _NUMBER_RE.findall(node):
                add(float(match), None)

    walk(payload)
    return numbers


def _number_allowed(number: float, allowed: set[float]) -> bool:
    if abs(number) in FREE_NUMBERS:
        return True
    return any(abs(abs(number) - abs(value)) <= max(0.051, 0.005 * abs(value)) for value in allowed)


def validate_feedback(feedback: AIFeedback, payload: dict) -> list[str]:
    """Reasons the reply can't be shown; an empty list means it's fine."""
    problems = []
    if not any(sentence.strip() for sentence in feedback.this_night):
        problems.append("no sentences about this night")
    if len(feedback.this_night) > MAX_THIS_NIGHT:
        problems.append("too many sentences about this night")
    if len(feedback.recent_nights) > MAX_RECENT_NIGHTS:
        problems.append("too many sentences about recent nights")
    if feedback.recent_nights and "recent_nights" not in payload:
        problems.append("recent-night claims without recent-night data")
    if len(feedback.habit_tips) > MAX_HABIT_TIPS:
        problems.append("too many habit tips")
    unknown = [tip for tip, _ in feedback.habit_tips if tip not in HABIT_TIPS]
    if unknown:
        problems.append(f"unknown habit tip ids: {', '.join(unknown)}")

    allowed = _payload_numbers(payload)
    texts = feedback.this_night + feedback.recent_nights + [reason for _, reason in feedback.habit_tips]
    for text in texts:
        if len(text) > MAX_SENTENCE_CHARS:
            problems.append("a sentence is too long")
        banned = _BANNED_RE.search(text)
        if banned:
            problems.append(f"banned wording: {banned.group(0)!r}")
        if _PERCENT_RE.search(text):
            # The payload has no percentages, so any percentage is a new calculation.
            problems.append("percentages are not in the data")
        invented = [n for n in _NUMBER_RE.findall(text) if not _number_allowed(float(n), allowed)]
        if invented:
            problems.append(f"numbers not in the data: {', '.join(invented)}")
    return problems


def parse_feedback(text: str) -> AIFeedback:
    """Turn the model's JSON text into AIFeedback, or raise FeedbackError."""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FeedbackError(f"reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise FeedbackError("reply is not a JSON object")

    def strings(key: str) -> list[str]:
        value = data.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise FeedbackError(f"{key} must be a list of strings")
        return [item.strip() for item in value if item.strip()]

    tips = data.get("habit_tips", [])
    if not isinstance(tips, list):
        raise FeedbackError("habit_tips must be a list")
    parsed_tips = []
    for tip in tips:
        if not isinstance(tip, dict) or not isinstance(tip.get("tip_id"), str):
            raise FeedbackError("each habit tip needs a tip_id")
        parsed_tips.append((tip["tip_id"], str(tip.get("reason", "")).strip()))
    return AIFeedback(strings("this_night"), strings("recent_nights"), parsed_tips)


def to_night_summary(feedback: AIFeedback, headline: str) -> NightSummary:
    """Fit the AI feedback into the panel's headline-plus-bullets format."""
    details = list(feedback.this_night)
    details += [f"Recent nights: {sentence}" for sentence in feedback.recent_nights]
    for tip_id, reason in feedback.habit_tips:
        tip = HABIT_TIPS[tip_id]
        details.append(f"Habit to try: {tip} {reason}" if reason else f"Habit to try: {tip}")
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
                "schema": FEEDBACK_SCHEMA,
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


# Errors that would fail the same way on every model with this key, so trying a
# backup model would only waste time.
_NO_RETRY_ERRORS = {"AuthenticationError", "PermissionDeniedError"}


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
    return to_night_summary(feedback, rule_summary.headline)


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
