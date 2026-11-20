"""Optional AI-generated sleep feedback with a rule-based fallback.

The AI writes three kinds of feedback, all from results the dashboard has already
computed:

1. This night: what changed compared with the person's own baseline, and which
   changes happened on the same night.
2. Recent nights: how this night fits the participant's previous nights.
3. Habit tips: general, non-medical sleep habits, picked by the model from the
   fixed list in ``HABIT_TIPS``. The model only chooses tip ids and gives a short
   reason; the tip wording itself always comes from this file.

The reply is structured JSON, and the model never writes numbers itself: it
writes placeholders such as ``{deep_sleep_min.observed}`` and this module fills
them in from the data, so every number shown comes from the results. Before
that, the reply is checked: no digits of its own, each placeholder next to the
name of its own measurement, direction words that match the data, every
sentence about a measurement, no diagnosis, condition or cause-and-effect
wording, no recent-night claims without history, and known tip ids only.
Anything that fails a check, times out or errors falls back to the rule-based
summary, so the dashboard never breaks.

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

# Words that would make the feedback sound like a diagnosis, a health claim,
# treatment or a cause-and-effect claim. Matching is case-insensitive on word starts.
BANNED_PATTERNS = [
    # diagnosis, conditions and symptoms
    r"\bdiagnos", r"\bdisorder", r"\bcondition", r"\bsyndrome", r"\bdisease", r"\billness",
    r"\bsick", r"\bsymptom", r"\bchronic", r"\binsomnia", r"\bapnea", r"\bapnoea", r"\bnarcolep",
    r"\brestless leg", r"\bhypertens", r"\bhypotens", r"\bblood pressure", r"\barrhythm",
    r"\btachycard", r"\bbradycard", r"\bfibrillat", r"\bafib\b",
    r"\bheart (?:problem|disease|attack|failure|condition)",
    r"\bdepress", r"\banxiety", r"\banxious", r"\bburnout", r"\bfever", r"\binfect", r"\bvirus",
    r"\bcovid", r"\bflu\b", r"\bdeprivation",
    # health judgements and warnings
    r"\bsigns? of\b", r"\bindicat", r"\bwarning", r"\brisk", r"\bdanger", r"\bconcern", r"\bworr",
    r"\balarming", r"\bserious", r"\bunhealthy", r"\babnormal", r"\bhealth (?:problem|issue)",
    r"\bsuggest", r"\bmeans you\b", r"\bpoints? to\b", r"\bstress", r"\bfatigue", r"\bexhaust",
    r"\brecover", r"\bimmune",
    # medical care and treatment
    r"\bmedical", r"\bdoctor", r"\bphysician", r"\bclinic", r"\bmedicat", r"\bprescri", r"\btreat",
    r"\bcure", r"\btherap", r"\bsupplement", r"\bmelatonin", r"\bpill",
    # cause and effect
    r"\bcaused\b", r"\bcauses\b", r"\bbecause\b", r"\bdue to\b", r"\bled to\b", r"\bleads to\b",
    r"\bresult(?:s|ed)? in\b", r"\bresult of\b", r"\btriggered", r"\bexplains?\b",
    # claims about the person rather than the measurements
    r"\byou(?:'re| are)? (?:may |might |could |likely |probably |possibly )?(?:have|has|suffer|be sick|are sick)\b",
    r"\byou(?:'re| are) (?:sick|ill|unwell)\b",
]
_BANNED_RE = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\.([a-z_]+)\}")
# Spelled-out amounts would be numbers the data didn't give ("twenty minutes", "half").
_NUMBER_WORDS = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                 r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|a few|several|a couple of")
_SPELLED_AMOUNT_RE = re.compile(
    rf"\b(?:{_NUMBER_WORDS})\s+(?:more\s+|fewer\s+|extra\s+)?(?:min|mins|minutes?|hours?|hrs?|bpm|beats|ms|"
    r"milliseconds?|degrees?|percent|times)\b"
    r"|\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|twice|double|"
    r"triple|quarter|dozen|percent|percentage)\b",
    re.IGNORECASE,
)
_MEASUREMENT_RE = re.compile(r"\bmeasurements?\b", re.IGNORECASE)
UP_WORDS = {"rose", "risen", "rise", "rises", "rising", "increased", "increase", "increases", "higher",
            "more", "longer", "up", "climbed", "above", "gained", "jumped", "longest", "highest", "most"}
DOWN_WORDS = {"fell", "fallen", "fall", "falls", "dropped", "drop", "drops", "decreased", "decrease",
              "decreases", "lower", "less", "fewer", "shorter", "down", "declined", "below", "lost",
              "shortest", "lowest", "least"}

# Other names the model may use for a measurement besides its dashboard label.
METRIC_ALIASES = {
    "total_sleep_min": ["total sleep", "sleep time"],
    "rem_sleep_min": ["rem"],
    "awake_min": ["time awake", "awake time"],
    "observed_awakenings_count": ["awakenings"],
    "sleeping_hr_avg_bpm": ["heart rate"],
    "hrv_rmssd_ms": ["hrv", "heart rate variability"],
    "temperature_delta_c": ["temperature"],
}

# Placeholder fields the model may use, and which payload value each one shows.
THIS_NIGHT_FIELDS = {
    "observed": "observed", "usual": "baseline", "change": "difference",
    "prior_nights": "prior_nights", "nights_needed": "nights_needed",
}
RECENT_FIELDS = {
    "recent_average": "previous_average", "recent_lowest": "previous_lowest",
    "recent_highest": "previous_highest", "recent_flagged": "previous_nights_flagged",
}

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
- Never write digits or spelled-out amounts. To show a number, copy one of the
  placeholders listed in the data, such as {{deep_sleep_min.observed}}; it is
  filled in with the value and unit, and {{x.change}} already includes its
  direction (for example "1 h 18 min shorter"). Put each placeholder right after
  the name of its own measurement, e.g. "Deep sleep was {{deep_sleep_min.change}}
  than usual." Do not calculate new numbers, percentages or averages.
- Every sentence, and every habit tip reason, must be about at least one of the
  measurements, named the way the data names it.
- Never make claims about the person's health. Only describe the measurements.
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
    entry = {key: value for key, value in entry.items() if value is not None and value != ""}
    entry["placeholders"] = [
        f"{{{fact.metric}.{name}}}" for name, key in THIS_NIGHT_FIELDS.items() if key in entry
    ]
    return entry


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
            "placeholders": [f"{{{fact.metric}.{name}}}" for name in RECENT_FIELDS],
        })
    if not metrics:
        return None
    return {"nights_included": len(nights), "placeholders": ["{recent.nights}"], "metrics": metrics}


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

def _change_text(difference: float, unit: str) -> str:
    """'1 h 18 min shorter', '10 bpm higher', '2 more' or 'unchanged'."""
    if not difference:
        return "unchanged"
    up_word, down_word = DIRECTION_WORDS.get(unit, ("higher", "lower"))
    amount = format_amount(abs(difference), unit).lstrip("+")
    return f"{amount} {up_word if difference > 0 else down_word}"


def placeholder_values(payload: dict) -> dict[str, str]:
    """What each placeholder the model may use shows, e.g. 'deep_sleep_min.observed' -> '1 h 32 min'."""
    values = {}
    for entry in payload.get("this_night", []):
        ref, unit = entry["ref"], entry.get("unit", "")
        for name, key in THIS_NIGHT_FIELDS.items():
            if key not in entry:
                continue
            if name == "change":
                values[f"{ref}.{name}"] = _change_text(entry[key], unit)
            elif name in {"prior_nights", "nights_needed"}:
                values[f"{ref}.{name}"] = str(int(entry[key]))
            else:
                values[f"{ref}.{name}"] = format_amount(entry[key], unit)
    recent = payload.get("recent_nights")
    if recent:
        values["recent.nights"] = str(int(recent["nights_included"]))
        for metric in recent["metrics"]:
            ref, unit = metric["ref"], metric.get("unit", "")
            for name, key in RECENT_FIELDS.items():
                if metric.get(key) is None:
                    continue
                shown = str(int(metric[key])) if name == "recent_flagged" else format_amount(metric[key], unit)
                values[f"{ref}.{name}"] = shown
    return values


def _metric_names(payload: dict) -> list[tuple[re.Pattern, str]]:
    """(pattern, ref) for every name a measurement in the payload may go by, longest first."""
    names: dict[str, str] = {}
    for entry in payload.get("this_night", []):
        for name in [entry["metric"], *METRIC_ALIASES.get(entry["ref"], [])]:
            names.setdefault(name.lower(), entry["ref"])
    ordered = sorted(names.items(), key=lambda item: len(item[0]), reverse=True)
    return [(re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ref) for name, ref in ordered]


def _name_spans(text: str, names: list[tuple[re.Pattern, str]]) -> list[tuple[int, int, str]]:
    """Where each measurement is named in the text; longer names win over names inside them."""
    spans: list[tuple[int, int, str]] = []
    for pattern, ref in names:
        for match in pattern.finditer(text):
            if all(match.end() <= start or match.start() >= end for start, end, _ in spans):
                spans.append((match.start(), match.end(), ref))
    return sorted(spans)


def _governing_name(spans, position: int):
    """The measurement a placeholder belongs to: the last one named before it, else the first after."""
    before = [span for span in spans if span[1] <= position]
    if before:
        return before[-1]
    after = [span for span in spans if span[0] >= position]
    return after[0] if after else None


def _direction_words(segment: str) -> tuple[bool, bool]:
    words = set(re.findall(r"[a-z]+", _PLACEHOLDER_RE.sub(" ", segment).lower()))
    return bool(words & UP_WORDS), bool(words & DOWN_WORDS)


def _check_text(text: str, values: dict[str, str], names, directions, is_reason: bool) -> list[str]:
    problems = []
    if len(text) > MAX_SENTENCE_CHARS:
        problems.append("a sentence is too long")
    banned = _BANNED_RE.search(text)
    if banned:
        problems.append(f"banned wording: {banned.group(0)!r}")

    without_placeholders = _PLACEHOLDER_RE.sub(" ", text)
    if _DIGIT_RE.search(without_placeholders):
        problems.append("numbers must come from placeholders, not be written out")
    if "{" in without_placeholders or "}" in without_placeholders:
        problems.append("malformed placeholder")
    spelled = _SPELLED_AMOUNT_RE.search(without_placeholders)
    if spelled:
        problems.append(f"amounts must come from placeholders: {spelled.group(0)!r}")

    spans = _name_spans(text, names)
    placeholders = list(_PLACEHOLDER_RE.finditer(text))
    if not placeholders and not spans and (is_reason or not _MEASUREMENT_RE.search(text)):
        problems.append("a sentence isn't about any of the measurements")

    for match in placeholders:
        ref, name = match.group(1), match.group(2)
        if f"{ref}.{name}" not in values:
            problems.append(f"unknown placeholder: {match.group(0)}")
            continue
        if ref == "recent":
            continue
        owner = _governing_name(spans, match.start())
        if owner is None or owner[2] != ref:
            problems.append(f"{match.group(0)} isn't next to the name of its own measurement")
            continue
        sign = directions.get(ref)
        if sign and name in {"observed", "change"}:
            # Words between the measurement's name and its number must not point the other way.
            start, end = (owner[1], match.start()) if owner[1] <= match.start() else (match.end(), owner[0])
            up, down = _direction_words(text[start:end])
            if (sign > 0 and down) or (sign < 0 and up):
                problems.append(f"direction words around {match.group(0)} contradict the data")
    return problems


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

    values = placeholder_values(payload)
    names = _metric_names(payload)
    directions = {
        entry["ref"]: (1 if entry["difference"] > 0 else -1)
        for entry in payload.get("this_night", [])
        if entry.get("status") == "change" and entry.get("difference")
    }
    for text in feedback.this_night + feedback.recent_nights:
        problems += _check_text(text, values, names, directions, is_reason=False)
    for _, reason in feedback.habit_tips:
        if reason:
            problems += _check_text(reason, values, names, directions, is_reason=True)
    return list(dict.fromkeys(problems))


def render(text: str, values: dict[str, str]) -> str:
    """Fill the placeholders in with the values from the data."""
    return _PLACEHOLDER_RE.sub(lambda match: values[f"{match.group(1)}.{match.group(2)}"], text)


def parse_feedback(text: str) -> AIFeedback:
    """Turn the model's JSON text into AIFeedback, or raise FeedbackError.

    The reply must match FEEDBACK_SCHEMA exactly: all three keys, nothing else,
    lists of strings, and habit tips with exactly a tip_id and a reason.
    """
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FeedbackError(f"reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise FeedbackError("reply is not a JSON object")
    expected = set(FEEDBACK_SCHEMA["required"])
    if set(data) != expected:
        missing, extra = expected - set(data), set(data) - expected
        raise FeedbackError(f"reply keys don't match the schema (missing {sorted(missing)}, extra {sorted(extra)})")

    def strings(key: str) -> list[str]:
        value = data[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise FeedbackError(f"{key} must be a list of strings")
        return [item.strip() for item in value if item.strip()]

    tips = data["habit_tips"]
    if not isinstance(tips, list):
        raise FeedbackError("habit_tips must be a list")
    parsed_tips = []
    for tip in tips:
        if (not isinstance(tip, dict) or set(tip) != {"tip_id", "reason"}
                or not isinstance(tip["tip_id"], str) or not isinstance(tip["reason"], str)):
            raise FeedbackError("each habit tip needs exactly a tip_id and a reason, both strings")
        parsed_tips.append((tip["tip_id"], tip["reason"].strip()))
    return AIFeedback(strings("this_night"), strings("recent_nights"), parsed_tips)


def to_night_summary(feedback: AIFeedback, headline: str, payload: dict) -> NightSummary:
    """Fit the AI feedback into the panel's headline-plus-bullets format, with the numbers filled in."""
    values = placeholder_values(payload)
    details = [render(sentence, values) for sentence in feedback.this_night]
    details += [f"Recent nights: {render(sentence, values)}" for sentence in feedback.recent_nights]
    for tip_id, reason in feedback.habit_tips:
        tip = HABIT_TIPS[tip_id]
        reason = render(reason, values)
        reason = reason[:1].upper() + reason[1:]
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
