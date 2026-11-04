"""Rule-based, plain-language summaries of one night's change-detection results.

The functions here only read the dashboard's long-format rows for a single
participant, night, and baseline window, so they can be tested without Streamlit.
Wording describes patterns compared with the person's own recent nights; it never
diagnoses or claims that one measurement caused another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


WINDOW_PHRASES = {
    "1_day": "the night before",
    "7_days": "your average over the previous 7 nights",
    "14_days": "your average over the previous 14 nights",
    "30_days": "your average over the previous 30 nights",
}
HEADLINE_WINDOWS = {
    "1_day": "the night before",
    "7_days": "your previous 7 nights",
    "14_days": "your previous 14 nights",
    "30_days": "your previous 30 nights",
}
DIRECTION_WORDS = {"min": ("longer", "shorter"), "count": ("more", "fewer")}
SLEEP_METRICS = {
    "total_sleep_min", "deep_sleep_min", "rem_sleep_min", "light_sleep_min",
    "awake_min", "observed_awakenings_count",
}
BODY_METRICS = {"sleeping_hr_avg_bpm", "hrv_rmssd_ms", "temperature_delta_c"}
MAX_HIGHLIGHTS = 3
DISCLAIMER = (
    "Patterns only, compared with your own recent nights; not medical advice. "
    "Changes on the same night are not necessarily caused by each other."
)


@dataclass(frozen=True)
class Fact:
    """One metric's comparison for the selected night."""

    metric: str
    label: str
    unit: str
    status: str
    window: str
    observed: float | None
    baseline: float | None
    difference: float | None
    z_score: float | None
    threshold_type: str
    threshold_value: float | None
    baseline_nights: int
    minimum_nights: int | None


@dataclass
class NightSummary:
    """Headline plus supporting sentences for one night."""

    headline: str
    details: list[str] = field(default_factory=list)


def _number(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def night_facts(rows: pd.DataFrame, labels: dict[str, str] | None = None) -> list[Fact]:
    """Turn the selected night's dashboard rows into one Fact per metric."""
    labels = labels or {}
    facts = []
    for _, row in rows.iterrows():
        metric = str(row["metric"])
        label = labels.get(metric) or str(row.get("metric_label") or "") or metric.replace("_", " ").capitalize()
        minimum = _number(row.get("minimum_nights"))
        facts.append(Fact(
            metric=metric,
            label=label,
            unit=str(row.get("unit") or ""),
            status=str(row.get("status") or ""),
            window=str(row.get("baseline_window") or ""),
            observed=_number(row.get("observed_value")),
            baseline=_number(row.get("baseline_value")),
            difference=_number(row.get("difference")),
            z_score=_number(row.get("z_score")),
            threshold_type=str(row.get("threshold_type") or ""),
            threshold_value=_number(row.get("threshold_value")),
            baseline_nights=int(_number(row.get("baseline_nights")) or 0),
            minimum_nights=int(minimum) if minimum is not None else None,
        ))
    return facts


def format_amount(value: float, unit: str) -> str:
    """Format a value for reading, e.g. 85 min -> '1 h 25 min'."""
    value = abs(value) if unit == "count" else value
    if unit == "min":
        minutes = round(abs(value))
        sign = "-" if value < 0 and minutes else ""
        hours, rest = divmod(minutes, 60)
        if hours and rest:
            return f"{sign}{hours} h {rest} min"
        return f"{sign}{hours} h" if hours else f"{sign}{rest} min"
    if unit == "°C":
        # Two decimals keep small nightly deviations readable and consistent with their differences.
        rounded = round(value, 2)
        return f"{rounded:+.2f} °C" if rounded else "0.00 °C"
    shown = f"{value:.1f}".rstrip("0").rstrip(".")
    if unit == "count":
        return shown
    return f"{shown} {unit}".strip()


def describe(fact: Fact) -> str:
    """One plain-language sentence for a single metric."""
    if fact.status == "insufficient_history":
        required = fact.minimum_nights if fact.minimum_nights is not None else "more"
        return (
            f"{fact.label}: not enough history yet to compare "
            f"({fact.baseline_nights} of {required} prior nights)."
        )
    if fact.status == "missing" or fact.observed is None:
        return f"{fact.label}: not recorded this night."
    observed = format_amount(fact.observed, fact.unit)
    if fact.status == "normal" or fact.difference is None or fact.baseline is None:
        return f"{fact.label}: within your usual range ({observed})."

    up_word, down_word = DIRECTION_WORDS.get(fact.unit, ("higher", "lower"))
    direction = up_word if fact.difference > 0 else down_word
    amount = format_amount(abs(fact.difference), fact.unit).lstrip("+")
    comparison = WINDOW_PHRASES.get(fact.window, "your recent nights")
    baseline = format_amount(fact.baseline, fact.unit)
    return f"{fact.label}: {amount} {direction} than {comparison} ({observed} vs {baseline})."


def change_strength(fact: Fact) -> float:
    """How far a change is beyond its rule, so the strongest changes are listed first."""
    if fact.difference is None:
        return 0.0
    if fact.threshold_type == "absolute_z_score" and fact.z_score is not None:
        return abs(fact.z_score)
    if fact.threshold_value:
        return abs(fact.difference) / fact.threshold_value
    if fact.baseline:
        return abs(fact.difference) / abs(fact.baseline)
    return abs(fact.difference)


def _join(labels: list[str]) -> str:
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _mid_sentence(label: str) -> str:
    """Lower-case a label's first letter unless it starts an acronym like 'REM'."""
    if len(label) > 1 and label[1].isupper():
        return label
    return label[:1].lower() + label[1:]


def summarize(facts: list[Fact]) -> NightSummary:
    """Headline and sentences for the night, strongest changes first."""
    if not facts:
        return NightSummary("No measurements are available for this night.")

    window = HEADLINE_WINDOWS.get(facts[0].window, "your recent nights")
    changes = sorted((f for f in facts if f.status == "change"), key=change_strength, reverse=True)
    unavailable = [f for f in facts if f.status in {"missing", "insufficient_history"} or f.observed is None]
    compared = [f for f in facts if f not in unavailable]

    if changes:
        count = len(changes)
        noun = "measurement" if count == 1 else "measurements"
        top = changes[:MAX_HIGHLIGHTS]
        names = [f.label for f in top] + ([f"{count - len(top)} more"] if count > len(top) else [])
        summary = NightSummary(f"{count} {noun} stood out compared with {window}: {_join(names)}.")
        summary.details = [describe(f) for f in top]
        sleep_changes = [_mid_sentence(f.label) for f in changes if f.metric in SLEEP_METRICS]
        body_changes = [_mid_sentence(f.label) for f in changes if f.metric in BODY_METRICS]
        if sleep_changes and body_changes:
            summary.details.append(
                f"The change in {_join(sleep_changes)} happened on the same night as the change in "
                f"{_join(body_changes)}. They may be related, but this does not show that one caused the other."
            )
    elif compared:
        summary = NightSummary(f"Nothing stood out: every measurement that could be compared was within your usual range for {window}.")
    else:
        summary = NightSummary("There is not enough data to compare this night yet.")

    if unavailable:
        noun = "measurement" if len(unavailable) == 1 else "measurements"
        summary.details.append(
            f"{len(unavailable)} {noun} could not be compared (missing data or not enough prior nights)."
        )
    return summary
