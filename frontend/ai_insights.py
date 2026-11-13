"""Optional AI-generated sleep insights with rule-based fallback."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from insights import Fact, NightSummary, summarize


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

# Project structure:
#
# project/
# ├── .env
# ├── frontend/
# │   ├── app.py
# │   ├── ai_insights.py
# │   └── insights.py
#
PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_DIR / ".env"

load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_TIMEOUT_SECONDS = 8.0


# ---------------------------------------------------------
# Check whether AI is available
# ---------------------------------------------------------

def ai_available() -> bool:
    """Return True when an OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# Convert nightly facts into text for the AI prompt
# ---------------------------------------------------------

def _facts_to_text(facts: list[Fact]) -> str:
    """Convert Fact objects into a simple text representation."""

    if not facts:
        return "No nightly facts are available."

    lines = []

    for fact in facts:
        # Fact is a dataclass, so this safely converts all of
        # its existing fields without depending on exact field names.
        fact_data = asdict(fact)

        pieces = []

        for key, value in fact_data.items():
            pieces.append(f"{key}={value}")

        lines.append("- " + ", ".join(pieces))

    return "\n".join(lines)


# ---------------------------------------------------------
# Call OpenAI
# ---------------------------------------------------------

def generate_ai_summary(
    facts: list[Fact],
    rule_summary: NightSummary,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> NightSummary:
    """
    Generate a plain-language sleep summary using OpenAI.

    Raises an exception if the API request fails.
    The caller is responsible for falling back to the
    rule-based summary.
    """

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available.")

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    client = OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
    )

    facts_text = _facts_to_text(facts)

    prompt = f"""
You are writing a short, user-friendly explanation of wearable sleep data.

Use ONLY the measurements supplied below.

Rules:
- Do not make medical diagnoses.
- Do not give medical or treatment advice.
- Do not claim that one measurement caused another.
- Describe relationships only as patterns or changes.
- Compare the selected night with the user's own recent baseline.
- Keep the explanation concise.
- Write 2 to 4 short sentences.
- Use clear language for a general user.
- Do not invent any values.

Rule-based summary headline:
{rule_summary.headline}

Nightly facts:
{facts_text}

Write only the plain-language explanation.
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    text = response.output_text.strip()

    if not text:
        raise RuntimeError("The AI returned an empty response.")

    return NightSummary(
        headline=rule_summary.headline,
        details=[text],
    )


# ---------------------------------------------------------
# Main function used by app.py
# ---------------------------------------------------------

def get_summary(
    facts: list[Fact],
    use_ai: bool = False,
) -> tuple[NightSummary, str]:
    """
    Return either an AI-generated or rule-based summary.

    AI OFF:
        Rule-based

    AI ON but no key:
        Rule-based (AI unavailable)

    AI ON and successful:
        AI-generated

    AI ON but API fails:
        Rule-based (AI fallback)
    """

    rule_summary = summarize(facts)

    # AI toggle is OFF
    if not use_ai:
        return rule_summary, "Rule-based"

    # AI toggle is ON, but no API key exists
    if not ai_available():
        return rule_summary, "Rule-based (AI unavailable)"

    # AI toggle is ON and key exists
    try:
        ai_summary = generate_ai_summary(
            facts=facts,
            rule_summary=rule_summary,
        )

        return ai_summary, "AI-generated"

    except Exception as exc:
        # Keep the dashboard running even if OpenAI fails.
        # Error appears only in the terminal for debugging.
        print(
            f"AI insight error: "
            f"{type(exc).__name__}: {exc}"
        )

        return rule_summary, "Rule-based (AI fallback)"