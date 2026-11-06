"""Night timeline chart: sleep stages with heart rate from the aligned 5-minute data."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
EPOCH = pd.Timedelta(minutes=5)
# Samples further apart than this are treated as a gap instead of being joined.
GAP_TOLERANCE = pd.Timedelta(minutes=5, seconds=30)
STAGE_ORDER = ["Awake", "REM", "Light", "Deep"]
STAGE_COLORS = ["#f28e2b", "#b07aa1", "#76b7b2", "#4e79a7"]
STAGE_NAMES = {"awake": "Awake", "rem": "REM", "light": "Light", "deep": "Deep"}
HEART_RATE_COLOR = "#e15759"


@st.cache_data(show_spinner=False)
def load_participant_signals(participant_id: str) -> pd.DataFrame | None:
    """Read one participant's aligned file, or return None when it does not exist."""
    path = ALIGNED_DIR / f"{participant_id}_aligned.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, usecols=["date", "datetime_utc", "hypnogram_class", "heart_rate"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], errors="coerce")
    return frame.dropna(subset=["date", "datetime_utc"])


def night_signals(signals: pd.DataFrame, night_date: str) -> pd.DataFrame:
    """Return one night's 5-minute epochs with display stage, epoch end, and heart-rate segment."""
    night = signals.loc[signals["date"].eq(night_date)].sort_values("datetime_utc").copy()
    night["start"] = night["datetime_utc"]
    night["end"] = night["start"] + EPOCH
    stages = night["hypnogram_class"].astype("string").str.strip().str.lower()
    night["stage"] = stages.map(STAGE_NAMES)
    night["heart_rate"] = pd.to_numeric(night["heart_rate"], errors="coerce")
    night.loc[night["heart_rate"] <= 0, "heart_rate"] = pd.NA
    # A new segment starts after every time gap so the line is not drawn across it.
    night["segment"] = night["start"].diff().gt(GAP_TOLERANCE).cumsum()
    return night[["start", "end", "stage", "heart_rate", "segment"]].reset_index(drop=True)


def stage_runs(night: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive epochs with the same stage into one block per uninterrupted run."""
    staged = night.dropna(subset=["stage"])
    if staged.empty:
        return pd.DataFrame(columns=["stage", "start", "end", "minutes"])
    new_run = staged["stage"].ne(staged["stage"].shift()) | staged["start"].diff().gt(GAP_TOLERANCE)
    runs = staged.groupby(new_run.cumsum()).agg(stage=("stage", "first"), start=("start", "min"), end=("end", "max"))
    runs["minutes"] = (runs["end"] - runs["start"]).dt.total_seconds() / 60
    return runs.reset_index(drop=True)


def timeline_chart(night: pd.DataFrame) -> alt.LayerChart:
    """Layer sleep-stage blocks with a heart-rate line on its own axis."""
    time_axis = alt.X("start:T", title="Time (UTC)", axis=alt.Axis(format="%H:%M"))
    stages = (
        alt.Chart(stage_runs(night))
        .mark_rect()
        .encode(
            x=time_axis,
            x2="end:T",
            y=alt.Y("stage:N", sort=STAGE_ORDER, title="Sleep stage"),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=STAGE_ORDER, range=STAGE_COLORS),
                legend=alt.Legend(title="Stage", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("start:T", title="From (UTC)", format="%H:%M"),
                alt.Tooltip("end:T", title="To (UTC)", format="%H:%M"),
                alt.Tooltip("minutes:Q", title="Minutes", format=".0f"),
            ],
        )
    )
    heart_rate_data = alt.Chart(night.dropna(subset=["heart_rate"])).encode(
        x=time_axis,
        y=alt.Y(
            "heart_rate:Q",
            title="Heart rate (bpm)",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(titleColor=HEART_RATE_COLOR, orient="right"),
        ),
    )
    heart_rate_line = heart_rate_data.mark_line(color=HEART_RATE_COLOR, strokeWidth=2).encode(detail="segment:O")
    # Invisible points give each 5-minute heart-rate sample its own hover tooltip.
    heart_rate_points = heart_rate_data.mark_point(filled=True, size=60, opacity=0).encode(
        tooltip=[
            alt.Tooltip("start:T", title="Time (UTC)", format="%H:%M"),
            alt.Tooltip("heart_rate:Q", title="Heart rate (bpm)"),
            alt.Tooltip("stage:N", title="Stage"),
        ],
    )
    return (
        alt.layer(stages, heart_rate_line + heart_rate_points)
        .resolve_scale(y="independent")
        .properties(height=280)
    )


def show_night_timeline(participant_id: str, night_date: str) -> None:
    """Render the timeline section for the selected participant and night."""
    st.markdown("### Night timeline")
    signals = load_participant_signals(str(participant_id))
    if signals is None:
        st.info("Timeline not available for this data source; it needs the participant's aligned 5-minute data.")
        return
    night = night_signals(signals, night_date)
    if night.empty:
        st.info("No 5-minute samples were recorded for this night.")
        return
    st.caption(
        "Colored blocks show the sleep stage through the night (5-minute resolution); the red line is heart rate. "
        "Hover for times and values. Gaps mean no data was recorded."
    )
    st.altair_chart(timeline_chart(night), use_container_width=True)
