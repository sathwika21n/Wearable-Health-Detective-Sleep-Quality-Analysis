"""Night timeline chart: sleep stages with heart rate from the aligned 5-minute data."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
BASELINES_PATH = PROJECT_DIR / "participant_baselines.csv"
EPOCH = pd.Timedelta(minutes=5)
# Samples further apart than this are treated as a gap instead of being joined.
GAP_TOLERANCE = pd.Timedelta(minutes=5, seconds=30)
STAGE_ORDER = ["Awake", "REM", "Light", "Deep"]
STAGE_COLORS = ["#f28e2b", "#b07aa1", "#76b7b2", "#4e79a7"]
STAGE_NAMES = {"awake": "Awake", "rem": "REM", "light": "Light", "deep": "Deep"}
HEART_RATE_COLOR = "#e15759"
TEMPERATURE_COLOR = "#4e79a7"
DEFAULT_TEMPERATURE_THRESHOLD_C = 0.5
WINDOW_NIGHT_COUNTS = {"1_day": 1, "7_days": 7, "14_days": 14, "30_days": 30}


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


def load_temperature_baselines() -> pd.DataFrame | None:
    """Read rolling participant baselines when the generated file is available."""
    if not BASELINES_PATH.exists():
        return None
    baselines = pd.read_csv(BASELINES_PATH)
    columns = ["participant_id", "sleep_date", "typical_temperature_delta_c"]
    if not set(columns).issubset(baselines.columns):
        return None
    baselines = baselines[columns].copy()
    baselines["participant_id"] = baselines["participant_id"].astype("string").str.strip()
    baselines["sleep_date"] = pd.to_datetime(
        baselines["sleep_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    baselines["typical_temperature_delta_c"] = pd.to_numeric(
        baselines["typical_temperature_delta_c"], errors="coerce"
    )
    return baselines.dropna(subset=["participant_id", "sleep_date"])


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


def temperature_trend_chart(nights: pd.DataFrame, threshold: float) -> alt.LayerChart:
    """Show nightly temperature deviations with the change-detection threshold band."""
    chart_data = nights.copy()
    chart_data["night_date"] = pd.to_datetime(chart_data["night_date"], errors="coerce")
    chart_data["observed_value"] = pd.to_numeric(chart_data["observed_value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["night_date"]).sort_values("night_date")
    chart_data["segment"] = chart_data["observed_value"].isna().cumsum()
    chart_data["baseline_value"] = pd.to_numeric(chart_data["baseline_value"], errors="coerce")
    chart_data["threshold_upper"] = chart_data["baseline_value"] + threshold
    chart_data["threshold_lower"] = chart_data["baseline_value"] - threshold
    chart_data["band_segment"] = chart_data["baseline_value"].isna().cumsum()

    base = alt.Chart(chart_data).encode(
        x=alt.X("night_date:T", title="Night"),
    )
    band = base.mark_area(color=TEMPERATURE_COLOR, opacity=0.12).encode(
        y=alt.Y(
            "threshold_upper:Q",
            title="Overall nightly temperature delta (°C)",
            scale=alt.Scale(zero=False),
        ),
        y2="threshold_lower:Q",
        detail="band_segment:O",
    )
    line = base.mark_line(color=TEMPERATURE_COLOR, strokeWidth=2).encode(
        y="observed_value:Q",
        detail="segment:O",
        tooltip=[
            alt.Tooltip("night_date:T", title="Night", format="%b %d, %Y"),
            alt.Tooltip("observed_value:Q", title="Temperature delta (°C)", format=".2f"),
        ],
    )
    points = base.mark_point(color=TEMPERATURE_COLOR, filled=True, size=48).encode(
        y="observed_value:Q",
        tooltip=[
            alt.Tooltip("night_date:T", title="Night", format="%b %d, %Y"),
            alt.Tooltip("observed_value:Q", title="Temperature delta (°C)", format=".2f"),
        ],
    )
    return alt.layer(band, line, points).properties(height=220)


def show_temperature_trend(
    data: pd.DataFrame,
    participant_id: str,
    baseline_window: str,
    selected_night: str,
) -> None:
    """Render the selected participant's temperature trend for the chosen window."""
    temperature = data.loc[
        data["participant_id"].eq(participant_id)
        & data["metric"].eq("temperature_delta_c")
    ].copy()
    if temperature.empty:
        return

    if "baseline_window" in temperature:
        temperature = temperature[temperature["baseline_window"].astype(str).eq(baseline_window)]
    window_nights = WINDOW_NIGHT_COUNTS.get(baseline_window)
    if window_nights is None:
        window_nights = next(
            (int(part) for part in baseline_window.replace("_", " ").split() if part.isdigit()),
            7,
        )
    temperature["night_date"] = pd.to_datetime(temperature["night_date"], errors="coerce")
    selected_date = pd.to_datetime(selected_night, errors="coerce")
    temperature = (
        temperature.dropna(subset=["night_date"])
        .loc[lambda frame: frame["night_date"].le(selected_date)]
        .sort_values("night_date")
        .drop_duplicates("night_date", keep="last")
        .tail(window_nights)
    )
    if temperature.empty:
        return

    temperature["observed_value"] = pd.to_numeric(temperature["observed_value"], errors="coerce")
    if "baseline_value" in temperature:
        temperature["baseline_value"] = pd.to_numeric(temperature["baseline_value"], errors="coerce")
    else:
        temperature["baseline_value"] = pd.NA
    participant_baselines = load_temperature_baselines()
    matching_baselines = pd.DataFrame()
    if participant_baselines is not None:
        matching_baselines = participant_baselines.loc[
            participant_baselines["participant_id"].eq(str(participant_id)),
            ["sleep_date", "typical_temperature_delta_c"],
        ].rename(columns={
            "sleep_date": "night_date",
            "typical_temperature_delta_c": "participant_baseline_value",
        })
        matching_baselines["night_date"] = pd.to_datetime(
            matching_baselines["night_date"], errors="coerce"
        )
        temperature = temperature.merge(
            matching_baselines,
            on="night_date",
            how="left",
            validate="one_to_one",
        )

    has_participant_baseline = (
        not matching_baselines.empty
        and temperature["participant_baseline_value"].notna().any()
    )
    if has_participant_baseline:
        temperature["baseline_value"] = temperature["participant_baseline_value"].combine_first(
            temperature["baseline_value"]
        )
    elif temperature["baseline_value"].isna().all():
        temperature["baseline_value"] = temperature["observed_value"].shift()

    threshold = DEFAULT_TEMPERATURE_THRESHOLD_C
    if "threshold_value" in temperature:
        threshold_rows = temperature
        if "threshold_type" in threshold_rows:
            threshold_rows = threshold_rows[threshold_rows["threshold_type"].eq("absolute_difference")]
        values = pd.to_numeric(threshold_rows["threshold_value"], errors="coerce").dropna()
        if not values.empty:
            threshold = float(values.iloc[0])

    st.markdown("### Overall nightly temperature delta")
    if has_participant_baseline:
        baseline_label = "prior-10-night participant baseline"
    elif temperature["baseline_value"].notna().any():
        baseline_label = "selected-window baseline"
    else:
        baseline_label = "previous night's value"
    st.caption(
        f"Each point is one overall nightly temperature value. Showing up to {window_nights} nights "
        f"through {selected_date:%b %d, %Y}. The shaded band shows the {baseline_label} ±{threshold:g} °C. "
        "Gaps indicate nights without a temperature value."
    )
    st.altair_chart(temperature_trend_chart(temperature, threshold), use_container_width=True)


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
