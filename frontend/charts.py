"""Night timeline chart: sleep stages with heart rate from the aligned 5-minute data."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
BASELINES_PATH = PROJECT_DIR / "outputs" / "participant_baselines.csv"
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
COMPARISONS = {
    "Total sleep vs. sleeping heart rate": ("total_sleep_min", "sleeping_hr_avg_bpm"),
    "Total sleep vs. temperature deviation": ("total_sleep_min", "temperature_delta_c"),
    "Deep sleep vs. sleeping heart rate": ("deep_sleep_min", "sleeping_hr_avg_bpm"),
    "Temperature deviation vs. sleeping heart rate": ("temperature_delta_c", "sleeping_hr_avg_bpm"),
}
COMPARISON_LABELS = {
    "total_sleep_min": "Total sleep (min)",
    "deep_sleep_min": "Deep sleep (min)",
    "sleeping_hr_avg_bpm": "Sleeping heart rate (bpm)",
    "temperature_delta_c": "Temperature deviation (°C)",
}


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
        st.warning("The participant baseline file is missing required columns; using the dashboard baseline instead.")
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


def comparison_pairs(data: pd.DataFrame, x_metric: str, y_metric: str) -> pd.DataFrame:
    """Pair finite observations from the same night for two metrics."""
    rows = data.loc[data["metric"].isin([x_metric, y_metric]), ["night_date", "metric", "observed_value"]].copy()
    if rows.duplicated(["night_date", "metric"]).any():
        raise ValueError("Duplicate metric rows were found for a night in the selected baseline window.")
    rows["observed_value"] = pd.to_numeric(rows["observed_value"], errors="coerce")
    rows["observed_value"] = rows["observed_value"].replace([np.inf, -np.inf], np.nan)
    pairs = rows.pivot(index="night_date", columns="metric", values="observed_value")
    pairs = pairs.reindex(columns=[x_metric, y_metric]).dropna()
    return pairs.reset_index()


def show_metric_comparison(data: pd.DataFrame, selected_night: str) -> None:
    """Display same-night metric associations for the selected participant."""
    st.markdown("### Compare nightly metrics")
    comparison = st.selectbox("Choose two measurements to compare", list(COMPARISONS))
    x_metric, y_metric = COMPARISONS[comparison]
    x_label, y_label = COMPARISON_LABELS[x_metric], COMPARISON_LABELS[y_metric]
    st.markdown(
        f"**How to read this:** Each dot is one night. Read **{x_label}** along the bottom "
        f"and **{y_label}** up the side. Hover over a dot to see its date and values."
    )
    try:
        pairs = comparison_pairs(data, x_metric, y_metric)
    except ValueError as exc:
        st.error(f"Cannot display this comparison: {exc}")
        return

    if pairs.empty:
        st.info("No nights have both measurements for this comparison. Try a different comparison or participant.")
        return

    first_night = pd.Timestamp(pairs["night_date"].min()).strftime("%b %d, %Y")
    last_night = pd.Timestamp(pairs["night_date"].max()).strftime("%b %d, %Y")
    st.caption(
        f"Paired nights: {len(pairs)} · {first_night} to {last_night}. "
        "This uses all available nights for the selected participant and baseline-window selection, "
        "not just the selected night or the previous 7, 14, or 30 nights."
    )

    base = alt.Chart(pairs).encode(
        x=alt.X(f"{x_metric}:Q", title=x_label, scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=6)),
        y=alt.Y(f"{y_metric}:Q", title=y_label, scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=6)),
    )
    points = base.mark_circle(size=60, opacity=0.55, color="#4e79a7").encode(
        tooltip=[
            alt.Tooltip("night_date:N", title="Night"),
            alt.Tooltip(f"{x_metric}:Q", title=x_label, format=".2f"),
            alt.Tooltip(f"{y_metric}:Q", title=y_label, format=".2f"),
        ],
    )
    has_correlation = len(pairs) >= 3 and pairs[x_metric].nunique() >= 2 and pairs[y_metric].nunique() >= 2
    layers = [points]
    if has_correlation:
        layers.append(
            base.transform_regression(x_metric, y_metric).mark_line(
                color="#555555", strokeDash=[6, 4], strokeWidth=2
            )
        )
    if pairs["night_date"].eq(selected_night).any():
        layers.append(
            base.transform_filter(alt.datum.night_date == selected_night)
            .mark_point(shape="diamond", filled=True, size=180, color="#d55e00", stroke="white", strokeWidth=1.5)
            .encode(
                tooltip=[
                    alt.Tooltip("night_date:N", title="Selected night"),
                    alt.Tooltip(f"{x_metric}:Q", title=x_label, format=".2f"),
                    alt.Tooltip(f"{y_metric}:Q", title=y_label, format=".2f"),
                ]
            )
        )
        selected_note = "The orange diamond marks the selected night."
    else:
        selected_note = "The selected night is not shown because it does not have both measurements."
    st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
    line_note = " The dashed line summarizes the overall straight-line pattern." if has_correlation else ""
    st.caption(f"{selected_note}{line_note}")
    if not has_correlation:
        st.info("A correlation is unavailable: at least 3 paired nights and variation in both measurements are needed.")
        return

    correlation = pairs[x_metric].corr(pairs[y_metric])
    if abs(correlation) < 0.3:
        relationship = "There is little or no clear straight-line pattern between these measurements."
    elif correlation > 0:
        relationship = f"On nights with higher {x_label}, {y_label} also tends to be higher."
    else:
        relationship = f"On nights with higher {x_label}, {y_label} tends to be lower."
    st.markdown(f"**What these nights show:** {relationship}")
    st.caption(
        f"Pearson correlation r = {correlation:.2f} (from −1 to +1; values near 0 indicate little linear association). "
        "This is an association, not evidence that one measurement causes the other."
    )


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
    """Show nightly temperatures, the reference band, and flagged nights."""
    chart_data = nights.copy()
    chart_data["night_date"] = pd.to_datetime(chart_data["night_date"], errors="coerce")
    chart_data["observed_value"] = pd.to_numeric(chart_data["observed_value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["night_date"]).sort_values("night_date")
    chart_data["night_label"] = chart_data["night_date"].dt.strftime("%Y-%m-%d")
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
            alt.Tooltip("night_label:N", title="Night"),
            alt.Tooltip("observed_value:Q", title="Temperature delta (°C)", format=".2f"),
            alt.Tooltip("status:N", title="Change status"),
        ],
    )
    points = base.mark_point(color=TEMPERATURE_COLOR, filled=True, size=48).encode(
        y="observed_value:Q",
        tooltip=[
            alt.Tooltip("night_label:N", title="Night"),
            alt.Tooltip("observed_value:Q", title="Temperature delta (°C)", format=".2f"),
            alt.Tooltip("status:N", title="Change status"),
        ],
    )
    flagged = base.transform_filter(alt.datum.status == "change").mark_point(
        color="#d62728", shape="diamond", filled=True, size=150
    ).encode(
        y="observed_value:Q",
        tooltip=[
            alt.Tooltip("night_label:N", title="Night"),
            alt.Tooltip("observed_value:Q", title="Temperature delta (°C)", format=".2f"),
            alt.Tooltip("status:N", title="Change status"),
        ],
    )
    return alt.layer(band, line, points, flagged).properties(height=220)


def show_selected_temperature_status(row: pd.Series | None) -> None:
    """Explain the selected night's supplied change-detection result."""
    if row is None:
        st.info("No temperature measurement is available for the selected night.")
        return
    status = row["status"]
    if status == "change":
        details = []
        for column, label in [
            ("observed_value", "observed"),
            ("baseline_value", "selected-window baseline"),
            ("difference", "difference"),
        ]:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                details.append(f"{label} {value:+.2f} °C")
        threshold = pd.to_numeric(row.get("threshold_value"), errors="coerce")
        z_score = pd.to_numeric(row.get("z_score"), errors="coerce")
        if row.get("threshold_type") == "absolute_z_score" and pd.notna(threshold):
            if pd.notna(z_score):
                details.append(f"z-score {z_score:+.2f}")
            details.append(f"flag rule |z-score| ≥ {threshold:g}")
        elif row.get("threshold_type") == "absolute_difference" and pd.notna(threshold):
            details.append(f"flag rule |difference| ≥ {threshold:g} °C")
        suffix = f" ({'; '.join(details)})" if details else ""
        st.warning(f"Temperature change detected for the selected night{suffix}.")
    elif status == "normal":
        st.success("No temperature change was detected for the selected night.")
    elif status == "insufficient_history":
        st.info("Temperature change cannot be assessed for the selected night: not enough baseline history.")
    else:
        st.info("Temperature change cannot be assessed for the selected night: measurement missing.")


def show_temperature_trend(
    data: pd.DataFrame,
    participant_id: str,
    baseline_window: str,
    selected_night: str,
    use_participant_baselines: bool,
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

    selected_rows = temperature.loc[temperature["night_date"].eq(selected_date)]
    selected_temperature = selected_rows.iloc[-1].copy() if not selected_rows.empty else None
    temperature["observed_value"] = pd.to_numeric(temperature["observed_value"], errors="coerce")
    if "baseline_value" in temperature:
        temperature["baseline_value"] = pd.to_numeric(temperature["baseline_value"], errors="coerce")
    else:
        temperature["baseline_value"] = pd.NA
    participant_baselines = load_temperature_baselines() if use_participant_baselines else None
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
        baseline_label = "prior-10-night participant baseline (or selected-window baseline where unavailable)"
    elif temperature["baseline_value"].isna().all():
        temperature["baseline_value"] = temperature["observed_value"].shift()
        baseline_label = "previous night's value"
    else:
        baseline_label = "selected-window baseline"

    threshold = DEFAULT_TEMPERATURE_THRESHOLD_C
    if "threshold_value" in temperature:
        threshold_rows = temperature
        if "threshold_type" in threshold_rows:
            threshold_rows = threshold_rows[threshold_rows["threshold_type"].eq("absolute_difference")]
        values = pd.to_numeric(threshold_rows["threshold_value"], errors="coerce").dropna()
        if not values.empty:
            threshold = float(values.iloc[0])

    st.markdown("### Overall nightly temperature delta")
    show_selected_temperature_status(selected_temperature)
    flagged_nights = (temperature["status"].eq("change") & temperature["observed_value"].notna()).sum()
    st.caption(
        f"Each point is one overall nightly temperature value. Showing up to {window_nights} nights "
        f"through {selected_date:%b %d, %Y}. The shaded band shows the {baseline_label} ±{threshold:g} °C. "
        "It is a visual reference, not necessarily the selected window's flag rule. "
        f"Red diamonds mark {flagged_nights} night(s) flagged by that rule. "
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
