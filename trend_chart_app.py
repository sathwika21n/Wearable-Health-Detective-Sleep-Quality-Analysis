"""Trend chart app: Streamlit sleep dashboard with a metric trend chart.

Plots a selected metric across nights with the participant's rolling baseline and
threshold band, highlighting nights flagged by change detection.

Run from the project root:
    streamlit run frontend/trend_chart_app.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st



APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SAMPLE_PATH = APP_DIR / "sample_sleep_dashboard.csv"
GENERATED_PATH = PROJECT_DIR / "change_detection_results.csv"
REQUIRED_COLUMNS = {
    "participant_id", "night_date", "metric", "unit", "observed_value",
    "baseline_value", "difference", "baseline_window", "baseline_nights",
    "minimum_nights", "status", "message",
}
METRIC_LABELS = {
    "total_sleep_min": "Total sleep",
    "deep_sleep_min": "Deep sleep",
    "rem_sleep_min": "REM sleep",
    "light_sleep_min": "Light sleep",
    "awake_min": "Awake time",
    "observed_awakenings_count": "Observed awakenings",
    "sleeping_hr_avg_bpm": "Sleeping heart rate",
    "hrv_rmssd_ms": "Heart rate variability",
    "temperature_delta_c": "Temperature deviation",
}
STATUS_LABELS = {
    "change": "Change detected",
    "normal": "Within baseline range",
    "missing": "Measurement missing",
    "insufficient_history": "Insufficient history",
}
WINDOW_LABELS = {
    "1_day": "Previous night",
    "7_days": "Previous 7 nights",
    "14_days": "Previous 14 nights",
    "30_days": "Previous 30 nights",
}


# ---------- Metric trend chart ----------
# The band reproduces the flag rule in build_change_detection_table.py:
#   absolute_difference (previous-night window): baseline +/- threshold_value
#   absolute_z_score (7/14/30-night windows):    baseline +/- threshold_value * baseline_std
# Highlighted nights come straight from the status column, so they always match change detection.

# Colors from the shared dataviz palette: one series hue, neutral baseline, status "critical" for flags
SERIES_COLOR = "#2a78d6"
BASELINE_COLOR = "#8a8984"
FLAG_COLOR = "#d03b3b"
SHORT_HISTORY_COLOR = "#b5b4ad"

STATUS_ORDER = ["normal", "change", "missing", "insufficient_history"]
STATUS_LEGEND = {
    "normal": "Within baseline range",
    "change": "Flagged change",
    "missing": "Measurement missing",
    "insufficient_history": "Insufficient history",
}
# Statuses where a baseline comparison was actually made, so the band is meaningful
COMPARED = {"normal", "change", "missing"}


def threshold_band(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Lower and upper flag limits for each row; NaN where no band applies."""
    threshold = pd.to_numeric(rows["threshold_value"], errors="coerce")
    baseline = pd.to_numeric(rows["baseline_value"], errors="coerce")
    std = pd.to_numeric(rows["baseline_std"], errors="coerce")
    kind = rows["threshold_type"].fillna("").astype(str)

    half_width = pd.Series(np.nan, index=rows.index)
    absolute = kind.eq("absolute_difference")
    half_width[absolute] = threshold[absolute]
    # A zero spread gives z = 0 in change detection (never flagged), so no band is drawn
    z_based = kind.eq("absolute_z_score") & std.gt(0)
    half_width[z_based] = threshold[z_based] * std[z_based]

    half_width[~rows["status"].isin(COMPARED)] = np.nan
    return baseline - half_width, baseline + half_width


def build_trend_frame(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per night for a single participant, metric and baseline window."""
    frame = rows.loc[rows["metric"].eq(metric)].copy()
    if frame["night_date"].duplicated().any():
        raise ValueError("Each night must appear once per metric and baseline window.")
    frame["night"] = pd.to_datetime(frame["night_date"], errors="coerce")
    frame = frame.dropna(subset=["night"]).sort_values("night").reset_index(drop=True)

    frame["band_lower"], frame["band_upper"] = threshold_band(frame)
    frame["baseline_shown"] = frame["baseline_value"].where(frame["status"].isin(COMPARED))
    frame["flagged"] = frame["status"].eq("change")
    frame["status_label"] = frame["status"].map(STATUS_LEGEND).fillna(frame["status"])
    return frame


def make_trend_chart(frame: pd.DataFrame, label: str, unit: str, selected_night: str | None = None) -> alt.LayerChart:
    """Layered Altair chart: band, dashed baseline, observed line, status points, selected night."""
    y_title = f"{label} ({unit})" if unit else label
    x = alt.X("night:T", title="Night", axis=alt.Axis(format="%b %d", labelAngle=0, grid=False))
    base = alt.Chart(frame)

    def fmt(column: str) -> alt.Tooltip:
        return alt.Tooltip(f"{column}:Q", format=".1f")

    tooltip = [
        alt.Tooltip("night:T", title="Night", format="%b %d, %Y"),
        alt.Tooltip("status_label:N", title="Status"),
        alt.Tooltip("observed_value:Q", title=f"Observed ({unit})" if unit else "Observed", format=".1f"),
        alt.Tooltip("baseline_value:Q", title="Baseline", format=".1f"),
        alt.Tooltip("band_lower:Q", title="Band low", format=".1f"),
        alt.Tooltip("band_upper:Q", title="Band high", format=".1f"),
        alt.Tooltip("z_score:Q", title="Z-score", format=".2f"),
        alt.Tooltip("baseline_nights:Q", title="Baseline nights"),
        alt.Tooltip("message:N", title="Rule"),
    ]
    hover = alt.selection_point(fields=["night"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    y_scale = alt.Scale(zero=False, nice=True)

    band = base.mark_area(color=SERIES_COLOR, opacity=0.14).encode(
        x=x, y=alt.Y("band_lower:Q", title=y_title, scale=y_scale), y2="band_upper:Q",
    )
    baseline = base.mark_line(color=BASELINE_COLOR, strokeDash=[5, 4], strokeWidth=1.5).encode(
        x=x, y="baseline_shown:Q",
    )
    observed = base.mark_line(color=SERIES_COLOR, strokeWidth=2).encode(x=x, y="observed_value:Q")

    status_scale = alt.Scale(
        domain=[STATUS_LEGEND[s] for s in STATUS_ORDER if s != "missing"],
        range=[SERIES_COLOR, FLAG_COLOR, SHORT_HISTORY_COLOR],
    )
    shape_scale = alt.Scale(
        domain=[STATUS_LEGEND[s] for s in STATUS_ORDER if s != "missing"],
        range=["circle", "diamond", "circle"],
    )
    points = base.transform_filter("isValid(datum.observed_value)").mark_point(
        filled=True, stroke="white", strokeWidth=1.5, opacity=1,
    ).encode(
        x=x,
        y="observed_value:Q",
        color=alt.Color("status_label:N", title="Night status", scale=status_scale,
                        legend=alt.Legend(orient="top", direction="horizontal")),
        shape=alt.Shape("status_label:N", title="Night status", scale=shape_scale),
        size=alt.condition("datum.flagged", alt.value(150), alt.value(45)),
        tooltip=tooltip,
    )

    # Hover crosshair: invisible wide hit targets, then a rule at the nearest night
    hit = base.mark_point(opacity=0, size=400).encode(x=x, y="observed_value:Q", tooltip=tooltip).add_params(hover)
    crosshair = base.mark_rule(color=BASELINE_COLOR, strokeWidth=1).encode(x=x).transform_filter(hover)

    layers = [band, baseline, observed, crosshair, points, hit]
    if selected_night:
        marker = pd.DataFrame({"night": [pd.Timestamp(selected_night)], "label": ["Selected night"]})
        rule = alt.Chart(marker).mark_rule(color="#52514e", strokeDash=[2, 3], strokeWidth=1.5).encode(x="night:T")
        text = alt.Chart(marker).mark_text(
            align="left", baseline="top", dx=5, color="#52514e", fontSize=11, fontWeight="bold",
        ).encode(x="night:T", y=alt.value(4), text="label:N")
        layers += [rule, text]

    return alt.layer(*layers).properties(height=360).interactive(bind_y=False)


def load_dashboard_data(uploaded_file=None) -> tuple[pd.DataFrame, str]:
    """Load upload, generated project results, or the sample fallback, in that order."""
    if uploaded_file is not None:
        source = uploaded_file
        source_name = uploaded_file.name
    elif GENERATED_PATH.exists():
        source = GENERATED_PATH
        source_name = f"{GENERATED_PATH.name} (generated project results)"
    else:
        source = SAMPLE_PATH
        source_name = f"{SAMPLE_PATH.name} (sample fallback)"
    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype("string")
    frame["night_date"] = pd.to_datetime(frame["night_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["metric"] = frame["metric"].astype(str)
    frame["unit"] = frame["unit"].fillna("").astype(str)
    frame["status"] = frame["status"].fillna("").astype(str).str.lower()
    for column in ["observed_value", "baseline_value", "difference", "baseline_nights", "minimum_nights"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["message"] = frame["message"].fillna("").astype(str)
    for column in ["baseline_std", "z_score", "threshold_value"]:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "threshold_type" not in frame:
        frame["threshold_type"] = ""
    valid_statuses = set(STATUS_LABELS)
    invalid_statuses = set(frame["status"].dropna().unique()) - valid_statuses
    if invalid_statuses:
        raise ValueError(f"Unsupported status values: {', '.join(sorted(invalid_statuses))}")
    missing_observation = frame["observed_value"].isna() & ~frame["status"].eq("insufficient_history")
    frame.loc[missing_observation, "status"] = "missing"
    frame = frame.dropna(subset=["participant_id", "night_date"])
    return frame, source_name


def format_value(value, unit: str) -> str:
    if pd.isna(value):
        return "—"
    number = float(value)
    shown = f"{number:.1f}" if number % 1 else f"{number:.0f}"
    return f"{shown} {unit}".strip()


def show_metric_card(row: pd.Series) -> None:
    metric = str(row["metric"])
    label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    status = str(row["status"])
    observed = format_value(row["observed_value"], row["unit"])
    baseline = format_value(row["baseline_value"], row["unit"])
    diff = format_value(row["difference"], row["unit"])
    status_label = STATUS_LABELS.get(status, status.replace("_", " ").title() or "Status unavailable")

    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.metric("Observed", observed)
        if status in {"missing", "insufficient_history"}:
            st.info(status_label)
            if row["message"]:
                st.caption(row["message"])
            if status == "insufficient_history":
                used = int(row["baseline_nights"]) if pd.notna(row["baseline_nights"]) else 0
                required = int(row["minimum_nights"]) if pd.notna(row["minimum_nights"]) else "—"
                st.caption(f"Baseline history: {used} of {required} nights required.")
        else:
            st.caption(f"Baseline: {baseline} · Difference: {diff}")
            if pd.notna(row["threshold_value"]):
                if row["threshold_type"] == "absolute_difference":
                    st.caption(f"Flag rule: |difference| ≥ {format_value(row['threshold_value'], row['unit'])}")
                elif row["threshold_type"] == "absolute_z_score":
                    z_text = format_value(row["z_score"], "z")
                    st.caption(f"Flag rule: |z-score| ≥ {row['threshold_value']:g} · Current z-score: {z_text}")
            if status == "change":
                st.warning(status_label, icon="⚠️")
                if row["message"]:
                    st.caption(row["message"])
            else:
                st.success(status_label, icon="✅")


st.set_page_config(page_title="Sleep Detective Dashboard", page_icon="🌙", layout="wide")
st.title("🌙 Sleep Detective Dashboard")
st.caption("Explore nightly sleep metrics and personal baseline changes.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload dashboard CSV", type=["csv"])
    if uploaded is not None:
        st.caption("Using uploaded CSV.")
    elif GENERATED_PATH.exists():
        st.caption("Using the generated baseline and change-detection table.")
    else:
        st.caption("Generated results not found; using documented sample data.")

try:
    data, source_name = load_dashboard_data(uploaded)
except (OSError, pd.errors.ParserError, ValueError) as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

if data.empty:
    st.warning("No dashboard records are available in this file.")
    st.stop()

with st.sidebar:
    st.caption(f"Source: `{source_name}` · {len(data):,} metric records")
    participants = sorted(data["participant_id"].unique())
    participant = st.selectbox("Participant", participants)

participant_data = data.loc[data["participant_id"].eq(participant)]
windows = sorted(participant_data["baseline_window"].dropna().astype(str).unique())
if not windows:
    st.warning("No baseline-window values are available for this participant.")
    st.stop()
default_window = next((value for value in windows if value == "7_days"), windows[0])
with st.sidebar:
    baseline_window = st.selectbox(
        "Baseline window",
        windows,
        index=windows.index(default_window),
        format_func=lambda value: WINDOW_LABELS.get(value, value.replace("_", " ").title()),
    )
participant_data = participant_data.loc[participant_data["baseline_window"].astype(str).eq(baseline_window)]
dates = sorted(participant_data["night_date"].dropna().unique(), reverse=True)
if not dates:
    st.warning("No valid night dates are available for this participant.")
    st.stop()

with st.sidebar:
    night = st.selectbox("Night", dates, format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"))

selected = participant_data.loc[participant_data["night_date"].eq(night)].copy()
if selected["metric"].duplicated(keep=False).any():
    st.error("This participant-night has duplicate metric rows. Keep one row per metric in the input.")
    st.stop()

st.subheader(f"Participant {participant} · Night of {pd.Timestamp(night).strftime('%B %d, %Y')}")
st.caption("Units appear beside each observed value. Differences are observed minus baseline.")

changes = selected.loc[selected["status"].eq("change")]
missing = selected.loc[selected["status"].eq("missing") | selected["observed_value"].isna()]
insufficient = selected.loc[selected["status"].eq("insufficient_history")]
summary_cols = st.columns(3)
summary_cols[0].metric("Metrics shown", len(selected))
summary_cols[1].metric("Changes detected", len(changes))
summary_cols[2].metric("Data gaps / short history", len(missing) + len(insufficient))

if selected.empty:
    st.warning("No metrics are available for the selected night.")
else:
    preferred_order = list(METRIC_LABELS)
    selected["_order"] = selected["metric"].map({name: i for i, name in enumerate(preferred_order)}).fillna(len(preferred_order))
    selected = selected.sort_values(["_order", "metric"]).drop(columns="_order")
    st.markdown("### Nightly metrics and comparisons")
    for start in range(0, len(selected), 4):
        cards = st.columns(4)
        for card, (_, row) in zip(cards, selected.iloc[start:start + 4].iterrows()):
            with card:
                show_metric_card(row)

    st.markdown("### Change-detection details")
    if changes.empty:
        st.info("No changes were flagged for this night.")
    else:
        details = changes.copy()
        details["threshold"] = details.apply(
            lambda row: (
                f"|difference| ≥ {format_value(row['threshold_value'], row['unit'])}"
                if row["threshold_type"] == "absolute_difference" and pd.notna(row["threshold_value"])
                else f"|z-score| ≥ {row['threshold_value']:g}"
                if row["threshold_type"] == "absolute_z_score" and pd.notna(row["threshold_value"])
                else "—"
            ),
            axis=1,
        )
        details["metric"] = details["metric"].map(METRIC_LABELS).fillna(details["metric"])
        details = details.rename(columns={
            "metric": "Metric", "observed_value": "Observed", "baseline_value": "Baseline",
            "difference": "Difference", "unit": "Unit", "baseline_window": "Baseline window",
            "threshold": "Applied threshold", "baseline_std": "Baseline SD",
            "z_score": "Z-score", "message": "Details",
        })
        st.dataframe(
            details[[
                "Metric", "Observed", "Baseline", "Difference", "Unit", "Baseline window",
                "Applied threshold", "Baseline SD", "Z-score", "Details",
            ]],
            hide_index=True,
            use_container_width=True,
        )

    if not missing.empty or not insufficient.empty:
        st.markdown("### Data availability")
        if not missing.empty:
            labels = ", ".join(METRIC_LABELS.get(value, value) for value in missing["metric"])
            st.warning(f"Missing measurements: {labels}. No baseline difference is shown for these metrics.")
        if not insufficient.empty:
            labels = ", ".join(METRIC_LABELS.get(value, value) for value in insufficient["metric"])
            st.info(f"Insufficient history: {labels}. A baseline comparison needs more prior nights.")

st.markdown("### Metric trend across nights")
window_label = WINDOW_LABELS.get(baseline_window, baseline_window.replace("_", " ").title())
metric_order = {name: i for i, name in enumerate(METRIC_LABELS)}
trend_metrics = sorted(participant_data["metric"].dropna().unique(), key=lambda m: (metric_order.get(m, len(metric_order)), m))
if not trend_metrics:
    st.info("No metrics are available to plot for this participant.")
else:
    trend_metric = st.selectbox(
        "Metric",
        trend_metrics,
        format_func=lambda value: METRIC_LABELS.get(value, value.replace("_", " ").title()),
        key="trend_metric",
    )
    trend_label = METRIC_LABELS.get(trend_metric, trend_metric.replace("_", " ").title())
    try:
        trend = build_trend_frame(participant_data, trend_metric)
    except ValueError as exc:
        st.error(f"Could not plot this metric: {exc}")
        st.stop()
    trend_unit = next(iter(trend["unit"].dropna().astype(str)), "")

    flagged_nights = trend.loc[trend["flagged"]]
    compared_nights = trend["status"].isin(["normal", "change"]).sum()
    trend_cols = st.columns(3)
    trend_cols[0].metric("Nights plotted", len(trend))
    trend_cols[1].metric("Flagged nights", len(flagged_nights))
    trend_cols[2].metric("Nights compared", int(compared_nights))

    st.caption(
        f"Baseline window: **{window_label}** (change it in the sidebar). "
        "Solid line = observed value · dashed line = rolling personal baseline · "
        "shaded band = threshold range; nights on or outside it are flagged (red diamonds). "
        "Grey points had too few prior nights for a comparison."
    )
    if trend["band_lower"].isna().all() and compared_nights:
        st.caption("This file has no threshold or baseline spread columns, so the threshold band can't be drawn.")
    st.altair_chart(
        make_trend_chart(trend, trend_label, trend_unit, selected_night=night),
        width="stretch",
    )

    with st.expander(f"Flagged nights for {trend_label} ({len(flagged_nights)})"):
        if flagged_nights.empty:
            st.write("No nights were flagged for this metric and baseline window.")
        else:
            st.dataframe(
                flagged_nights.rename(columns={
                    "night_date": "Night", "observed_value": "Observed", "baseline_value": "Baseline",
                    "band_lower": "Band low", "band_upper": "Band high", "difference": "Difference",
                    "z_score": "Z-score", "message": "Rule",
                })[["Night", "Observed", "Baseline", "Band low", "Band high", "Difference", "Z-score", "Rule"]],
                hide_index=True,
                width="stretch",
            )
