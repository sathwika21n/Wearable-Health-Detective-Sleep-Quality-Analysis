"""Streamlit sleep dashboard backed by a documented long-format CSV."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_insights import get_summary
from charts import show_metric_comparison, show_night_timeline, show_temperature_trend
from data_loader import (
    APPLE_SOURCE, GENERATED_PATH, OURA_SOURCE, apple_output_directory,
    load_apple_nightly_features, load_dashboard_data,
)
from insights import DISCLAIMER, night_facts
from report import build_participant_report, report_filename


METRIC_LABELS = {
    "total_sleep_min": "Total sleep",
    "deep_sleep_min": "Deep sleep",
    "rem_sleep_min": "REM sleep",
    "light_sleep_min": "Light sleep",
    "awake_min": "Awake time",
    "observed_awakenings_count": "Observed awakenings",
    "sleeping_hr_avg_bpm": "Sleeping heart rate",
    "hrv_rmssd_ms": "Heart rate variability (RMSSD)",
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
st.markdown(
    "This dashboard is for people exploring their wearable sleep data. A **change** means a "
    "measurement crossed a rule when compared with the person's previous nights. "
    "Flags are statistical or rule-based observations, not diagnoses or medical advice."
)

with st.sidebar:
    source = st.selectbox("Data source", [OURA_SOURCE, APPLE_SOURCE])
    is_apple = source == APPLE_SOURCE
    apple_dir = apple_output_directory() if is_apple else None
    uploaded = None
    if not is_apple:
        uploaded = st.file_uploader("Upload dashboard CSV", type=["csv"])
        if uploaded is not None:
            st.caption("Using uploaded CSV.")
        elif GENERATED_PATH.exists():
            st.caption("Using the generated baseline and change-detection table.")
        else:
            st.caption("Generated results not found; using documented sample data.")

    st.divider()
    st.header("Insights")
    use_ai = st.toggle(
        "Use AI-generated insights",
        value=False,
        key=f"ai_insights_{source}",
        help=(
            "Sends the selected night's summary measurements to the configured AI provider."
            if is_apple else
            "Uses AI when an API key is available. "
            "If AI is unavailable or fails, the dashboard automatically uses the rule-based summary."
        ),
    )

try:
    data, source_name = load_dashboard_data(uploaded, data_source=source, apple_output_dir=apple_dir)
except (OSError, pd.errors.ParserError, ValueError) as exc:
    if is_apple:
        st.error("Could not read the Apple Health change-detection CSV. Check the setup in the README.")
    else:
        st.error(f"Could not load dashboard data: {exc}")
    st.stop()

if data.empty:
    if is_apple:
        st.info("Apple Health results are unavailable. Generate the Apple results and set APPLE_HEALTH_OUTPUT_DIR.")
    else:
        st.warning("No dashboard records are available in this file.")
    st.stop()

with st.sidebar:
    st.caption(f"Source: `{source_name}` · {len(data):,} metric records")
    participants = sorted(data["participant_id"].unique())
    participant = st.selectbox("Participant", participants)

participant_data = data.loc[data["participant_id"].eq(participant)]
apple_nights = None
if apple_dir is not None:
    try:
        apple_features = load_apple_nightly_features(apple_dir)
        apple_nights = apple_features.loc[apple_features["participant_id"].eq(participant)]
    except (OSError, ValueError):
        st.warning("Apple nightly features could not be loaded. The change-detection results are still available.")
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

range_start = pd.Timestamp(min(dates)).date()
range_end = pd.Timestamp(max(dates)).date()
with st.sidebar:
    report_range = st.date_input(
        "PDF report date range",
        value=(range_start, range_end),
        min_value=range_start,
        max_value=range_end,
        help="Choose the nights included in the shareable participant summary.",
    )
if isinstance(report_range, tuple) and len(report_range) == 2:
    report_start, report_end = report_range
else:
    report_start = report_end = report_range

with st.sidebar:
    night = st.selectbox("Night", dates, format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"))

selected = participant_data.loc[participant_data["night_date"].eq(night)].copy()
if selected["metric"].duplicated(keep=False).any():
    st.error("This participant-night has duplicate metric rows. Keep one row per metric in the input.")
    st.stop()

try:
    pdf_bytes = build_participant_report(
        participant_data,
        participant,
        baseline_window,
        report_start,
        report_end,
        METRIC_LABELS,
    )
    with st.sidebar:
        st.download_button(
            "Download participant report (PDF)",
            data=pdf_bytes,
            file_name=report_filename(participant, report_start, report_end),
            mime="application/pdf",
            use_container_width=True,
        )
except ValueError as exc:
    with st.sidebar:
        st.warning(f"PDF report unavailable: {exc}")

st.subheader(f"Participant {participant} · Night of {pd.Timestamp(night).strftime('%B %d, %Y')}")
st.caption("Units appear beside each observed value. Differences are observed minus baseline.")

changes = selected.loc[selected["status"].eq("change")]
unavailable = selected[
    selected["status"].isin({"missing", "insufficient_history"})
    | selected["observed_value"].isna()
]
overview_tab, metrics_tab, timeline_tab, temperature_tab, comparison_tab = st.tabs([
    "Overview", "Metric details", "Night timeline", "Temperature trend", "Compare metrics",
])

with overview_tab:
    facts = night_facts(selected, METRIC_LABELS)
    night_summary, insight_source = get_summary(facts, use_ai=use_ai)
    with st.container(border=True):
        st.markdown("### What stood out")
        st.markdown(f"**{night_summary.headline}**")
        for sentence in night_summary.details:
            st.markdown(f"- {sentence}")
        st.caption(f"{insight_source} summary. {DISCLAIMER}")

    summary_cols = st.columns(3)
    summary_cols[0].metric("Metrics shown", len(selected))
    summary_cols[1].metric("Changes detected", len(changes))
    summary_cols[2].metric("Data gaps / short history", len(unavailable))
    st.caption("Explore the other tabs for individual metrics, the night timeline, temperature, and comparisons.")

with metrics_tab:
    if is_apple:
        sdnn = None if apple_nights is None else apple_nights.loc[
            apple_nights["sleep_date"].eq(night), "hrv_sdnn_ms"
        ].mean()
        st.metric("Heart rate variability (SDNN)", "Unavailable" if pd.isna(sdnn) else format_value(sdnn, "ms"))
    if selected.empty:
        st.warning("No metrics are available for the selected night.")
    else:
        preferred_order = list(METRIC_LABELS)
        order = {name: index for index, name in enumerate(preferred_order)}
        selected["_order"] = selected["metric"].map(order).fillna(len(order))
        selected = selected.sort_values(["_order", "metric"]).drop(columns="_order")

        changed_metrics = selected.loc[selected["status"].eq("change")]
        normal_metrics = selected.loc[selected["status"].eq("normal")]
        unavailable_metrics = selected.loc[
            selected["status"].isin({"missing", "insufficient_history"})
            | selected["observed_value"].isna()
        ]

        def show_metric_group(title: str, explanation: str, frame: pd.DataFrame, empty_message: str) -> None:
            st.markdown(f"### {title}")
            st.caption(explanation)
            if frame.empty:
                st.info(empty_message)
                return
            for start in range(0, len(frame), 4):
                cards = st.columns(4)
                for card, (_, row) in zip(cards, frame.iloc[start:start + 4].iterrows()):
                    with card:
                        show_metric_card(row)

        show_metric_group(
            "Changes to review",
            "These measurements crossed the selected change-detection rule and are the highest-priority items to review.",
            changed_metrics,
            "No measurements crossed the change-detection rule for this night.",
        )
        show_metric_group(
            "Within the usual range",
            "These measurements were available and did not cross the selected change-detection rule.",
            normal_metrics,
            "No measurements are available in this group.",
        )
        show_metric_group(
            "Missing data or not enough history",
            "These measurements could not be compared reliably because a value is missing or there are too few prior nights.",
            unavailable_metrics,
            "All displayed measurements have enough data for a comparison.",
        )

        if not changed_metrics.empty:
            with st.expander("Technical change-detection details"):
                st.caption("This table is for reviewing the values and rules behind the change flags.")
                details = changed_metrics.copy()
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

with timeline_tab:
    show_night_timeline(participant, night, changes, aligned_dir=apple_dir)

with temperature_tab:
    if is_apple:
        st.markdown("### Sleeping wrist temperature")
        temperatures = None if apple_nights is None else apple_nights.dropna(subset=["wrist_temperature_c"])
        if temperatures is None or temperatures.empty:
            st.info("Wrist temperature is unavailable in these Apple Health results.")
        else:
            plot = temperatures.assign(sleep_date=pd.to_datetime(temperatures["sleep_date"])).rename(
                columns={"sleep_date": "Night", "wrist_temperature_c": "Wrist temperature (°C)"}
            )
            st.line_chart(plot.sort_values("Night"), x="Night", y="Wrist temperature (°C)")
            if not temperatures["sleep_date"].eq(night).any():
                st.info("Wrist temperature is unavailable for the selected night.")
    else:
        show_temperature_trend(
            participant_data, participant, baseline_window, night,
            use_participant_baselines=uploaded is None and GENERATED_PATH.exists(),
        )

with comparison_tab:
    show_metric_comparison(participant_data, night, available_only=is_apple)
