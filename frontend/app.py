"""Streamlit sleep dashboard backed by a documented long-format CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from charts import show_night_timeline


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SAMPLE_PATH = APP_DIR / "sample_sleep_dashboard.csv"
GENERATED_PATH = PROJECT_DIR / "outputs" / "change_detection_results.csv"
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

show_night_timeline(participant, night)

changes = selected.loc[selected["status"].eq("change")]
unavailable = selected[
    selected["status"].isin({"missing", "insufficient_history"})
    | selected["observed_value"].isna()
]
summary_cols = st.columns(3)
summary_cols[0].metric("Metrics shown", len(selected))
summary_cols[1].metric("Changes detected", len(changes))
summary_cols[2].metric("Data gaps / short history", len(unavailable))

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
