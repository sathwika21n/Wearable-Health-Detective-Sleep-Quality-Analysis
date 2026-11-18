"""History view: one participant's metric across a range of nights (#74), plus a
plain-language summary of their last 7 nights (#80).

The band and flag markers follow the same rule as Joshua's plot_metric_trend.py
(#56, #88) and change detection, so the chart and the flags always agree:
- previous-night window: baseline +/- the threshold
- 7, 14 and 30-night windows: baseline +/- threshold x the baseline's standard deviation

Everything except show_history() is plain pandas/Altair, so it can be tested
without Streamlit.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from insights import DISCLAIMER, HEADLINE_WINDOWS, NightSummary


COMPARED = {"normal", "change", "missing"}  # statuses where a baseline comparison applies
SERIES_COLOR = "#2a78d6"
BASELINE_COLOR = "#8a8984"
FLAG_COLOR = "#d03b3b"
SHORT_HISTORY_COLOR = "#b5b4ad"
STATUS_NAMES = {
    "normal": "Within range",
    "change": "Flagged change",
    "insufficient_history": "Not enough history",
    "missing": "Measurement missing",
}
SUMMARY_NIGHTS = 7
MAX_SUMMARY_METRICS = 3


def threshold_band(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Lower and upper flag limits per night; NaN where no comparison applies."""
    threshold = pd.to_numeric(rows.get("threshold_value"), errors="coerce")
    baseline = pd.to_numeric(rows.get("baseline_value"), errors="coerce")
    std = pd.to_numeric(rows.get("baseline_std"), errors="coerce")
    kind = rows.get("threshold_type", pd.Series("", index=rows.index)).fillna("").astype(str)

    half_width = pd.Series(np.nan, index=rows.index)
    absolute = kind.eq("absolute_difference")
    half_width[absolute] = threshold[absolute]
    z_based = kind.eq("absolute_z_score") & std.gt(0)  # zero spread is never flagged, so no band
    half_width[z_based] = threshold[z_based] * std[z_based]
    half_width[~rows["status"].isin(COMPARED)] = np.nan
    return baseline - half_width, baseline + half_width


def history_rows(
    data: pd.DataFrame,
    participant: str,
    window: str,
    metric: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per night for this participant, metric and window, oldest first."""
    rows = data.loc[
        data["participant_id"].astype(str).eq(str(participant))
        & data["baseline_window"].astype(str).eq(window)
        & data["metric"].astype(str).eq(metric)
    ].copy()
    rows["night"] = pd.to_datetime(rows["night_date"], errors="coerce")
    rows = rows.dropna(subset=["night"])
    if start is not None:
        rows = rows.loc[rows["night"].ge(pd.Timestamp(start))]
    if end is not None:
        rows = rows.loc[rows["night"].le(pd.Timestamp(end))]
    rows = rows.sort_values("night").drop_duplicates("night", keep="last").reset_index(drop=True)
    rows["observed_value"] = pd.to_numeric(rows["observed_value"], errors="coerce")
    rows["baseline_shown"] = pd.to_numeric(rows["baseline_value"], errors="coerce").where(rows["status"].isin(COMPARED))
    rows["band_lower"], rows["band_upper"] = threshold_band(rows)
    rows["flagged"] = rows["status"].eq("change")
    rows["status_name"] = rows["status"].map(STATUS_NAMES).fillna(rows["status"])
    return rows


def trend_chart(rows: pd.DataFrame, label: str, unit: str) -> alt.LayerChart:
    """Band, dashed baseline, observed line and one point per night colored by status."""
    axis_title = f"{label} ({unit})" if unit else label
    base = alt.Chart(rows).encode(x=alt.X("night:T", title="Night"))
    band = base.mark_area(color=SERIES_COLOR, opacity=0.14).encode(
        y=alt.Y("band_lower:Q", title=axis_title, scale=alt.Scale(zero=False)),
        y2="band_upper:Q",
    )
    baseline = base.mark_line(color=BASELINE_COLOR, strokeDash=[6, 4], strokeWidth=1.5).encode(y="baseline_shown:Q")
    observed = base.mark_line(color=SERIES_COLOR, strokeWidth=1.8).encode(y="observed_value:Q")
    tooltip = [
        alt.Tooltip("night:T", title="Night", format="%b %d, %Y"),
        alt.Tooltip("observed_value:Q", title=label, format=".1f"),
        alt.Tooltip("baseline_shown:Q", title="Usual", format=".1f"),
        alt.Tooltip("status_name:N", title="Status"),
    ]
    points = base.mark_point(filled=True, size=45).encode(
        y="observed_value:Q",
        color=alt.Color(
            "status_name:N",
            title="Night",
            scale=alt.Scale(
                domain=[STATUS_NAMES["normal"], STATUS_NAMES["change"], STATUS_NAMES["insufficient_history"]],
                range=[SERIES_COLOR, FLAG_COLOR, SHORT_HISTORY_COLOR],
            ),
            legend=alt.Legend(orient="bottom"),
        ),
        shape=alt.condition("datum.flagged", alt.value("diamond"), alt.value("circle")),
        size=alt.condition("datum.flagged", alt.value(110), alt.value(45)),
        tooltip=tooltip,
    )
    return alt.layer(band, baseline, observed, points).properties(height=300)


def recent_nights_summary(
    data: pd.DataFrame,
    participant: str,
    window: str,
    end_night: str | pd.Timestamp,
    labels: dict[str, str] | None = None,
    nights: int = SUMMARY_NIGHTS,
) -> NightSummary:
    """Plain-language summary of the last `nights` nights up to and including end_night."""
    labels = labels or {}
    rows = data.loc[
        data["participant_id"].astype(str).eq(str(participant))
        & data["baseline_window"].astype(str).eq(window)
    ].copy()
    rows["night"] = pd.to_datetime(rows["night_date"], errors="coerce")
    rows = rows.loc[rows["night"].le(pd.Timestamp(end_night))]
    recent = sorted(rows["night"].dropna().unique())[-nights:]
    rows = rows.loc[rows["night"].isin(recent)]
    count = len(recent)
    compared_with = HEADLINE_WINDOWS.get(window, "your recent nights")
    span = f"your last {count} night{'s' if count != 1 else ''}"
    if count == 0:
        return NightSummary(headline="No nights to summarize yet.", details=[])

    def label_for(metric: str) -> str:
        if metric in labels:
            return labels[metric]
        named = rows.loc[rows["metric"].eq(metric), "metric_label"] if "metric_label" in rows else pd.Series(dtype=str)
        return str(named.iloc[0]) if not named.empty and pd.notna(named.iloc[0]) else metric.replace("_", " ").capitalize()

    flagged = rows.loc[rows["status"].eq("change")].groupby("metric")["night"].nunique().sort_values(ascending=False)
    compared_nights = rows.loc[rows["status"].isin({"normal", "change"}), "night"].nunique()
    missing_nights = rows.loc[rows["status"].eq("missing"), "night"].nunique()
    statuses_by_night = rows.groupby("night")["status"].agg(set)
    short_nights = int(statuses_by_night.apply(lambda found: found == {"insufficient_history"}).sum())

    details = []
    for metric, times in flagged.head(MAX_SUMMARY_METRICS).items():
        details.append(f"{label_for(metric)} was flagged on {times} of the last {count} nights.")
    if len(flagged) > MAX_SUMMARY_METRICS:
        details.append(f"{len(flagged) - MAX_SUMMARY_METRICS} more measurements were flagged at least once.")
    if short_nights:
        details.append(f"{short_nights} of these nights didn't have enough earlier nights to compare yet.")
    if missing_nights:
        details.append(f"Some measurements were missing on {missing_nights} night{'s' if missing_nights != 1 else ''}.")

    if flagged.empty:
        headline = (f"Over {span}, nothing was flagged compared with {compared_with}."
                    if compared_nights else f"Over {span}, there wasn't enough history to compare yet.")
    else:
        repeated = [label_for(m) for m, times in flagged.items() if times > 1]
        if repeated:
            names = repeated[0] if len(repeated) == 1 else ", ".join(repeated[:-1]) + f" and {repeated[-1]}"
            headline = f"Over {span}, {names} stood out more than once compared with {compared_with}."
        else:
            headline = (f"Over {span}, {len(flagged)} measurement{'s' if len(flagged) != 1 else ''} "
                        f"stood out on one night each compared with {compared_with}.")
    return NightSummary(headline=headline, details=details)


def show_history(
    data: pd.DataFrame,
    participant: str,
    window: str,
    labels: dict[str, str] | None = None,
    selected_night: str | None = None,
) -> None:
    """Render the history view for one participant and baseline window."""
    import streamlit as st

    labels = labels or {}
    participant_rows = data.loc[
        data["participant_id"].astype(str).eq(str(participant))
        & data["baseline_window"].astype(str).eq(window)
    ]
    nights = pd.to_datetime(participant_rows["night_date"], errors="coerce").dropna()
    if nights.empty:
        st.info("No nights are available for this participant and baseline window.")
        return

    metrics = list(dict.fromkeys(participant_rows["metric"].astype(str)))
    first, last = nights.min().date(), nights.max().date()
    end_default = pd.Timestamp(selected_night).date() if selected_night else last
    start_default = max(first, (pd.Timestamp(end_default) - pd.Timedelta(days=29)).date())

    st.markdown("### History")
    controls = st.columns([2, 3])
    metric = controls[0].selectbox(
        "Measurement", metrics,
        index=metrics.index("total_sleep_min") if "total_sleep_min" in metrics else 0,
        format_func=lambda m: labels.get(m, m.replace("_", " ").capitalize()),
        key="history_metric",
    )
    picked = controls[1].date_input(
        "Nights", value=(start_default, end_default), min_value=first, max_value=last, key="history_range",
    )
    start, end = (picked if isinstance(picked, (list, tuple)) and len(picked) == 2 else (start_default, end_default))

    rows = history_rows(data, participant, window, metric, start, end)
    if rows.empty:
        st.info("No nights in this range.")
        return
    unit = str(rows["unit"].dropna().iloc[0]) if rows["unit"].notna().any() else ""
    label = labels.get(metric, metric.replace("_", " ").capitalize())
    flagged = int(rows["flagged"].sum())
    compared = int(rows["status"].isin({"normal", "change"}).sum())
    st.caption(
        f"{len(rows)} nights shown, {compared} compared with {HEADLINE_WINDOWS.get(window, 'the baseline')}, "
        f"{flagged} flagged. The shaded band is the range counted as usual; diamonds mark flagged nights."
    )
    st.altair_chart(trend_chart(rows, label, unit), use_container_width=True)

    summary = recent_nights_summary(data, participant, window, end, labels)
    with st.container(border=True):
        st.markdown("#### Your last 7 nights")
        st.markdown(f"**{summary.headline}**")
        for sentence in summary.details:
            st.markdown(f"- {sentence}")
        st.caption(f"Rule-based summary. {DISCLAIMER}")
