"""PDF summaries of participant sleep metrics and baseline comparisons."""

from __future__ import annotations

from datetime import date
from io import BytesIO
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


INK = colors.HexColor("#18324B")
PALE_BLUE = colors.HexColor("#EAF2F7")
GRAY = colors.HexColor("#657482")
GRID = "#D9E2E8"


def _fmt(value, unit: str = "") -> str:
    if pd.isna(value):
        return "Not available"
    value = float(value)
    shown = f"{value:.2f}" if unit in {"°C", "z"} else (f"{value:.1f}" if value % 1 else f"{value:.0f}")
    if unit == "count":
        unit = ""
    return f"{shown} {unit}".strip()


def _threshold(row: pd.Series) -> str:
    value = row.get("threshold_value")
    if pd.isna(value):
        return "Threshold not supplied"
    if row.get("threshold_type") == "absolute_difference":
        return f"absolute difference >= {_fmt(value, str(row.get('unit', '')))}"
    if row.get("threshold_type") == "absolute_z_score":
        return f"absolute z-score >= {_fmt(value, 'z')}"
    return "Threshold type not supplied"


def _metric_chart(frame: pd.DataFrame, metric: str, label: str) -> BytesIO:
    series = frame.loc[frame["metric"].eq(metric)].copy()
    series["night_date"] = pd.to_datetime(series["night_date"], errors="coerce")
    series = series.dropna(subset=["night_date"]).sort_values("night_date")
    series = series.drop_duplicates("night_date", keep="last")

    fig, ax = plt.subplots(figsize=(8.5, 2.45), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    observed = pd.to_numeric(series["observed_value"], errors="coerce")
    baseline = pd.to_numeric(series["baseline_value"], errors="coerce")
    changes = series["status"].eq("change") & observed.notna()
    ax.plot(series["night_date"], observed, color=GRID, linewidth=1.2, zorder=2)
    ax.scatter(series.loc[observed.notna(), "night_date"], observed.dropna(), s=20, color="#3B78A3", label="Observed", zorder=3)
    if baseline.notna().any():
        ax.plot(series["night_date"], baseline, color="#637A52", linewidth=1.4, linestyle="--", label="Baseline", zorder=2)
    if changes.any():
        ax.scatter(series.loc[changes, "night_date"], observed.loc[changes], s=34, color="#C94C4C", label="Detected change", zorder=4)
    ax.set_title(label, loc="left", color="#18324B", fontsize=10, fontweight="bold", pad=8)
    unit = str(series["unit"].dropna().iloc[0]) if series["unit"].notna().any() else ""
    ax.set_ylabel(unit, color="#657482", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.tick_params(axis="both", labelsize=7, colors="#657482")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", frameon=False, ncol=3, fontsize=7)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output


def build_participant_report(
    data: pd.DataFrame,
    participant_id: str,
    baseline_window: str,
    start_date: date | str,
    end_date: date | str,
    metric_labels: dict[str, str],
) -> bytes:
    """Build a shareable PDF from already-validated dashboard rows."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("The report start date must be on or before its end date.")

    report = data.loc[data["participant_id"].astype(str).eq(str(participant_id))].copy()
    for column in ("threshold_value", "threshold_type", "z_score", "baseline_std"):
        if column not in report:
            report[column] = pd.NA if column != "threshold_type" else ""
    report["night_date"] = pd.to_datetime(report["night_date"], errors="coerce")
    report = report.loc[report["night_date"].between(start, end)].copy()
    if report.empty:
        raise ValueError("There are no participant records in the selected report date range.")

    report["metric_label"] = report["metric"].map(metric_labels).fillna(report["metric"].str.replace("_", " ").str.title())
    report = report.sort_values(["night_date", "metric_label"])
    metrics = list(dict.fromkeys(report["metric"].tolist()))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=INK, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=10, leading=14, textColor=GRAY, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, leading=16, textColor=INK, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=INK))
    styles.add(ParagraphStyle(name="Note", parent=styles["BodyText"], fontSize=8, leading=11, textColor=GRAY))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.6 * inch,
        title=f"Sleep summary - participant {participant_id}",
        author="Sleep Detective Dashboard",
    )
    story = [
        Paragraph("Sleep Summary", styles["ReportTitle"]),
        Paragraph(
            f"Participant {escape(str(participant_id))} &nbsp;&bull;&nbsp; {start:%B %d, %Y} to {end:%B %d, %Y}<br/>"
            f"Baseline window: {escape(baseline_window.replace('_', ' '))} &nbsp;&bull;&nbsp; {report['night_date'].nunique()} nights",
            styles["SubTitle"],
        ),
    ]

    metric_nights = report.groupby("metric")["night_date"].nunique()
    first_values = report.sort_values("night_date").groupby("metric", as_index=False).first().set_index("metric")
    last_values = report.sort_values("night_date").groupby("metric", as_index=False).last().set_index("metric")
    trend_rows = [["Metric", "First observed", "Latest observed", "Difference"]]
    for metric in metrics:
        unit = str(report.loc[report["metric"].eq(metric), "unit"].iloc[0])
        first_value = first_values.loc[metric, "observed_value"]
        last_value = last_values.loc[metric, "observed_value"]
        delta = last_value - first_value if pd.notna(first_value) and pd.notna(last_value) else pd.NA
        trend_rows.append([
            metric_labels.get(metric, metric.replace("_", " ").title()),
            _fmt(first_value, unit),
            _fmt(last_value, unit),
            _fmt(delta, unit),
        ])
    story.extend([
        Paragraph("Key trends", styles["Section"]),
        Paragraph(
            "Observed values below show the first and latest available reading in this date range. Differences are descriptive; they do not establish cause or diagnosis. Personal baselines are not clinical reference ranges.<br/><b>Thresholds:</b> absolute-difference rules compare |observed minus baseline| with a supplied limit; z-score rules compare the absolute z-score with its limit. This report repeats the supplied flags and rules without recalculating them.",
            styles["Note"],
        ),
        Spacer(1, 5),
        _table(trend_rows, [2.35 * inch, 1.45 * inch, 1.45 * inch, 1.45 * inch]),
        Paragraph("Trend charts", styles["Section"]),
    ])

    chart_metrics = [metric for metric in metrics if metric_nights.get(metric, 0) > 1 and report.loc[report["metric"].eq(metric), "observed_value"].notna().any()]
    if chart_metrics:
        for metric in chart_metrics:
            label = metric_labels.get(metric, metric.replace("_", " ").title())
            chart = Image(_metric_chart(report, metric, label), width=7.18 * inch, height=2.07 * inch)
            story.append(KeepTogether([chart, Spacer(1, 5)]))
    else:
        story.append(Paragraph("Only one night of observed data is available; a trend chart is not shown.", styles["Note"]))

    story.append(Paragraph("Baseline comparisons", styles["Section"]))
    comparison_rows = [["Metric", "Mean observed", "Mean baseline", "Mean difference", "Flagged", "Applied rule"]]
    for metric in metrics:
        metric_rows = report.loc[report["metric"].eq(metric)]
        unit = str(metric_rows["unit"].iloc[0])
        observed = pd.to_numeric(metric_rows["observed_value"], errors="coerce")
        baseline = pd.to_numeric(metric_rows["baseline_value"], errors="coerce")
        difference = pd.to_numeric(metric_rows["difference"], errors="coerce")
        rule_rows = metric_rows.loc[metric_rows["threshold_value"].notna()]
        applied_rules = list(dict.fromkeys(_threshold(row) for _, row in rule_rows.iterrows()))
        comparison_rows.append([
            metric_labels.get(metric, metric.replace("_", " ").title()),
            _fmt(observed.mean(), unit),
            _fmt(baseline.mean(), unit),
            _fmt(difference.mean(), unit),
            str(metric_rows.loc[metric_rows["status"].eq("change"), "night_date"].nunique()),
            "; ".join(applied_rules) if applied_rules else "Not supplied",
        ])
    story.append(Paragraph("Mean values are calculated over available rows in the selected date range. Differences are observed minus baseline.", styles["Note"]))
    story.append(_table(comparison_rows, [1.45 * inch, 1.05 * inch, 1.05 * inch, 1.1 * inch, 0.65 * inch, 1.65 * inch], compact=True))

    changes = report.loc[report["status"].eq("change")].copy()
    story.append(Paragraph("Detected changes and baseline comparisons", styles["Section"]))
    story.append(Paragraph("Each row compares an observed value with the supplied baseline for that night and records the rule used by the source data.", styles["Note"]))
    if changes.empty:
        story.append(Paragraph("No measurements were flagged as changes in this date range.", styles["BodySmall"]))
    else:
        change_rows = [["Date", "Metric", "Observed", "Baseline", "Difference", "Rule"]]
        for _, row in changes.iterrows():
            change_rows.append([
                row["night_date"].strftime("%Y-%m-%d"),
                metric_labels.get(row["metric"], row["metric"].replace("_", " ").title()),
                _fmt(row["observed_value"], row["unit"]),
                _fmt(row["baseline_value"], row["unit"]),
                _fmt(row["difference"], row["unit"]),
                _threshold(row),
            ])
        story.append(_table(change_rows, [0.77 * inch, 1.28 * inch, 0.92 * inch, 0.92 * inch, 0.9 * inch, 1.95 * inch], compact=True))

    missing = report["status"].eq("missing") | report["observed_value"].isna()
    insufficient = report["status"].eq("insufficient_history")
    gaps = report.loc[missing | insufficient]
    story.append(Paragraph("Data gaps and history", styles["Section"]))
    if gaps.empty:
        story.append(Paragraph("No missing measurements or insufficient-history records were reported in this date range.", styles["BodySmall"]))
    else:
        gap_rows = [["Date", "Metric", "Availability", "History"]]
        for _, row in gaps.iterrows():
            status = "Insufficient history" if row["status"] == "insufficient_history" else "Measurement missing"
            used = int(row["baseline_nights"]) if pd.notna(row["baseline_nights"]) else 0
            required = int(row["minimum_nights"]) if pd.notna(row["minimum_nights"]) else None
            history = f"{used} prior nights" + (f"; {required} required" if required is not None else "")
            gap_rows.append([row["night_date"].strftime("%Y-%m-%d"), metric_labels.get(row["metric"], row["metric"].replace("_", " ").title()), status, history])
        story.append(_table(gap_rows, [1.0 * inch, 2.1 * inch, 1.65 * inch, 2.0 * inch], compact=True))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(PALE_BLUE)
        canvas.line(document.leftMargin, 0.48 * inch, letter[0] - document.rightMargin, 0.48 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(document.leftMargin, 0.32 * inch, "Sleep Detective - factual summary")
        canvas.drawRightString(letter[0] - document.rightMargin, 0.32 * inch, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def _table(rows: list[list[str]], widths: list[float], compact: bool = False) -> Table:
    cell_style = ParagraphStyle(
        name=f"Cell{'Small' if compact else 'Regular'}",
        fontName="Helvetica",
        fontSize=7 if compact else 8,
        leading=(7 if compact else 8) + 2,
        textColor=INK,
    )
    header_style = ParagraphStyle(
        name=f"Header{'Small' if compact else 'Regular'}",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )
    formatted = [
        [Paragraph(escape(str(value)), header_style if row_index == 0 else cell_style) for value in row]
        for row_index, row in enumerate(rows)
    ]
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    font_size = 7 if compact else 8
    padding = 4 if compact else 5
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
    ]))
    return table


def report_filename(participant_id: str, start_date: date | str, end_date: date | str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(participant_id)).strip("_") or "participant"
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    return f"sleep_summary_{safe_id}_{start}_{end}.pdf"
