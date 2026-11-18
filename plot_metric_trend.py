# Plot one metric across nights for one participant with the rolling baseline and threshold band
# Nights flagged by change detection (status == "change") are highlighted as red diamonds
# The band reproduces the flag rule in build_change_detection_table.py:
#   1_day window (absolute_difference):       baseline +/- threshold_value
#   7/14/30-day windows (absolute_z_score):   baseline +/- threshold_value * baseline_std
# Highlights come straight from the status column, so they always match change detection
# Run build_change_detection_table.py first to create change_detection_results.csv
# Example: python3 plot_metric_trend.py --participant par_16 --metric sleeping_hr_avg_bpm --window 14_days

from __future__ import annotations  # Postponed type hints

import argparse  # Command-line parsing
from pathlib import Path  # Filesystem paths

import matplotlib  # Plotting backend setup
matplotlib.use("Agg")  # Save images without a display
import matplotlib.dates as mdates  # Date axis formatting
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numeric arrays
import pandas as pd  # Tabular data


PROJECT_DIR = Path(__file__).resolve().parent  # Project root folder
DEFAULT_INPUT = PROJECT_DIR / "change_detection_results.csv"  # Change-detection table
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "metric_trend_charts"  # Where charts are saved

WINDOW_LABELS = {  # Display names for baseline windows
    "1_day": "previous night",
    "7_days": "previous 7 nights",
    "14_days": "previous 14 nights",
    "30_days": "previous 30 nights",
}
COMPARED = {"normal", "change", "missing"}  # Statuses where a baseline comparison was made

SERIES_COLOR = "#2a78d6"  # Observed values and band
BASELINE_COLOR = "#8a8984"  # Rolling baseline
FLAG_COLOR = "#d03b3b"  # Flagged nights
SHORT_HISTORY_COLOR = "#b5b4ad"  # Nights without enough history


def threshold_band(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # Lower and upper flag limits per night; NaN where no comparison applies
    threshold = pd.to_numeric(rows["threshold_value"], errors="coerce")
    baseline = pd.to_numeric(rows["baseline_value"], errors="coerce")
    std = pd.to_numeric(rows["baseline_std"], errors="coerce")
    kind = rows["threshold_type"].fillna("").astype(str)

    half_width = pd.Series(np.nan, index=rows.index)
    absolute = kind.eq("absolute_difference")
    half_width[absolute] = threshold[absolute]
    z_based = kind.eq("absolute_z_score") & std.gt(0)  # Zero spread gives z = 0, never flagged, so no band
    half_width[z_based] = threshold[z_based] * std[z_based]

    half_width[~rows["status"].isin(COMPARED)] = np.nan
    return baseline - half_width, baseline + half_width


def build_trend(table: pd.DataFrame, participant: str, metric: str, window: str) -> pd.DataFrame:
    # One row per night for the chosen participant, metric and baseline window
    rows = table[(table["participant_id"] == participant) & (table["metric"] == metric)
                 & (table["baseline_window"] == window)].copy()
    if rows.empty:
        raise SystemExit(f"No rows for participant={participant}, metric={metric}, window={window}.")
    if rows["night_date"].duplicated().any():
        raise SystemExit("Each night must appear once per participant, metric and window.")

    rows["night"] = pd.to_datetime(rows["night_date"])
    rows = rows.sort_values("night").reset_index(drop=True)
    rows["band_lower"], rows["band_upper"] = threshold_band(rows)
    rows["baseline_shown"] = rows["baseline_value"].where(rows["status"].isin(COMPARED))
    rows["flagged"] = rows["status"].eq("change")
    return rows


def plot_trend(trend: pd.DataFrame, window: str, output_path: Path, selected_night: str | None) -> None:
    label = trend["metric_label"].iloc[0]
    unit = trend["unit"].iloc[0]
    participant = trend["participant_id"].iloc[0]
    x = trend["night"]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.fill_between(x, trend["band_lower"], trend["band_upper"], color=SERIES_COLOR, alpha=0.14,
                    linewidth=0, label="Threshold band")
    ax.plot(x, trend["baseline_shown"], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
            label=f"Baseline ({WINDOW_LABELS.get(window, window)})")
    ax.plot(x, trend["observed_value"], color=SERIES_COLOR, linewidth=1.6, label="Observed")

    normal = trend["status"].eq("normal")
    short = trend["status"].eq("insufficient_history")
    flagged = trend["flagged"]
    ax.scatter(x[normal], trend.loc[normal, "observed_value"], s=18, color=SERIES_COLOR,
               edgecolor="white", linewidth=0.8, zorder=3, label="Within range")
    ax.scatter(x[short], trend.loc[short, "observed_value"], s=18, color=SHORT_HISTORY_COLOR,
               edgecolor="white", linewidth=0.8, zorder=3, label="Insufficient history")
    ax.scatter(x[flagged], trend.loc[flagged, "observed_value"], s=70, marker="D", color=FLAG_COLOR,
               edgecolor="white", linewidth=1.0, zorder=4, label=f"Flagged change ({int(flagged.sum())})")

    if selected_night:
        night = pd.Timestamp(selected_night)
        ax.axvline(night, color="#52514e", linestyle=":", linewidth=1.5)
        ax.annotate("Selected night", xy=(night, 1), xycoords=("data", "axes fraction"),
                    xytext=(5, -14), textcoords="offset points", fontsize=9, color="#52514e", fontweight="bold")

    ax.set_title(f"{participant} · {label} · baseline: {WINDOW_LABELS.get(window, window)}", loc="left", fontsize=13)
    ax.set_ylabel(f"{label} ({unit})")
    ax.set_xlabel("Night")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=6, frameon=False, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a metric across nights with its baseline and threshold band")
    parser.add_argument("--participant", required=True, help="e.g. par_1")
    parser.add_argument("--metric", default="total_sleep_min", help="e.g. sleeping_hr_avg_bpm")
    parser.add_argument("--window", default="7_days", choices=sorted(WINDOW_LABELS), help="Baseline window")
    parser.add_argument("--night", default=None, help="Optional night to mark, YYYY-MM-DD")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="PNG path (default: metric_trend_charts/...)")
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    if args.metric not in set(table["metric"]):
        raise SystemExit(f"Unknown metric. Choose from: {', '.join(sorted(table['metric'].unique()))}")

    trend = build_trend(table, args.participant, args.metric, args.window)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.participant}_{args.metric}_{args.window}.png"
    plot_trend(trend, args.window, output, args.night)

    flagged = trend[trend["flagged"]]
    compared = int(trend["status"].isin(["normal", "change"]).sum())
    print(f"{args.participant} · {trend['metric_label'].iloc[0]} · baseline: {WINDOW_LABELS[args.window]}")
    print(f"Nights plotted: {len(trend)} · compared: {compared} · flagged: {len(flagged)}")
    if not flagged.empty:
        print(flagged[["night_date", "observed_value", "baseline_value", "band_lower", "band_upper", "z_score"]]
              .round(2).to_string(index=False))
    print(f"Saved chart to {output}")


if __name__ == "__main__":
    main()
