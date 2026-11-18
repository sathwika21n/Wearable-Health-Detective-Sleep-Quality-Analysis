"""Measure how much heart rate rises before each awakening, and roughly when it starts.

For every awakening (a sleep stage followed 5 minutes later by "awake", the same
rule as scripts/explore_data.py), this compares the heart rate in the 30 minutes
before waking with that night's sleeping heart rate (the average heart rate over
the night's light, deep and REM epochs).

Writes two CSVs:
    outputs/pre_waking_hr_rise.csv     one row per awakening
    outputs/pre_waking_hr_nightly.csv  one row per participant-night

Limitation: heart rate and sleep stages are sampled every 5 minutes, so the rise
start is only known to the nearest 5 minutes, and the last sample before waking is
up to 5 minutes before the actual wake time. Heart-rate readings of 0 are sensor
gaps and are treated as missing.

Run from the project root:
    python scripts/pre_waking_hr_rise.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ALIGNED_DIR = PROJECT_DIR / "aligned_data"
DEFAULT_AWAKENINGS_OUTPUT = PROJECT_DIR / "outputs" / "pre_waking_hr_rise.csv"
DEFAULT_NIGHTLY_OUTPUT = PROJECT_DIR / "outputs" / "pre_waking_hr_nightly.csv"

EPOCH = pd.Timedelta(minutes=5)
SLEEP_STAGES = {"light", "deep", "rem"}
OFFSETS_MIN = [30, 25, 20, 15, 10, 5]  # minutes before waking, oldest first
DEFAULT_RISE_THRESHOLD_BPM = 3.0  # how far above sleeping HR counts as "risen"
MIN_SAMPLES = 3  # samples needed in the 30-minute window to judge a rise


def prepare_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean one participant's aligned rows: lower-case stages, 0 bpm as missing."""
    signals = frame[["date", "datetime_utc", "hypnogram_class", "heart_rate"]].copy()
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    signals["datetime_utc"] = pd.to_datetime(signals["datetime_utc"], errors="coerce")
    signals["stage"] = signals["hypnogram_class"].astype("string").str.strip().str.lower()
    heart_rate = pd.to_numeric(signals["heart_rate"], errors="coerce")
    signals["heart_rate"] = heart_rate.where(heart_rate > 0)
    signals = signals.dropna(subset=["date", "datetime_utc"])
    return signals.sort_values("datetime_utc").reset_index(drop=True)


def night_sleeping_hr(night: pd.DataFrame) -> float | None:
    """Average heart rate over the night's sleep epochs (same as sleeping_hr_avg_bpm)."""
    values = night.loc[night["stage"].isin(SLEEP_STAGES), "heart_rate"].dropna()
    return float(values.mean()) if not values.empty else None


def awakening_times(night: pd.DataFrame) -> list[pd.Timestamp]:
    """Times of sleep -> awake transitions exactly one 5-minute epoch apart."""
    previous_stage = night["stage"].shift()
    gap = night["datetime_utc"].diff()
    mask = night["stage"].eq("awake") & previous_stage.isin(SLEEP_STAGES) & gap.eq(EPOCH)
    return list(night.loc[mask, "datetime_utc"])


def rise_metrics(window: dict[int, float | None], sleeping_hr: float | None,
                 threshold_bpm: float = DEFAULT_RISE_THRESHOLD_BPM) -> dict:
    """Rise size and start for one awakening.

    window maps minutes-before-waking (30 ... 5) to heart rate (None if missing).
    The rise start is the earliest sample of the unbroken run of samples, ending at
    5 minutes before waking, that are at least threshold_bpm above sleeping HR.
    """
    samples = sum(value is not None for value in window.values())
    last = window.get(5)
    first = window.get(30)
    result = {
        "samples_in_window": samples,
        "hr_5min_before": last,
        "rise_vs_sleeping_bpm": None,
        "rise_vs_30min_before_bpm": None,
        "rise_start_min_before": None,
        "status": "insufficient_data",
    }
    if sleeping_hr is None or last is None or samples < MIN_SAMPLES:
        return result
    result["rise_vs_sleeping_bpm"] = round(last - sleeping_hr, 1)
    if first is not None:
        result["rise_vs_30min_before_bpm"] = round(last - first, 1)
    if last - sleeping_hr < threshold_bpm:
        result["status"] = "no_rise"
        return result
    start = 5
    for minutes in sorted(window, reverse=False):  # 5, 10, ... 30: walk back from waking
        value = window[minutes]
        if value is None or value - sleeping_hr < threshold_bpm:
            break
        start = minutes
    result["rise_start_min_before"] = start
    result["status"] = "rise"
    return result


def awakening_rows(participant_id: str, signals: pd.DataFrame,
                   threshold_bpm: float = DEFAULT_RISE_THRESHOLD_BPM) -> list[dict]:
    rows = []
    by_time = signals.set_index("datetime_utc")["heart_rate"]
    by_time = by_time[~by_time.index.duplicated(keep="first")]
    for night_date, night in signals.groupby("date", sort=True):
        sleeping_hr = night_sleeping_hr(night)
        times = awakening_times(night)
        for order, wake_time in enumerate(times, start=1):
            window = {}
            for minutes in OFFSETS_MIN:
                value = by_time.get(wake_time - pd.Timedelta(minutes=minutes))
                window[minutes] = None if value is None or pd.isna(value) else float(value)
            rows.append({
                "participant_id": participant_id,
                "night_date": night_date,
                "awakening_time_utc": wake_time,
                "awakening_number": order,
                "is_final_wake": order == len(times),
                "night_sleeping_hr_bpm": None if sleeping_hr is None else round(sleeping_hr, 1),
                **{f"hr_minus_{m}m": window[m] for m in OFFSETS_MIN},
                **rise_metrics(window, sleeping_hr, threshold_bpm),
            })
    return rows


def nightly_summary(awakenings: pd.DataFrame) -> pd.DataFrame:
    """One row per participant-night summarizing its awakenings."""
    if awakenings.empty:
        return pd.DataFrame()
    rows = []
    for (participant_id, night_date), night in awakenings.groupby(["participant_id", "night_date"], sort=True):
        judged = night.loc[night["status"].isin({"rise", "no_rise"})]
        risen = night.loc[night["status"].eq("rise")]
        final = night.loc[night["is_final_wake"]]
        final_row = final.iloc[0] if not final.empty else None
        rows.append({
            "participant_id": participant_id,
            "night_date": night_date,
            "awakenings": len(night),
            "awakenings_judged": len(judged),
            "awakenings_with_rise": len(risen),
            "mean_rise_vs_sleeping_bpm": round(judged["rise_vs_sleeping_bpm"].mean(), 1) if not judged.empty else None,
            "max_rise_vs_sleeping_bpm": judged["rise_vs_sleeping_bpm"].max() if not judged.empty else None,
            "median_rise_start_min_before": risen["rise_start_min_before"].median() if not risen.empty else None,
            "final_wake_status": None if final_row is None else final_row["status"],
            "final_wake_rise_vs_sleeping_bpm": None if final_row is None else final_row["rise_vs_sleeping_bpm"],
            "final_wake_rise_start_min_before": None if final_row is None else final_row["rise_start_min_before"],
        })
    return pd.DataFrame(rows)


def build_tables(aligned_dir: Path, threshold_bpm: float = DEFAULT_RISE_THRESHOLD_BPM) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    files = sorted(Path(aligned_dir).glob("par_*_aligned.csv"))
    if not files:
        raise FileNotFoundError(f"No par_*_aligned.csv files found in {aligned_dir}")
    for path in files:
        participant_id = path.name.removesuffix("_aligned.csv")
        signals = prepare_signals(pd.read_csv(path))
        rows += awakening_rows(participant_id, signals, threshold_bpm)
    awakenings = pd.DataFrame(rows)
    if not awakenings.empty:
        awakenings["awakening_time_utc"] = pd.to_datetime(awakenings["awakening_time_utc"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return awakenings, nightly_summary(awakenings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_ALIGNED_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_RISE_THRESHOLD_BPM,
                        help="bpm above sleeping heart rate that counts as a rise (default 3)")
    parser.add_argument("--output", type=Path, default=DEFAULT_AWAKENINGS_OUTPUT)
    parser.add_argument("--nightly-output", type=Path, default=DEFAULT_NIGHTLY_OUTPUT)
    args = parser.parse_args()

    awakenings, nightly = build_tables(args.aligned_dir, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    awakenings.to_csv(args.output, index=False)
    nightly.to_csv(args.nightly_output, index=False)

    judged = awakenings["status"].isin({"rise", "no_rise"})
    print(f"Wrote {len(awakenings)} awakenings across {len(nightly)} nights to {args.output}")
    print(f"Wrote the nightly summary to {args.nightly_output}")
    print(awakenings["status"].value_counts().to_string())
    if judged.any():
        risen = awakenings.loc[awakenings["status"].eq("rise")]
        print(f"Heart rate had risen by waking in {len(risen)} of {int(judged.sum())} awakenings "
              f"(threshold {args.threshold:g} bpm above sleeping heart rate).")
        print(f"Median rise vs sleeping heart rate: {awakenings.loc[judged, 'rise_vs_sleeping_bpm'].median():.1f} bpm; "
              f"median rise start: {risen['rise_start_min_before'].median():.0f} min before waking.")


if __name__ == "__main__":
    main()
