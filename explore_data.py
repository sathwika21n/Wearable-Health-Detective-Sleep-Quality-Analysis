from pathlib import Path

import pandas as pd


# Source sleep durations are in seconds; these values describe the whole night.
NIGHTLY_COLUMNS = {
    "temperature_delta": "temperature_delta",
    "duration": "night_time_in_bed_seconds",
    "total": "night_total_sleep_seconds",
    "deep": "night_deep_sleep_seconds",
    "rem": "night_rem_sleep_seconds",
    "light": "night_light_sleep_seconds",
    "awake": "night_awake_seconds",
    "efficiency": "night_sleep_efficiency_percent",
    "hr_average": "night_hr_average",
    "hr_lowest": "night_hr_lowest",
    "rmssd": "night_hrv_rmssd",
}

VALID_STAGES = ["awake", "light", "deep", "rem"]
SLEEP_STAGES = ["light", "deep", "rem"]


def awakening_metrics(hypnogram):
    if hypnogram.empty:
        return pd.NA, pd.NA

    hypnogram = hypnogram.sort_values("timestamp_dt").copy()

    current = hypnogram["hypnogram_class"]
    previous = current.shift()

    # Only treat rows as consecutive if exactly 5 minutes apart
    adjacent = (
        hypnogram["timestamp_dt"]
        .diff()
        .eq(pd.Timedelta(minutes=5))
    )

    # Awakening = sleep -> awake
    awakenings = (
        adjacent
        & current.eq("awake")
        & previous.isin(SLEEP_STAGES)
    )

    awakening_count = int(awakenings.sum())

    # Each awake epoch represents 5 minutes
    awake_epochs = int(current.eq("awake").sum())
    total_awake_minutes = awake_epochs * 5

    return awakening_count, total_awake_minutes

def observed_transitions(hypnogram):
    """Count only adjacent 5-minute labels; do not bridge missing samples."""
    stages = hypnogram["hypnogram_class"]
    previous = stages.shift()
    adjacent = hypnogram["timestamp_dt"].diff().eq(pd.Timedelta(minutes=5))
    valid = stages.isin(["awake", "light", "deep", "rem"])
    valid_previous = previous.isin(["awake", "light", "deep", "rem"])
    changes = adjacent & valid & valid_previous & stages.ne(previous)
    awakenings = changes & stages.eq("awake")
    # No labels means unknown, rather than zero awakenings.
    if not valid.any():
        return pd.NA, pd.NA
    return int(awakenings.sum()), int(changes.sum())


def export_participant(frame, path):
    """Keep timezone-aware alignment internally, but export readable UTC times."""
    frame = frame.drop(columns=["participant_id", "timestamp"], errors="ignore").copy()
    for column in ["timestamp_dt", "sleep_start", "sleep_end"]:
        frame[column] = frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame = frame.rename(columns={
        "timestamp_dt": "datetime_utc",
        "sleep_start": "sleep_start_utc",
        "sleep_end": "sleep_end_utc",
    })
    first = ["date", "datetime_utc", "sleep_start_utc", "sleep_end_utc"]
    frame = frame[first + [column for column in frame if column not in first]]
    frame.to_csv(path, index=False)


def to_datetime_ms(series):
    return pd.to_datetime(series, unit="ms", utc=True)


project_dir = Path(__file__).resolve().parent
base_dir = project_dir / "data" / "ifh_affect"
aligned_output_dir = project_dir / "aligned_data"
participant_dirs = sorted(base_dir.glob("par_*"))

all_sleep = []
all_aligned = []
nightly_awakening_rows = []

for participant_dir in participant_dirs:
    sleep_path = participant_dir / "oura" / "sleep.csv"
    heart_path = participant_dir / "oura" / "heart_rate.csv"
    hypnogram_path = participant_dir / "oura" / "sleep_hypnogram.csv"

    if not sleep_path.exists() or not heart_path.exists() or not hypnogram_path.exists():
        continue

    sleep = pd.read_csv(sleep_path)
    sleep["date"] = pd.to_datetime(sleep["date"])
    sleep["bedtime_start_dt"] = to_datetime_ms(sleep["bedtime_start_timestamp"])
    sleep["bedtime_end_dt"] = to_datetime_ms(sleep["bedtime_end_timestamp"])
    sleep["participant_id"] = participant_dir.name
    all_sleep.append(sleep)

    heart_rate = pd.read_csv(heart_path)
    heart_rate["timestamp_dt"] = to_datetime_ms(heart_rate["timestamp"])
    heart_rate = heart_rate.sort_values("timestamp_dt").drop_duplicates(subset="timestamp_dt")
    heart_rate["participant_id"] = participant_dir.name

    sleep_hypnogram = pd.read_csv(hypnogram_path)
    sleep_hypnogram["timestamp_dt"] = to_datetime_ms(sleep_hypnogram["timestamp"])
    sleep_hypnogram = sleep_hypnogram.sort_values("timestamp_dt").drop_duplicates(subset="timestamp_dt")
    sleep_hypnogram["participant_id"] = participant_dir.name

    participant_aligned = []

    for _, row in sleep.iterrows():
        start = row["bedtime_start_dt"]
        end = row["bedtime_end_dt"]
        night_date = row["date"]

        hr_window = heart_rate[
            (heart_rate["timestamp_dt"] >= start) & (heart_rate["timestamp_dt"] <= end)
        ].copy()
        hr_window["date"] = night_date
        hr_window["sleep_start"] = start
        hr_window["sleep_end"] = end

        hyp_window = sleep_hypnogram[
            (sleep_hypnogram["timestamp_dt"] >= start) & (sleep_hypnogram["timestamp_dt"] <= end)
        ].copy()
        hyp_window["date"] = night_date
        hyp_window["sleep_start"] = start
        hyp_window["sleep_end"] = end
       
        
        merged = hr_window.merge(
            hyp_window[["timestamp_dt", "hypnogram_level", "hypnogram_class", "date", "sleep_start", "sleep_end"]],
            on=["timestamp_dt", "date", "sleep_start", "sleep_end"],
            how="outer",
        ).sort_values("timestamp_dt")

        
        if not merged.empty:
            for source, target in NIGHTLY_COLUMNS.items():
                merged[target] = row[source]
            awakenings, changes = observed_transitions(hyp_window)
            awakening_count, awake_minutes = awakening_metrics(hyp_window)
            merged["night_observed_awakenings"] = awakenings
            merged["night_observed_stage_changes"] = changes
            merged["night_observed_awake_minutes"] = awake_minutes
            merged["participant_id"] = participant_dir.name

            all_aligned.append(merged)
            participant_aligned.append(merged)

            nightly_awakening_rows.append({
                "participant_id": participant_dir.name,
                "date": night_date,
                "awakening_count": awakening_count,
                "total_awake_minutes": awake_minutes
            })
            
    if participant_aligned:
        aligned_output_dir.mkdir(parents=True, exist_ok=True)
        participant_df = pd.concat(participant_aligned, ignore_index=True)
        export_participant(
            participant_df,
            aligned_output_dir / f"{participant_dir.name}_aligned.csv",
        )

all_sleep_df = pd.concat(all_sleep, ignore_index=True) if all_sleep else pd.DataFrame()
aligned_df = pd.concat(all_aligned, ignore_index=True) if all_aligned else pd.DataFrame()


print("Participants found:", len(participant_dirs))
print("Total sleep rows:", len(all_sleep_df))
print("Aligned signal rows:", len(aligned_df))
if not all_sleep_df.empty:
    print("Date range:", all_sleep_df["date"].min(), "to", all_sleep_df["date"].max())
else:
    print("Date range: unavailable (no participant sleep data found)")

print("Aligned participant files:", aligned_output_dir)

print("\nSample aligned data:")
sample_columns = [
    "participant_id", "date", "timestamp_dt", "heart_rate", "heart_rmssd",
    "hypnogram_level", "hypnogram_class",
    "temperature_delta",
]
if not aligned_df.empty:
    print(aligned_df[sample_columns].head())
else:
    print("No aligned signal rows found.")

print("\nMissing values:")
if not aligned_df.empty:
    print(aligned_df[["heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class", "temperature_delta"]].isnull().sum())
else:
    print("No aligned values to summarize.")

print("\nSummary columns available:")
print(all_sleep_df.columns[:10].tolist())
print("...")
if not all_sleep_df.empty:
    print(all_sleep_df[["date", "total", "deep", "rem", "awake", "efficiency", "hr_average", "rmssd", "temperature_delta"]].head())

# Save final nightly awakening table
nightly_awakenings_df = pd.DataFrame(nightly_awakening_rows)

if not nightly_awakenings_df.empty:
    nightly_awakenings_df["participant_number"] = (
        nightly_awakenings_df["participant_id"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    nightly_awakenings_df = nightly_awakenings_df.sort_values(
        ["participant_number", "date"]
    )

    # Remove helper column BEFORE saving
    nightly_awakenings_df = nightly_awakenings_df.drop(
        columns="participant_number"
    )

    # Save full nightly table
    nightly_awakenings_df.to_csv(
        project_dir / "participant_nightly_awakening/nightly_awakenings.csv",
        index=False
    )

    # Create participant summary
    participant_summary = (
        nightly_awakenings_df
        .groupby("participant_id")
        .agg(
            nights=("date", "count"),
            avg_awakenings=("awakening_count", "mean"),
            avg_awake_minutes=("total_awake_minutes", "mean"),
            max_awakenings=("awakening_count", "max"),
            max_awake_minutes=("total_awake_minutes", "max")
        )
        .reset_index()
    )

    # Add helper column for numeric sorting
    participant_summary["participant_number"] = (
        participant_summary["participant_id"]
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    participant_summary = participant_summary.sort_values(
        "participant_number"
    )

    # Remove helper column BEFORE printing
    participant_summary = participant_summary.drop(
        columns="participant_number"
    )

    print("\nParticipant awakening summary:")
    print(
        participant_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

else:
    print("\nNo nightly awakening data found.")