from pathlib import Path

import pandas as pd


def to_datetime_ms(series):
    return pd.to_datetime(series, unit="ms", utc=True)


base_dir = Path("data/ifh_affect")
participant_dirs = sorted(base_dir.glob("par_*"))

all_sleep = []
all_aligned = []

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
            merged["participant_id"] = participant_dir.name
            all_aligned.append(merged)

all_sleep_df = pd.concat(all_sleep, ignore_index=True) if all_sleep else pd.DataFrame()
aligned_df = pd.concat(all_aligned, ignore_index=True) if all_aligned else pd.DataFrame()

print("Participants found:", len(participant_dirs))
print("Total sleep rows:", len(all_sleep_df))
print("Aligned signal rows:", len(aligned_df))
print("Date range:", all_sleep_df["date"].min(), "to", all_sleep_df["date"].max())

print("\nSample aligned data:")
print(
    aligned_df[
        ["participant_id", "date", "timestamp_dt", "heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class"]
    ].head()
)

print("\nMissing values:")
print(aligned_df[["heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class"]].isnull().sum())

print("\nSummary columns available:")
print(all_sleep_df.columns[:10].tolist())
print("...")
print(all_sleep_df[["date", "total", "deep", "rem", "awake", "efficiency", "hr_average", "rmssd", "temperature_delta"]].head())

