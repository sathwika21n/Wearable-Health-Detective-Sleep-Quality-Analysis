import pandas as pd


def to_datetime_ms(series):
    return pd.to_datetime(series, unit="ms", utc=True)


# Load nightly sleep summary
sleep = pd.read_csv("data/ifh_affect/par_1/oura/sleep.csv")
sleep["date"] = pd.to_datetime(sleep["date"])

# Convert the nightly sleep window timestamps into a comparable datetime range
sleep["bedtime_start_dt"] = to_datetime_ms(sleep["bedtime_start_timestamp"])
sleep["bedtime_end_dt"] = to_datetime_ms(sleep["bedtime_end_timestamp"])

# Load time-series signals
heart_rate = pd.read_csv("data/ifh_affect/par_1/oura/heart_rate.csv")
heart_rate["timestamp_dt"] = to_datetime_ms(heart_rate["timestamp"])
heart_rate = heart_rate.sort_values("timestamp_dt").drop_duplicates(subset="timestamp_dt")

sleep_hypnogram = pd.read_csv("data/ifh_affect/par_1/oura/sleep_hypnogram.csv")
sleep_hypnogram["timestamp_dt"] = to_datetime_ms(sleep_hypnogram["timestamp"])
sleep_hypnogram = sleep_hypnogram.sort_values("timestamp_dt").drop_duplicates(subset="timestamp_dt")

# Align each signal to the sleep window for that night
aligned_frames = []
for _, row in sleep.iterrows():
    start = row["bedtime_start_dt"]
    end = row["bedtime_end_dt"]
    night_date = row["date"]

    hr_window = heart_rate[(heart_rate["timestamp_dt"] >= start) & (heart_rate["timestamp_dt"] <= end)].copy()
    hr_window["date"] = night_date
    hr_window["sleep_start"] = start
    hr_window["sleep_end"] = end

    hyp_window = sleep_hypnogram[(sleep_hypnogram["timestamp_dt"] >= start) & (sleep_hypnogram["timestamp_dt"] <= end)].copy()
    hyp_window["date"] = night_date
    hyp_window["sleep_start"] = start
    hyp_window["sleep_end"] = end

    merged = hr_window.merge(
        hyp_window[["timestamp_dt", "hypnogram_level", "hypnogram_class", "date", "sleep_start", "sleep_end"]],
        on=["timestamp_dt", "date", "sleep_start", "sleep_end"],
        how="outer"
    ).sort_values("timestamp_dt")

    aligned_frames.append(merged)

aligned = pd.concat(aligned_frames, ignore_index=True) if aligned_frames else pd.DataFrame()

# Check dataset structure
print("Number of nights:", len(sleep))
print("Date range:", sleep["date"].min(), "to", sleep["date"].max())
print("Aligned signal rows:", len(aligned))

print("\nAligned sample:")
print(aligned[["date", "timestamp_dt", "heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class"]].head())

print("\nMissing values in aligned data:")
print(aligned[["heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class"]].isnull().sum())

print("\nColumns we will use:")
columns = [
    "date",
    "total",
    "deep",
    "rem",
    "awake",
    "efficiency",
    "hr_average",
    "rmssd",
    "temperature_delta"
]

print(columns)

print("\nSample nightly summary data:")
print(sleep[columns].head())

print("\nMissing values in summary data:")
print(sleep[columns].isnull().sum())