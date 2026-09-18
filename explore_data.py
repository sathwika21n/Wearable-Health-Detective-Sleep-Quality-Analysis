import pandas as pd

sleep = pd.read_csv("data/ifh_affect/par_1/oura/sleep.csv")
hypnogram = pd.read_csv("data/ifh_affect/par_1/oura/sleep_hypnogram.csv")
heart_rate = pd.read_csv("data/ifh_affect/par_1/oura/heart_rate.csv")

# Basic dataset structure
print("Number of nights:", len(sleep))
print("Date range:", sleep["date"].min(), "to", sleep["date"].max())

print("\nSleep columns:")
print(sleep.columns.tolist())

print("\nHypnogram sample:")
print(hypnogram.head())

print("\nHeart rate sample:")
print(heart_rate.head())

# Merge sleep stage + HR/HRV by timestamp
merged = pd.merge(hypnogram, heart_rate, on="timestamp", how="inner")

print("\nMerged sample:")
print(
    merged[
        ["timestamp", "hypnogram_class", "heart_rate", "heart_rmssd"]
    ].head()
)

print("\nSleep stages:")
print(merged["hypnogram_class"].unique())

print("\nTemperature sample:")
print(sleep[["date", "temperature_delta"]].head())