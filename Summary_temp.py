from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "ifh_affect"

#Load every participants sleep csv into one table
frames = []
for path in DATA_DIR.glob("par_*/oura/sleep.csv"):
    df = pd.read_csv(path)
    df["participant_id"] = path.parents[1].name
    frames.append(df)
nights = pd.concat(frames, ignore_index=True)
nights = nights.dropna(subset=["temperature_delta"])
temp = nights["temperature_delta"]

#Overall summary
print("=== Overall ===")
print(f"Participants: {nights['participant_id'].nunique()}")
print(f"Nights: {len(temp)}")
print(f"Mean: {temp.mean():.2f}  Median: {temp.median():.2f}  SD: {temp.std():.2f}")
print(f"Range: {temp.min():.2f} to {temp.max():.2f}")
for limit in [0.3, 0.5, 1.0]:
    count = (temp.abs() >= limit).sum()
    print(f"Nights at least {limit} C from baseline: {count} ({count / len(temp):.1%})")

#Per participant
print("\n=== By participant ===")
by_person = nights.groupby("participant_id")["temperature_delta"].agg(
    ["count", "mean", "std", "min", "max"]
)
print(by_person.round(2))

#Most extreme nights
print("\n=== 10 most extreme nights ===")
extreme = nights.loc[temp.abs().sort_values(ascending=False).index[:10]]
print(extreme[["participant_id", "date", "temperature_delta", "hr_average", "rmssd"]])

#Warm nights vs normal nights
print("\n=== Warm nights (>= +0.5 C) vs normal nights (within 0.3 C) ===")
warm = nights[temp >= 0.5]
normal = nights[temp.abs() < 0.3]
for col in ["hr_average", "rmssd", "efficiency"]:
    print(f"{col}: warm {warm[col].mean():.1f} vs normal {normal[col].mean():.1f}")
