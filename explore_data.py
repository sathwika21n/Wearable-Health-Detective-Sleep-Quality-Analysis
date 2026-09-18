import pandas as pd

# Load dataset
sleep = pd.read_csv("data/ifh_affect/par_1/oura/sleep.csv")

# Check dataset structure
print("Number of nights:", len(sleep))
print("Date range:", sleep["date"].min(), "to", sleep["date"].max())

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

print("\nSample data:")
print(sleep[columns].head())

print("\nMissing values:")
print(sleep[columns].isnull().sum())