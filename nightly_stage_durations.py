"""Nightly sleep-stage minutes."""  # module purpose, see note
from pathlib import Path  # filesystem path handling

import numpy as np  # numeric arrays and math
import pandas as pd  # tabular data handling

EPOCH_SECONDS = 300  # one label = 5 minutes
EPOCH_MINUTES = EPOCH_SECONDS / 60  # same epoch in minutes

LABEL_MAP = {1: ("deep", "deep"), 2: ("light", "light"), 3: ("REM", "rem"), 4: ("awake", "awake")}  # level -> (file label, column)
STAGES = ["deep", "light", "rem", "awake"]  # output stage order

project_dir = Path(__file__).resolve().parent  # folder containing this script
base_dir = project_dir / "data" / "ifh_affect"  # participant data root

rows = []  # one dict per night
for participant_dir in sorted(base_dir.glob("par_*"), key=lambda p: int(p.name.split("_")[1])):  # participants in numeric order
    hyp_path = participant_dir / "oura" / "sleep_hypnogram.csv"  # 5-minute stage labels
    sleep_path = participant_dir / "oura" / "sleep.csv"  # nightly sleep records
    if not (hyp_path.exists() and sleep_path.exists()):  # skip incomplete participants
        continue  # skips par_21 here

    hyp = pd.read_csv(hyp_path)  # load stage labels
    pairs = set(map(tuple, hyp[["hypnogram_level", "hypnogram_class"]].drop_duplicates().values))  # observed level/label pairs
    expected = {(lvl, cls) for lvl, (cls, _) in LABEL_MAP.items()}  # documented level/label pairs
    if not pairs <= expected:  # any undocumented pairing?
        raise ValueError(f"{participant_dir.name}: unexpected labels {pairs - expected}")  # stop on bad mapping
    hyp["stage"] = hyp["hypnogram_level"].map({k: v[1] for k, v in LABEL_MAP.items()})  # level to stage name
    hyp = hyp.set_index("timestamp")["stage"]  # stage lookup by timestamp
    if hyp.index.duplicated().any():  # duplicate timestamps present?
        raise ValueError(f"{participant_dir.name}: duplicate hypnogram timestamps")  # stop on duplicates

    sleep = pd.read_csv(sleep_path)  # load nightly records
    assigned = pd.Series(0, index=hyp.index)  # times each label used

    for _, night in sleep.iterrows():  # loop over nights
        n_epochs = int(np.ceil(night["duration"] / EPOCH_SECONDS))  # epochs, rounded up
        grid = night["bedtime_start_timestamp"] + EPOCH_SECONDS * 1000 * np.arange(n_epochs)  # label times from start
        labels = hyp.reindex(grid)  # this night's stage labels
        assigned[labels.dropna().index] += 1  # mark labels as used
        counts = labels.value_counts()  # epochs per stage

        start = pd.to_datetime(night["bedtime_start_timestamp"], unit="ms", utc=True)  # night start in UTC
        row = {  # build output row
            "participant": participant_dir.name,  # participant folder name
            "date": night["date"],  # Oura night date
            "hypnogram_start_utc": start.strftime("%Y-%m-%d %H:%M:%S"),  # first epoch start time
            "hypnogram_end_utc": (start + pd.Timedelta(seconds=n_epochs * EPOCH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"),  # last epoch end time
        }  # end of row dict
        for stage in STAGES:  # each sleep stage
            row[f"{stage}_min"] = int(counts.get(stage, 0)) * EPOCH_MINUTES  # epochs times 5 minutes
        row["total_staged_min"] = sum(row[f"{s}_min"] for s in STAGES)  # all stages incl. awake
        row["asleep_min"] = row["total_staged_min"] - row["awake_min"]  # total minus awake time
        row["n_epochs"] = int(labels.notna().sum())  # labels found this night
        row["missing_epochs"] = int(labels.isna().sum())  # expected labels not found
        row["oura_duration_min"] = night["duration"] / 60  # Oura time in bed
        rows.append(row)  # store finished row

    unassigned = int((assigned == 0).sum())  # labels matching no night
    double = int((assigned > 1).sum())  # labels matching several nights
    if unassigned or double:  # any assignment problems?
        print(f"WARNING {participant_dir.name}: {unassigned} unassigned, {double} double-assigned labels")  # report assignment problems

table = pd.DataFrame(rows)  # rows into a table

assert np.allclose(table[[f"{s}_min" for s in STAGES]].sum(axis=1), table["total_staged_min"])  # stages sum to total
assert np.allclose(table["total_staged_min"], table["n_epochs"] * EPOCH_MINUTES)  # total matches label count

out = project_dir / "nightly_stage_durations.csv"  # output file path
table.to_csv(out, index=False)  # write nightly table
print(f"{len(table)} nights, {table['participant'].nunique()} participants -> {out.name}")  # report table size
print("missing epochs:", int(table["missing_epochs"].sum()))  # report missing labels
print(table.head().to_string())  # preview first rows
