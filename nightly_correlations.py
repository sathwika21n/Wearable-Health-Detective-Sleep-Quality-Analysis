# Note correlates nightly sleep heart rate and temperature features
# Missing values are not imputed so each pair uses only nights where both
# values exist and n_nights reports how many nights each result used
# Use within_r and within_p for inference because they centre each person
# Pooled p values treat nights as independent which they are not
# Run python3 nightly_correlations.py to write the correlation_results folder

from __future__ import annotations  # Postponed type hints

import argparse  # Command-line parsing
from itertools import product  # Feature pair combinations
from pathlib import Path  # Filesystem paths

import numpy as np  # Numeric arrays
import pandas as pd  # Tabular data
from scipy import stats  # Statistical tests


PROJECT_DIR = Path(__file__).resolve().parent  # Project root folder

SLEEP_FEATURES_FROM_SUMMARY = {  # Oura sleep summary fields
    "total_sleep_time_min": ("total", 1 / 60),  # Seconds to minutes
    "deep_sleep_min": ("deep", 1 / 60),  # Seconds to minutes
    "rem_sleep_min": ("rem", 1 / 60),  # Seconds to minutes
    "light_sleep_min": ("light", 1 / 60),  # Seconds to minutes
    "awake_min": ("awake", 1 / 60),  # Seconds to minutes
    "time_in_bed_min": ("duration", 1 / 60),  # Seconds to minutes
    "sleep_efficiency_percent": ("efficiency", 1),  # Already a percent
    "onset_latency_min": ("onset_latency", 1 / 60),  # Seconds to minutes
    "restless_percent": ("restless", 1),  # Already a percent
}  # End summary fields
SLEEP_FEATURES_FROM_HYPNOGRAM = [  # Counts from stage labels
    "observed_awakenings_count",  # Sleep-to-awake transitions
    "observed_stage_changes_count",  # Any stage transition
]  # End hypnogram counts
HEART_FEATURES = {  # Heart feature mapping
    "hr_average_bpm": "hr_average",  # Average heart rate
    "hr_lowest_bpm": "hr_lowest",  # Lowest heart rate
    "hrv_rmssd_ms": "rmssd",  # Heart rate variability
}  # End heart features
TEMPERATURE_FEATURES = {"temperature_delta_c": "temperature_delta"}  # Temperature deviation

MODALITIES = {  # Features grouped by modality
    "sleep": list(SLEEP_FEATURES_FROM_SUMMARY) + SLEEP_FEATURES_FROM_HYPNOGRAM,  # All sleep features
    "heart": list(HEART_FEATURES),  # All heart features
    "temperature": list(TEMPERATURE_FEATURES),  # All temperature features
}  # End modalities
CROSS_MODALITY_PAIRS = [("sleep", "heart"), ("sleep", "temperature"), ("heart", "temperature")]  # Modality pairs compared

VALID_STAGES = {"awake", "light", "deep", "REM"}  # Known stage labels
EPOCH = pd.Timedelta(minutes=5)  # One label length

MIN_NIGHTS_PER_PAIR = 10  # Minimum pair nights
MIN_NIGHTS_PER_PARTICIPANT = 5  # Minimum nights per person


def load_sleep_summaries(data_dir: Path) -> pd.DataFrame:  # Load sleep summaries
    frames = []  # Collected participant tables
    for path in sorted(data_dir.glob("par_*/oura/sleep.csv")):  # Each participant file
        frame = pd.read_csv(path)  # Read sleep CSV
        frame["participant_id"] = path.parents[1].name  # Tag participant
        frames.append(frame)  # Store table
    if not frames:  # No files found
        raise FileNotFoundError(f"No par_*/oura/sleep.csv files found in {data_dir}")  # Stop with error
    nights = pd.concat(frames, ignore_index=True)  # Combine all participants
    start = pd.to_datetime(nights["bedtime_start_timestamp"], unit="ms", utc=True)  # Parse start time
    nights["sleep_start_utc"] = start.dt.strftime("%Y-%m-%d %H:%M:%S")  # Readable start key
    nights["_start"] = start  # Window start
    nights["_end"] = start + pd.to_timedelta(nights["duration"], unit="s")  # Window end
    return nights  # Return combined table


def flag_overlapping_records(nights: pd.DataFrame) -> pd.Series:  # Find overlapping windows
    ordered = nights.sort_values(["participant_id", "_start"])  # Sort by time
    previous_end = ordered.groupby("participant_id")["_end"].shift()  # Prior record end
    next_start = ordered.groupby("participant_id")["_start"].shift(-1)  # Next record start
    overlaps = (ordered["_start"] < previous_end) | (ordered["_end"] > next_start)  # Overlap either side
    return overlaps.reindex(nights.index).fillna(False)  # Restore original order


def observed_transitions(night: pd.DataFrame) -> tuple[object, object]:  # Count night transitions
    labels = (  # Clean label rows
        night.dropna(subset=["hypnogram_class"])  # Drop missing labels
        .drop_duplicates(subset="datetime_utc")  # Drop duplicate times
        .sort_values("datetime_utc")  # Order by time
    )  # End label cleaning
    stages = labels["hypnogram_class"]  # Stage per epoch
    valid = stages.isin(VALID_STAGES)  # Known stage mask
    if not valid.any():  # No usable labels
        return pd.NA, pd.NA  # Unknown counts
    previous = stages.shift()  # Previous epoch stage
    adjacent = labels["datetime_utc"].diff().eq(EPOCH)  # Exactly five minutes
    changes = adjacent & valid & previous.isin(VALID_STAGES) & stages.ne(previous)  # Valid stage change
    awakenings = changes & stages.eq("awake")  # Changes into awake
    return int(awakenings.sum()), int(changes.sum())  # Return both counts


def load_hypnogram_counts(aligned_dir: Path) -> pd.DataFrame:  # Recompute transition counts
    rows = []  # Collected night rows
    for path in sorted(aligned_dir.glob("par_*_aligned.csv")):  # Each aligned file
        participant_id = path.name.removesuffix("_aligned.csv")  # Participant from filename
        aligned = pd.read_csv(  # Read needed columns
            path, usecols=["sleep_start_utc", "datetime_utc", "hypnogram_class"]  # Columns to load
        )  # End read
        aligned["datetime_utc"] = pd.to_datetime(aligned["datetime_utc"])  # Parse epoch times
        for start, night in aligned.groupby("sleep_start_utc", sort=False):  # Each sleep window
            awakenings, changes = observed_transitions(night)  # Count transitions
            rows.append({  # Add night row
                "participant_id": participant_id,  # Participant key
                "sleep_start_utc": start,  # Window key
                "observed_awakenings_count": awakenings,  # Awakening count
                "observed_stage_changes_count": changes,  # Stage change count
            })  # End night row
    return pd.DataFrame(rows)  # Return counts table


def build_nightly_table(data_dir: Path, aligned_dir: Path) -> tuple[pd.DataFrame, dict]:  # Build nightly features
    raw = load_sleep_summaries(data_dir)  # Load all records
    overlapping = flag_overlapping_records(raw)  # Mark overlaps

    table = raw.loc[~overlapping, ["participant_id", "date", "sleep_start_utc"]].copy()  # Keep non-overlapping keys
    table = table.rename(columns={"date": "sleep_date"})  # Rename date column
    kept = raw.loc[~overlapping]  # Non-overlapping records
    for column, (source, scale) in SLEEP_FEATURES_FROM_SUMMARY.items():  # Each sleep field
        table[column] = pd.to_numeric(kept[source], errors="coerce") * scale  # Convert and scale
    for column, source in {**HEART_FEATURES, **TEMPERATURE_FEATURES}.items():  # Heart and temperature
        table[column] = pd.to_numeric(kept[source], errors="coerce")  # Convert to numeric

    counts = load_hypnogram_counts(aligned_dir)  # Get transition counts
    table = table.merge(counts, on=["participant_id", "sleep_start_utc"], how="left")  # Attach counts
    for column in SLEEP_FEATURES_FROM_HYPNOGRAM:  # Each count column
        table[column] = pd.to_numeric(table[column], errors="coerce")  # Ensure numeric

    for column in ["hr_average_bpm", "hr_lowest_bpm", "hrv_rmssd_ms"]:  # Heart columns
        table.loc[table[column] <= 0, column] = np.nan  # Zeros become missing

    log = {  # Exclusion summary
        "sleep_records_in_source": len(raw),  # Records loaded
        "overlapping_records_excluded": int(overlapping.sum()),  # Records removed
        "nights_in_table": len(table),  # Nights kept
        "participants": table["participant_id"].nunique(),  # Participant count
        "nights_without_hypnogram_labels": int(  # Nights lacking labels
            table["observed_stage_changes_count"].isna().sum()  # Count missing counts
        ),  # End count
    }  # End log
    table = table.sort_values(["participant_id", "sleep_start_utc"]).reset_index(drop=True)  # Sort nights
    return table, log  # Return table and log


def repeated_measures_correlation(  # Within-person correlation
    pair: pd.DataFrame, x: str, y: str, min_nights: int  # Inputs
) -> dict[str, object]:  # Returns result dict
    counts = pair.groupby("participant_id")[x].transform("size")  # Nights per person
    pair = pair[counts >= min_nights]  # Drop sparse participants
    n_nights = len(pair)  # Nights used
    n_participants = pair["participant_id"].nunique()  # Participants used
    result = {"within_n_nights": n_nights, "within_n_participants": n_participants,  # Sample sizes
              "within_r": np.nan, "within_p": np.nan}  # Default empty result
    df = n_nights - n_participants - 1  # Degrees of freedom
    if n_participants == 0 or df < 2:  # Too little data
        return result  # Return empty result
    centred_x = pair[x] - pair.groupby("participant_id")[x].transform("mean")  # Centre x per person
    centred_y = pair[y] - pair.groupby("participant_id")[y].transform("mean")  # Centre y per person
    denominator = np.sqrt((centred_x**2).sum() * (centred_y**2).sum())  # Correlation denominator
    if denominator == 0:  # No variation
        return result  # Return empty result
    r = float((centred_x * centred_y).sum() / denominator)  # Within-person r
    t = r * np.sqrt(df / max(1 - r**2, 1e-12))  # t statistic
    result["within_r"] = r  # Store r
    result["within_p"] = float(2 * stats.t.sf(abs(t), df))  # Two-sided p-value
    return result  # Return result


def correlate_pair(nights: pd.DataFrame, x: str, y: str) -> dict[str, object]:  # Correlate one pair
    pair = nights[["participant_id", x, y]].dropna()  # Pairwise deletion
    row: dict[str, object] = {  # Result row
        "feature_x": x,  # First feature
        "feature_y": y,  # Second feature
        "n_nights": len(pair),  # Nights used
        "n_participants": pair["participant_id"].nunique(),  # Participants used
        "nights_dropped_missing": len(nights) - len(pair),  # Nights dropped
    }  # End row
    if len(pair) >= MIN_NIGHTS_PER_PAIR and pair[x].nunique() > 1 and pair[y].nunique() > 1:  # Enough varied data
        spearman = stats.spearmanr(pair[x], pair[y])  # Rank correlation
        pearson = stats.pearsonr(pair[x], pair[y])  # Linear correlation
        row.update(pooled_spearman_rho=float(spearman.statistic),  # Store rho
                   pooled_spearman_p=float(spearman.pvalue),  # Store p-value
                   pooled_pearson_r=float(pearson.statistic))  # Store r
    else:  # Not enough data
        row.update(pooled_spearman_rho=np.nan, pooled_spearman_p=np.nan,  # Empty pooled values
                   pooled_pearson_r=np.nan)  # Empty Pearson value
    row.update(repeated_measures_correlation(pair, x, y, MIN_NIGHTS_PER_PARTICIPANT))  # Add within-person result
    return row  # Return row


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:  # FDR adjustment
    valid = p_values.dropna()  # Non-missing p-values
    adjusted = pd.Series(np.nan, index=p_values.index)  # Empty output
    if not valid.empty:  # Any p-values
        adjusted[valid.index] = stats.false_discovery_control(valid.to_numpy())  # Adjust p-values
    return adjusted  # Return q-values


def correlate_features(nights: pd.DataFrame) -> pd.DataFrame:  # All cross-modality pairs
    rows = []  # Collected results
    for group_x, group_y in CROSS_MODALITY_PAIRS:  # Each modality pair
        for x, y in product(MODALITIES[group_x], MODALITIES[group_y]):  # Each feature pair
            row = correlate_pair(nights, x, y)  # Compute correlation
            row["pair_type"] = f"{group_x} x {group_y}"  # Label pair type
            rows.append(row)  # Store result
    results = pd.DataFrame(rows)  # Results table
    results["within_q_fdr"] = benjamini_hochberg(results["within_p"])  # Add q-values
    columns = [  # Output column order
        "pair_type", "feature_x", "feature_y",  # Pair identifiers
        "n_nights", "n_participants", "nights_dropped_missing",  # Sample sizes
        "pooled_spearman_rho", "pooled_pearson_r", "pooled_spearman_p",  # Pooled results
        "within_r", "within_p", "within_q_fdr",  # Within-person results
        "within_n_nights", "within_n_participants",  # Within sample sizes
    ]  # End column order
    return results[columns]  # Return ordered table


def missingness_summary(nights: pd.DataFrame) -> pd.DataFrame:  # Missing values per feature
    features = [f for group in MODALITIES.values() for f in group]  # All features
    return pd.DataFrame({  # Summary table
        "feature": features,  # Feature name
        "modality": [m for m, group in MODALITIES.items() for _ in group],  # Feature modality
        "nights_present": [int(nights[f].notna().sum()) for f in features],  # Present counts
        "nights_missing": [int(nights[f].isna().sum()) for f in features],  # Missing counts
    })  # End summary


def main() -> None:  # Script entry point
    parser = argparse.ArgumentParser(description="Correlate nightly sleep, heart-rate, and temperature features.")  # Argument parser
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data" / "ifh_affect")  # Raw data folder
    parser.add_argument("--aligned-dir", type=Path, default=PROJECT_DIR / "aligned_data")  # Aligned data folder
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "correlation_results")  # Output folder
    args = parser.parse_args()  # Read arguments

    nights, log = build_nightly_table(args.data_dir, args.aligned_dir)  # Build nightly table
    results = correlate_features(nights)  # Compute correlations
    missing = missingness_summary(nights)  # Summarise missingness

    args.output_dir.mkdir(parents=True, exist_ok=True)  # Create output folder
    nights.to_csv(args.output_dir / "nightly_correlation_features.csv", index=False)  # Save nightly table
    results = results.round({"pooled_spearman_rho": 4, "pooled_pearson_r": 4, "within_r": 4})  # Round coefficients
    results.to_csv(args.output_dir / "nightly_correlations.csv", index=False)  # Save correlations
    missing.to_csv(args.output_dir / "feature_missingness.csv", index=False)  # Save missingness

    for key, value in log.items():  # Each log entry
        print(f"{key}: {value}")  # Print entry
    print(f"\nWrote {len(results)} correlations to {args.output_dir}")  # Print summary


if __name__ == "__main__":  # Run as script
    main()  # Start program
