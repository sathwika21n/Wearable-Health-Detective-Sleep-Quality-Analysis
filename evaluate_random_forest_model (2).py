# Estimate the same night's Oura sleep score with a random forest
# Uses the same rules as evaluate_sleep_score_model.py so the results can be compared directly
# Evaluation holds out one participant at a time so every test person is unseen
# Missing values are filled by a median imputer fitted on training participants only
# The baseline predicts the training participants' average score for every night
# The low-score flag is fixed before testing: flag a night when the predicted score is below 70
# A night counts as truly low when the actual Oura score is below 70 (Oura's pay attention range)
# Run python3 evaluate_random_forest_model.py to write the random_forest_results folder

from __future__ import annotations  # Postponed type hints

import argparse  # Command-line parsing
from pathlib import Path  # Filesystem paths

import numpy as np  # Numeric arrays
import pandas as pd  # Tabular data
from sklearn.dummy import DummyRegressor  # Mean-score baseline
from sklearn.ensemble import RandomForestRegressor  # Random forest model
from sklearn.impute import SimpleImputer  # Median imputation
from sklearn.inspection import permutation_importance  # Input usefulness
from sklearn.model_selection import LeaveOneGroupOut  # Participant hold-out
from sklearn.pipeline import make_pipeline  # Imputer and model fitted together


PROJECT_DIR = Path(__file__).resolve().parent  # Project root folder
DEFAULT_INPUT = PROJECT_DIR / "model_ready_sleep_score.csv"  # Model-ready table
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "random_forest_results"  # Results folder

TARGET = "target_sleep_score"  # Oura overall sleep score
GROUP = "participant_id"  # Split unit
FEATURES = [  # Allowed inputs from build_sleep_score_dataset.py
    "average_heart_rate_bpm",
    "lowest_heart_rate_bpm",
    "hrv_rmssd_ms",
    "average_breath_rate_bpm",
    "temperature_delta_c",
]
LOW_SCORE = 70  # Predefined low-score cutoff
MAIN_MODEL = "random_forest"  # Model evaluated for input importance
RANDOM_STATE = 0  # Reproducible results


def make_models() -> dict[str, object]:
    # Every model is a pipeline so the imputer only ever sees training rows
    return {
        "baseline_mean": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean")),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,  # Number of trees
                min_samples_leaf=20,  # Each leaf needs 20 nights, limits overfitting
                max_features=0.6,  # Each split considers 60% of inputs
                n_jobs=-1,  # Use all CPU cores
                random_state=RANDOM_STATE,
            ),
        ),
    }


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": error.abs().mean(),  # Average miss in points
        "rmse": np.sqrt((error ** 2).mean()),  # Average miss with large misses weighted more
        "r2": 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),  # Share of variation explained
    }


def flag_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    truly_low = actual < LOW_SCORE
    flagged = predicted < LOW_SCORE
    tp = int((truly_low & flagged).sum())
    fp = int((~truly_low & flagged).sum())
    fn = int((truly_low & ~flagged).sum())
    tn = int((~truly_low & ~flagged).sum())
    return {
        "nights": len(actual),
        "truly_low_nights": int(truly_low.sum()),
        "flagged_nights": int(flagged.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_low_nights": fn,
        "correct_no_alert": tn,
        "recall": tp / (tp + fn) if tp + fn else np.nan,  # Share of low nights caught
        "precision": tp / (tp + fp) if tp + fp else np.nan,  # Share of alerts that were right
        "false_alert_rate": fp / (fp + tn) if fp + tn else np.nan,  # Share of normal nights flagged
        "false_alerts_per_30_nights": 30 * fp / len(actual),
    }


def run_evaluation(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Leave one participant out and collect predictions and permutation importance
    x, y, groups = table[FEATURES], table[TARGET], table[GROUP]
    predictions = table[[GROUP, "sleep_date", TARGET]].copy()
    importance_rows = []

    for name, model in make_models().items():
        predictions[name] = np.nan
        for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            predictions.iloc[test_idx, predictions.columns.get_loc(name)] = model.predict(x.iloc[test_idx])

            if name == MAIN_MODEL and len(test_idx) >= 20:  # Skip tiny held-out participants
                result = permutation_importance(
                    model, x.iloc[test_idx], y.iloc[test_idx],
                    scoring="neg_mean_absolute_error", n_repeats=10, random_state=RANDOM_STATE,
                )
                for feature, gain in zip(FEATURES, result.importances_mean):
                    importance_rows.append({GROUP: groups.iloc[test_idx[0]], "feature": feature,
                                            "mae_increase": gain, "nights": len(test_idx)})

    return predictions, pd.DataFrame(importance_rows)


def summarize(predictions: pd.DataFrame, importance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_names = list(make_models())
    actual = predictions[TARGET]

    overall = pd.DataFrame({name: regression_metrics(actual, predictions[name]) for name in model_names}).T
    overall.index.name = "model"

    per_person = []
    for participant, rows in predictions.groupby(GROUP):
        row = {GROUP: participant, "nights": len(rows), "mean_score": rows[TARGET].mean()}
        for name in model_names:
            row[f"{name}_mae"] = (rows[name] - rows[TARGET]).abs().mean()
        per_person.append(row)
    per_person = pd.DataFrame(per_person).sort_values(f"{MAIN_MODEL}_mae")

    weights = importance.groupby("feature")["nights"].transform(lambda s: s / s.sum())  # Weight people by nights
    inputs = (
        importance.assign(weighted=importance["mae_increase"] * weights)
        .groupby("feature")
        .agg(mae_increase=("weighted", "sum"),
             participants_where_helpful=("mae_increase", lambda s: int((s > 0).sum())),
             participants_tested=("mae_increase", "size"))
        .sort_values("mae_increase", ascending=False)
    )

    flags = pd.DataFrame({name: flag_metrics(actual, predictions[name]) for name in model_names}).T
    flags.index.name = "model"

    return {"overall_metrics": overall, "per_participant_mae": per_person,
            "input_importance": inputs, "low_score_flag": flags}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a random forest sleep score estimate on unseen participants")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    table = pd.read_csv(args.input).dropna(subset=[TARGET]).reset_index(drop=True)
    predictions, importance = run_evaluation(table)
    results = summarize(predictions, importance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    for name, frame in results.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=name != "per_participant_mae")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"{len(table)} nights from {table[GROUP].nunique()} participants, each held out once")
    print(f"Low-score nights (actual score < {LOW_SCORE}): {(table[TARGET] < LOW_SCORE).mean():.1%}")
    for name, frame in results.items():
        print(f"\n=== {name.replace('_', ' ').title()} ===")
        print(frame.to_string())
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
