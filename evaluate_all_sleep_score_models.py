# Evaluate every sleep-score model in one run, on the same nights, with the same rules
# Task: estimate the same night's Oura sleep score
# Rules shared by every model:
#   - Leave one participant out, so every test person is unseen
#   - Median imputer fitted inside each training fold (training participants only)
#   - Predefined low-score flag: predicted score below 70; truly low = actual score below 70
# Two setups:
#   Original (five allowed inputs as nightly values)
#       baseline_mean, ridge, gradient_boosting, random_forest
#   Personalized (each person's own PRIOR nights only: last 14 observed nights, at least 3)
#       A_ridge_with_deviations      allowed inputs + each input's change from the person's usual (no past scores)
#       B_personal_usual_score       predicts the person's own recent average score
#       B_ridge / B_gradient_boosting / B_random_forest
#                                    start from the usual score, then model the change from it
#                                    using each input's change from the person's usual
# Every model is scored on the same nights: nights with at least 3 prior nights of history
# Run python3 evaluate_all_sleep_score_models.py to write the all_model_results folder

from __future__ import annotations  # Postponed type hints

import argparse  # Command-line parsing
from pathlib import Path  # Filesystem paths

import numpy as np  # Numeric arrays
import pandas as pd  # Tabular data
from sklearn.dummy import DummyRegressor  # Mean-score baseline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # Tree models
from sklearn.impute import SimpleImputer  # Median imputation
from sklearn.inspection import permutation_importance  # Input usefulness
from sklearn.linear_model import Ridge  # Linear model
from sklearn.model_selection import LeaveOneGroupOut  # Participant hold-out
from sklearn.pipeline import make_pipeline  # Imputer and model fitted together
from sklearn.preprocessing import StandardScaler  # Scale inputs for ridge


PROJECT_DIR = Path(__file__).resolve().parent  # Project root folder
DEFAULT_INPUT = PROJECT_DIR / "model_ready_sleep_score.csv"  # Model-ready table
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "all_model_results"  # Results folder

TARGET = "target_sleep_score"  # Oura overall sleep score
GROUP = "participant_id"  # Split unit
RAW = [  # Allowed inputs from build_sleep_score_dataset.py
    "average_heart_rate_bpm",
    "lowest_heart_rate_bpm",
    "hrv_rmssd_ms",
    "average_breath_rate_bpm",
    "temperature_delta_c",
]
DEV = [f"{c}_vs_usual" for c in RAW]  # Change from the person's usual for each input
WINDOW = 14  # Prior nights used for the personal baseline
MIN_PRIOR = 3  # Minimum prior nights before a baseline exists
LOW_SCORE = 70  # Predefined low-score cutoff
MIN_IMPORTANCE_NIGHTS = 20  # Skip tiny held-out participants for importance
RANDOM_STATE = 0  # Reproducible results


# ---------- Model factories ----------

def baseline():
    return make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="mean"))


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def gradient_boosting():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                      min_samples_leaf=40, random_state=RANDOM_STATE),
    )


def random_forest():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.6,
                              n_jobs=-1, random_state=RANDOM_STATE),
    )


# name: (setup, input columns, model factory, start from usual score?)
# A factory of None means the prediction is the usual score itself (no model fitted)
MODELS = {
    "baseline_mean":           ("original", RAW, baseline, False),
    "ridge":                   ("original", RAW, ridge, False),
    "gradient_boosting":       ("original", RAW, gradient_boosting, False),
    "random_forest":           ("original", RAW, random_forest, False),
    "A_ridge_with_deviations": ("personalized, allowed inputs only", RAW + DEV, ridge, False),
    "B_personal_usual_score":  ("personalized, uses past scores", [], None, True),
    "B_ridge":                 ("personalized, uses past scores", DEV, ridge, True),
    "B_gradient_boosting":     ("personalized, uses past scores", DEV, gradient_boosting, True),
    "B_random_forest":         ("personalized, uses past scores", DEV, random_forest, True),
}


# ---------- Personal baselines ----------

def add_personal_baselines(table: pd.DataFrame) -> pd.DataFrame:
    # Baselines use only nights BEFORE the current night, so nothing from tonight or later leaks in
    table = table.sort_values([GROUP, "sleep_date"]).reset_index(drop=True)
    by_person = table.groupby(GROUP)

    def prior_mean(column: str) -> pd.Series:
        return by_person[column].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_PRIOR).mean())

    for raw, dev in zip(RAW, DEV):
        table[dev] = table[raw] - prior_mean(raw)  # Tonight minus the person's usual
    table["usual_score"] = prior_mean(TARGET)  # Person's recent average score
    return table


# ---------- Metrics ----------

def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae": error.abs().mean(),  # Average miss in points
        "rmse": np.sqrt((error ** 2).mean()),  # Average miss with large misses weighted more
        "r2": 1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),  # Share of variation explained
        "within_5_points": (error.abs() <= 5).mean(),  # Share of nights within 5 points
        "within_10_points": (error.abs() <= 10).mean(),  # Share of nights within 10 points
    }


def flag_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    truly_low = actual < LOW_SCORE
    flagged = predicted < LOW_SCORE
    tp = int((truly_low & flagged).sum())
    fp = int((~truly_low & flagged).sum())
    fn = int((truly_low & ~flagged).sum())
    tn = int((~truly_low & ~flagged).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan  # Share of low nights caught
    precision = tp / (tp + fp) if tp + fp else np.nan  # Share of alerts that were right
    return {
        "nights": len(actual),
        "truly_low_nights": int(truly_low.sum()),
        "flagged_nights": int(flagged.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_low_nights": fn,
        "correct_no_alert": tn,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,  # Balance of both
        "false_alert_rate": fp / (fp + tn) if fp + tn else np.nan,  # Share of normal nights flagged
        "false_alerts_per_30_nights": 30 * fp / len(actual),
    }


# ---------- Evaluation ----------

def run_evaluation(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, groups, usual = table[TARGET], table[GROUP], table["usual_score"]
    predictions = table[[GROUP, "sleep_date", TARGET, "usual_score"]].copy()
    importance_rows = []

    for name, (_, columns, factory, from_usual) in MODELS.items():
        print(f"Evaluating {name} ...")
        predictions[name] = np.nan
        for train_idx, test_idx in LeaveOneGroupOut().split(table, y, groups):
            start_test = usual.iloc[test_idx].to_numpy() if from_usual else 0.0

            if factory is None:  # Usual score on its own
                predictions.iloc[test_idx, predictions.columns.get_loc(name)] = start_test
                continue

            # Personalized B models learn the change from the usual score
            target_train = y.iloc[train_idx] - usual.iloc[train_idx] if from_usual else y.iloc[train_idx]
            model = factory().fit(table[columns].iloc[train_idx], target_train)
            predicted = model.predict(table[columns].iloc[test_idx]) + start_test
            predictions.iloc[test_idx, predictions.columns.get_loc(name)] = predicted

            if factory is not baseline and len(test_idx) >= MIN_IMPORTANCE_NIGHTS:
                result = permutation_importance(
                    model, table[columns].iloc[test_idx], y.iloc[test_idx] - start_test,
                    scoring="neg_mean_absolute_error", n_repeats=10, random_state=RANDOM_STATE,
                )
                for feature, gain in zip(columns, result.importances_mean):
                    importance_rows.append({"model": name, GROUP: groups.iloc[test_idx[0]], "feature": feature,
                                            "mae_increase": gain, "nights": len(test_idx)})

    return predictions, pd.DataFrame(importance_rows)


def summarize(predictions: pd.DataFrame, importance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    actual = predictions[TARGET]
    setup = {name: spec[0] for name, spec in MODELS.items()}

    overall = pd.DataFrame({name: regression_metrics(actual, predictions[name]) for name in MODELS}).T
    overall.insert(0, "setup", pd.Series(setup))
    overall.index.name = "model"

    flags = pd.DataFrame({name: flag_metrics(actual, predictions[name]) for name in MODELS}).T
    flags.insert(0, "setup", pd.Series(setup))
    flags.index.name = "model"

    per_person = []
    for participant, rows in predictions.groupby(GROUP):
        row = {GROUP: participant, "nights": len(rows), "mean_score": rows[TARGET].mean()}
        for name in MODELS:
            row[f"{name}_mae"] = (rows[name] - rows[TARGET]).abs().mean()
        per_person.append(row)
    per_person = pd.DataFrame(per_person).set_index(GROUP).sort_values("B_random_forest_mae")

    # Weight each held-out person by their number of nights
    weights = importance.groupby(["model", "feature"])["nights"].transform(lambda s: s / s.sum())
    inputs = (
        importance.assign(weighted=importance["mae_increase"] * weights)
        .groupby(["model", "feature"])
        .agg(mae_increase=("weighted", "sum"),
             participants_where_helpful=("mae_increase", lambda s: int((s > 0).sum())),
             participants_tested=("mae_increase", "size"))
        .reset_index()
        .sort_values(["model", "mae_increase"], ascending=[True, False])
        .set_index(["model", "feature"])
    )

    return {"overall_metrics": overall, "low_score_flag": flags,
            "input_importance": inputs, "per_participant_mae": per_person}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all same-night Oura sleep score models on unseen participants")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    full = add_personal_baselines(pd.read_csv(args.input).dropna(subset=[TARGET]))
    table = full.dropna(subset=["usual_score"]).reset_index(drop=True)  # Same nights for every model
    predictions, importance = run_evaluation(table)
    results = summarize(predictions, importance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    for name, frame in results.items():
        frame.to_csv(args.output_dir / f"{name}.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"\n{len(table)} of {len(full)} nights scored from {table[GROUP].nunique()} participants "
          f"(first {MIN_PRIOR} nights per person have no personal baseline)")
    print(f"Low-score nights (actual score < {LOW_SCORE}): {(table[TARGET] < LOW_SCORE).mean():.1%}")
    for name in ["overall_metrics", "low_score_flag", "input_importance"]:
        print(f"\n=== {name.replace('_', ' ').title()} ===")
        print(results[name].to_string())
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
