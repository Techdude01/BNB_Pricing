"""Train baseline + added models with Optuna and print a comparison table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import dump

if __package__ is None or __package__ == "":
    # Support direct execution: python pricing_lab/run_all.py
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from pricing_lab import config
from pricing_lab.data import TrainTestData, load_train_test
from pricing_lab.models.elastic_net import ElasticNetResult, tune_elastic_net
from pricing_lab.models.ensemble import (
    EnsembleResult,
    fit_equal_voting_ensemble,
    fit_stacking_ensemble,
    fit_weighted_voting_ensemble,
    select_ensemble_candidates,
)
from pricing_lab.models.knn import KnnResult, tune_knn
from pricing_lab.models.neural_network import NeuralNetworkResult, tune_neural_network
from pricing_lab.models.random_forest import RandomForestResult, tune_random_forest
from pricing_lab.models.svm import SvmResult, tune_svm
from pricing_lab.models.xgboost_model import XgboostResult, tune_xgboost

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

TRAINING_MODES: tuple[str, str, str] = ("lite", "sample", "full")
LITE_TRAIN_ROWS: int = 800
SAMPLE_TRAIN_FRACTION: float = 0.35
LITE_TRIALS_ELASTIC: int = 1
LITE_TRIALS_KNN: int = 1
LITE_TRIALS_XGB: int = 1
LITE_TRIALS_SVM: int = 1
LITE_TRIALS_NN: int = 1
LITE_TRIALS_RF: int = 1
LITE_TRIALS_ENSEMBLE_WEIGHTS: int = 1
SAMPLE_TRIALS_ELASTIC: int = 10
SAMPLE_TRIALS_KNN: int = 6
SAMPLE_TRIALS_XGB: int = 14
SAMPLE_TRIALS_SVM: int = 8
SAMPLE_TRIALS_NN: int = 8
SAMPLE_TRIALS_RF: int = 10
SAMPLE_TRIALS_ENSEMBLE_WEIGHTS: int = 15
ENSEMBLE_CV_WINDOW: float = 0.03
ARTIFACT_FILENAMES: dict[str, str] = {
    "ElasticNet": "elastic_net.joblib",
    "KNN": "knn.joblib",
    "XGBoost": "xgboost.joblib",
    "SVM": "svm.joblib",
    "NeuralNetwork": "neural_network.joblib",
    "RandomForest": "random_forest.joblib",
    "VotingEnsembleEqual": "voting_ensemble_equal.joblib",
    "VotingEnsembleWeighted": "voting_ensemble_weighted.joblib",
    "StackingEnsemble": "stacking_ensemble.joblib",
}


def _result_row(
    name: str,
    best_cv_rmse_log: float,
    test_mae: float,
    test_rmse: float,
    test_r2: float,
    best_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": name,
        "cv_rmse_log": round(best_cv_rmse_log, 6),
        "test_mae_dollars": round(test_mae, 4),
        "test_rmse_dollars": round(test_rmse, 4),
        "test_r2": round(test_r2, 6),
        "best_params_json": json.dumps(best_params, sort_keys=True),
    }


def _collect_rows(results: list[Any]) -> list[dict[str, Any]]:
    return [
        _result_row(
            result.name,
            result.best_cv_rmse_log,
            result.test_metrics["mae"],
            result.test_metrics["rmse"],
            result.test_metrics["r2"],
            result.best_params,
        )
        for result in results
    ]


def _save_model_artifacts(
    output_dir: str,
    results: list[Any],
) -> None:
    artifacts_dir: Path = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        artifact_path = artifacts_dir / ARTIFACT_FILENAMES[result.name]
        dump(result.pipeline, artifact_path)
        print(f"Wrote {artifact_path}", flush=True)


def _resolve_trials(
    arg_value: int | None,
    lite_default: int,
    sample_default: int,
    full_default: int,
    mode: str,
) -> int:
    if arg_value is not None:
        return arg_value
    if mode == "lite":
        return lite_default
    if mode == "sample":
        return sample_default
    return full_default


def _representative_train_sample(
    data: TrainTestData,
    target_rows: int = LITE_TRAIN_ROWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample train rows across price bands, boroughs, and room types."""
    sample_size = min(target_rows, len(data.X_train))
    if sample_size == len(data.X_train):
        return data.X_train.copy(), data.y_train.copy()

    strata_frame = data.X_train[["neighbourhood_group", "room_type"]].copy()
    strata_frame["price_band"] = pd.qcut(data.y_train, q=5, labels=False, duplicates="drop").astype(str)
    strata = (
        strata_frame["neighbourhood_group"].astype(str)
        + "__"
        + strata_frame["room_type"].astype(str)
        + "__p"
        + strata_frame["price_band"].astype(str)
    )
    row_frame = pd.DataFrame({"stratum": strata}, index=data.X_train.index)
    stratum_counts = row_frame["stratum"].value_counts()
    # Proportional allocation keeps lite mode small without losing key price/location/room segments.
    exact_allocations = (stratum_counts / len(row_frame)) * sample_size
    allocations = np.floor(exact_allocations).astype(int)
    allocations = allocations.clip(upper=stratum_counts)

    remaining_rows = sample_size - int(allocations.sum())
    if remaining_rows > 0:
        fractional_remainders = (exact_allocations - np.floor(exact_allocations)).sort_values(ascending=False)
        for stratum_name in fractional_remainders.index:
            if remaining_rows <= 0:
                break
            if allocations[stratum_name] >= stratum_counts[stratum_name]:
                continue
            allocations[stratum_name] += 1
            remaining_rows -= 1

    sampled_indices: list[int] = []
    for stratum_name, rows_to_sample in allocations.items():
        if rows_to_sample <= 0:
            continue
        stratum_indices = row_frame.index[row_frame["stratum"] == stratum_name]
        sampled = pd.Series(stratum_indices).sample(
            n=int(rows_to_sample),
            random_state=config.RANDOM_STATE,
        )
        sampled_indices.extend(sampled.tolist())

    if len(sampled_indices) < sample_size:
        remaining_index = data.X_train.index.difference(pd.Index(sampled_indices))
        extra_indices = pd.Series(remaining_index).sample(
            n=sample_size - len(sampled_indices),
            random_state=config.RANDOM_STATE,
        )
        sampled_indices.extend(extra_indices.tolist())

    sampled_features = data.X_train.loc[sampled_indices].sample(frac=1.0, random_state=config.RANDOM_STATE)
    sampled_target = data.y_train.loc[sampled_features.index]
    return sampled_features, sampled_target


def _build_training_data(mode: str, data: TrainTestData, lite_train_rows: int = LITE_TRAIN_ROWS) -> TrainTestData:
    if mode == "full":
        return data
    if mode == "lite":
        sample_features, sample_target = _representative_train_sample(data, target_rows=lite_train_rows)
        sampled_rows = len(sample_features)
        print(
            f"Lite mode: using {sampled_rows} / {len(data.X_train)} train rows "
            "with stratified price-band/borough/room-type sampling for Optuna + fit.",
            flush=True,
        )
        return TrainTestData(
            X_train=sample_features.reset_index(drop=True),
            X_test=data.X_test,
            y_train=sample_target.reset_index(drop=True),
            y_test=data.y_test,
        )
    sample_features = data.X_train.sample(frac=SAMPLE_TRAIN_FRACTION, random_state=config.RANDOM_STATE)
    sample_target = data.y_train.loc[sample_features.index]
    sampled_rows: int = len(sample_features)
    print(
        f"Sample mode: using {sampled_rows} / {len(data.X_train)} train rows "
        f"({SAMPLE_TRAIN_FRACTION:.0%}) for Optuna + fit.",
        flush=True,
    )
    return TrainTestData(
        X_train=sample_features.reset_index(drop=True),
        X_test=data.X_test,
        y_train=sample_target.reset_index(drop=True),
        y_test=data.y_test,
    )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Optuna tuning for ElasticNet, KNN, XGBoost, SVM, Neural Network, "
            "and Random Forest (log1p price target) plus a voting ensemble."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to AB_NYC_2019.csv (defaults to project root next to pricing_lab).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lite",
        choices=TRAINING_MODES,
        help="Training mode: lite (fast representative smoke), sample (fast iteration), or full (heavier search).",
    )
    parser.add_argument(
        "--lite-train-rows",
        type=int,
        default=LITE_TRAIN_ROWS,
        help=f"Number of representative train rows for lite mode (default: {LITE_TRAIN_ROWS}).",
    )
    parser.add_argument(
        "--n-trials-elastic",
        type=int,
        default=None,
        help="Optuna trials for ElasticNet (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-knn",
        type=int,
        default=None,
        help="Optuna trials for KNN (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-xgb",
        type=int,
        default=None,
        help="Optuna trials for XGBoost (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-svm",
        type=int,
        default=None,
        help="Optuna trials for SVM (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-nn",
        type=int,
        default=None,
        help="Optuna trials for Neural Network (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-rf",
        type=int,
        default=None,
        help="Optuna trials for Random Forest (overrides mode defaults).",
    )
    parser.add_argument(
        "--n-trials-ensemble-weights",
        type=int,
        default=None,
        help="Optuna trials for weighted voting ensemble (overrides mode defaults).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save the summary table (e.g. results/baseline_metrics.csv).",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=None,
        help="Optional directory to save fitted model artifacts (joblib files).",
    )
    args: argparse.Namespace = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Use one shared split so model metrics are directly comparable.
    full_data: TrainTestData = load_train_test(csv_path=args.csv)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Train rows: {len(full_data.X_train)}, test rows: {len(full_data.X_test)}", flush=True)
    if args.lite_train_rows <= 0:
        raise ValueError("--lite-train-rows must be positive.")
    train_data: TrainTestData = _build_training_data(args.mode, full_data, lite_train_rows=args.lite_train_rows)
    elastic_trials: int = _resolve_trials(
        args.n_trials_elastic,
        LITE_TRIALS_ELASTIC,
        SAMPLE_TRIALS_ELASTIC,
        config.N_TRIALS_ELASTICNET,
        args.mode,
    )
    knn_trials: int = _resolve_trials(
        args.n_trials_knn,
        LITE_TRIALS_KNN,
        SAMPLE_TRIALS_KNN,
        config.N_TRIALS_KNN,
        args.mode,
    )
    xgb_trials: int = _resolve_trials(
        args.n_trials_xgb,
        LITE_TRIALS_XGB,
        SAMPLE_TRIALS_XGB,
        config.N_TRIALS_XGBOOST,
        args.mode,
    )
    svm_trials: int = _resolve_trials(
        args.n_trials_svm,
        LITE_TRIALS_SVM,
        SAMPLE_TRIALS_SVM,
        config.N_TRIALS_SVM,
        args.mode,
    )
    nn_trials: int = _resolve_trials(
        args.n_trials_nn,
        LITE_TRIALS_NN,
        SAMPLE_TRIALS_NN,
        config.N_TRIALS_NEURAL_NETWORK,
        args.mode,
    )
    rf_trials: int = _resolve_trials(
        args.n_trials_rf,
        LITE_TRIALS_RF,
        SAMPLE_TRIALS_RF,
        config.N_TRIALS_RANDOM_FOREST,
        args.mode,
    )
    ensemble_weight_trials: int = _resolve_trials(
        args.n_trials_ensemble_weights,
        LITE_TRIALS_ENSEMBLE_WEIGHTS,
        SAMPLE_TRIALS_ENSEMBLE_WEIGHTS,
        35,
        args.mode,
    )
    print(
        "Trials => "
        f"ElasticNet: {elastic_trials}, KNN: {knn_trials}, XGBoost: {xgb_trials}, "
        f"SVM: {svm_trials}, NeuralNetwork: {nn_trials}, RandomForest: {rf_trials}, "
        f"EnsembleWeights: {ensemble_weight_trials}",
        flush=True,
    )
    print("Tuning ElasticNet...", flush=True)
    elastic_result: ElasticNetResult = tune_elastic_net(train_data, n_trials=elastic_trials)
    print("Tuning KNN...", flush=True)
    knn_result: KnnResult = tune_knn(train_data, n_trials=knn_trials)
    print("Tuning XGBoost...", flush=True)
    xgb_result: XgboostResult = tune_xgboost(train_data, n_trials=xgb_trials)
    print("Tuning SVM...", flush=True)
    svm_result: SvmResult = tune_svm(train_data, n_trials=svm_trials)
    print("Tuning Neural Network...", flush=True)
    neural_network_result: NeuralNetworkResult = tune_neural_network(train_data, n_trials=nn_trials)
    print("Tuning Random Forest...", flush=True)
    random_forest_result: RandomForestResult = tune_random_forest(train_data, n_trials=rf_trials)
    base_pipelines: dict[str, Any] = {
        "elastic": elastic_result.pipeline,
        "knn": knn_result.pipeline,
        "xgb": xgb_result.pipeline,
        "svm": svm_result.pipeline,
        "nn": neural_network_result.pipeline,
        "rf": random_forest_result.pipeline,
    }
    base_cv_scores: dict[str, float] = {
        "elastic": elastic_result.best_cv_rmse_log,
        "knn": knn_result.best_cv_rmse_log,
        "xgb": xgb_result.best_cv_rmse_log,
        "svm": svm_result.best_cv_rmse_log,
        "nn": neural_network_result.best_cv_rmse_log,
        "rf": random_forest_result.best_cv_rmse_log,
    }
    # Keep ensembles focused on base models close to the best CV score, with a two-model minimum.
    selected_pipelines = select_ensemble_candidates(
        base_pipelines,
        base_cv_scores,
        cv_window=ENSEMBLE_CV_WINDOW,
    )
    print(
        "Ensemble candidates (CV-window filter): " + ", ".join(selected_pipelines.keys()),
        flush=True,
    )
    print("Fitting equal-weight voting ensemble...", flush=True)
    voting_equal_result: EnsembleResult = fit_equal_voting_ensemble(train_data, selected_pipelines)
    print("Tuning weighted voting ensemble...", flush=True)
    voting_weighted_result: EnsembleResult = fit_weighted_voting_ensemble(
        train_data,
        selected_pipelines,
        n_trials=ensemble_weight_trials,
    )
    print("Fitting stacking ensemble...", flush=True)
    stacking_result: EnsembleResult = fit_stacking_ensemble(train_data, selected_pipelines)
    ordered_results: list[Any] = [
        elastic_result,
        knn_result,
        xgb_result,
        svm_result,
        neural_network_result,
        random_forest_result,
        voting_equal_result,
        voting_weighted_result,
        stacking_result,
    ]
    # Aggregate and display metrics before any optional persistence.
    rows: list[dict[str, Any]] = _collect_rows(ordered_results)
    table: pd.DataFrame = pd.DataFrame(rows)
    print(table.to_string(index=False), flush=True)
    if args.output_csv is not None:
        table.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}", flush=True)
    if args.model_output_dir is not None:
        _save_model_artifacts(args.model_output_dir, ordered_results)


if __name__ == "__main__":
    main()
