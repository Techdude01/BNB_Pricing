"""Train baseline + added models with Optuna and print a comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

TRAINING_MODES: tuple[str, str] = ("sample", "full")
SAMPLE_TRAIN_FRACTION: float = 0.35
SAMPLE_TRIALS_ELASTIC: int = 10
SAMPLE_TRIALS_KNN: int = 6
SAMPLE_TRIALS_XGB: int = 14
SAMPLE_TRIALS_SVM: int = 8
SAMPLE_TRIALS_NN: int = 8
SAMPLE_TRIALS_RF: int = 10
SAMPLE_TRIALS_ENSEMBLE_WEIGHTS: int = 15
ENSEMBLE_CV_WINDOW: float = 0.03


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


def _collect_rows(
    elastic: ElasticNetResult,
    knn: KnnResult,
    xgb: XgboostResult,
    svm: SvmResult,
    neural_network: NeuralNetworkResult,
    random_forest: RandomForestResult,
    voting_equal: EnsembleResult,
    voting_weighted: EnsembleResult,
    stacking: EnsembleResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        _result_row(
            elastic.name,
            elastic.best_cv_rmse_log,
            elastic.test_metrics["mae"],
            elastic.test_metrics["rmse"],
            elastic.test_metrics["r2"],
            elastic.best_params,
        ),
        _result_row(
            knn.name,
            knn.best_cv_rmse_log,
            knn.test_metrics["mae"],
            knn.test_metrics["rmse"],
            knn.test_metrics["r2"],
            knn.best_params,
        ),
        _result_row(
            xgb.name,
            xgb.best_cv_rmse_log,
            xgb.test_metrics["mae"],
            xgb.test_metrics["rmse"],
            xgb.test_metrics["r2"],
            xgb.best_params,
        ),
        _result_row(
            svm.name,
            svm.best_cv_rmse_log,
            svm.test_metrics["mae"],
            svm.test_metrics["rmse"],
            svm.test_metrics["r2"],
            svm.best_params,
        ),
        _result_row(
            neural_network.name,
            neural_network.best_cv_rmse_log,
            neural_network.test_metrics["mae"],
            neural_network.test_metrics["rmse"],
            neural_network.test_metrics["r2"],
            neural_network.best_params,
        ),
        _result_row(
            random_forest.name,
            random_forest.best_cv_rmse_log,
            random_forest.test_metrics["mae"],
            random_forest.test_metrics["rmse"],
            random_forest.test_metrics["r2"],
            random_forest.best_params,
        ),
        _result_row(
            voting_equal.name,
            voting_equal.best_cv_rmse_log,
            voting_equal.test_metrics["mae"],
            voting_equal.test_metrics["rmse"],
            voting_equal.test_metrics["r2"],
            voting_equal.best_params,
        ),
        _result_row(
            voting_weighted.name,
            voting_weighted.best_cv_rmse_log,
            voting_weighted.test_metrics["mae"],
            voting_weighted.test_metrics["rmse"],
            voting_weighted.test_metrics["r2"],
            voting_weighted.best_params,
        ),
        _result_row(
            stacking.name,
            stacking.best_cv_rmse_log,
            stacking.test_metrics["mae"],
            stacking.test_metrics["rmse"],
            stacking.test_metrics["r2"],
            stacking.best_params,
        ),
    ]
    return rows


def _save_model_artifacts(
    output_dir: str,
    elastic: ElasticNetResult,
    knn: KnnResult,
    xgb: XgboostResult,
    svm: SvmResult,
    neural_network: NeuralNetworkResult,
    random_forest: RandomForestResult,
    voting_equal: EnsembleResult,
    voting_weighted: EnsembleResult,
    stacking: EnsembleResult,
) -> None:
    artifacts_dir: Path = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Keep artifact naming/order aligned with summary table model order.
    elastic_path: Path = artifacts_dir / "elastic_net.joblib"
    knn_path: Path = artifacts_dir / "knn.joblib"
    xgb_path: Path = artifacts_dir / "xgboost.joblib"
    svm_path: Path = artifacts_dir / "svm.joblib"
    neural_network_path: Path = artifacts_dir / "neural_network.joblib"
    random_forest_path: Path = artifacts_dir / "random_forest.joblib"
    voting_equal_path: Path = artifacts_dir / "voting_ensemble_equal.joblib"
    voting_weighted_path: Path = artifacts_dir / "voting_ensemble_weighted.joblib"
    stacking_path: Path = artifacts_dir / "stacking_ensemble.joblib"
    dump(elastic.pipeline, elastic_path)
    dump(knn.pipeline, knn_path)
    dump(xgb.pipeline, xgb_path)
    dump(svm.pipeline, svm_path)
    dump(neural_network.pipeline, neural_network_path)
    dump(random_forest.pipeline, random_forest_path)
    dump(voting_equal.pipeline, voting_equal_path)
    dump(voting_weighted.pipeline, voting_weighted_path)
    dump(stacking.pipeline, stacking_path)
    print(f"Wrote {elastic_path}", flush=True)
    print(f"Wrote {knn_path}", flush=True)
    print(f"Wrote {xgb_path}", flush=True)
    print(f"Wrote {svm_path}", flush=True)
    print(f"Wrote {neural_network_path}", flush=True)
    print(f"Wrote {random_forest_path}", flush=True)
    print(f"Wrote {voting_equal_path}", flush=True)
    print(f"Wrote {voting_weighted_path}", flush=True)
    print(f"Wrote {stacking_path}", flush=True)


def _resolve_trials(arg_value: int | None, sample_default: int, full_default: int, mode: str) -> int:
    if arg_value is not None:
        return arg_value
    if mode == "sample":
        return sample_default
    return full_default


def _build_training_data(mode: str, data: TrainTestData) -> TrainTestData:
    if mode == "full":
        return data
    sample_features: pd.DataFrame = data.X_train.sample(
        frac=SAMPLE_TRAIN_FRACTION,
        random_state=config.RANDOM_STATE,
    )
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
        default="sample",
        choices=TRAINING_MODES,
        help="Training mode: sample (fast iteration) or full (heavier search).",
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
    train_data: TrainTestData = _build_training_data(args.mode, full_data)
    elastic_trials: int = _resolve_trials(
        args.n_trials_elastic,
        SAMPLE_TRIALS_ELASTIC,
        config.N_TRIALS_ELASTICNET,
        args.mode,
    )
    knn_trials: int = _resolve_trials(
        args.n_trials_knn,
        SAMPLE_TRIALS_KNN,
        config.N_TRIALS_KNN,
        args.mode,
    )
    xgb_trials: int = _resolve_trials(
        args.n_trials_xgb,
        SAMPLE_TRIALS_XGB,
        config.N_TRIALS_XGBOOST,
        args.mode,
    )
    svm_trials: int = _resolve_trials(
        args.n_trials_svm,
        SAMPLE_TRIALS_SVM,
        config.N_TRIALS_SVM,
        args.mode,
    )
    nn_trials: int = _resolve_trials(
        args.n_trials_nn,
        SAMPLE_TRIALS_NN,
        config.N_TRIALS_NEURAL_NETWORK,
        args.mode,
    )
    rf_trials: int = _resolve_trials(
        args.n_trials_rf,
        SAMPLE_TRIALS_RF,
        config.N_TRIALS_RANDOM_FOREST,
        args.mode,
    )
    ensemble_weight_trials: int = _resolve_trials(
        args.n_trials_ensemble_weights,
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
    # Aggregate and display metrics before any optional persistence.
    rows: list[dict[str, Any]] = _collect_rows(
        elastic_result,
        knn_result,
        xgb_result,
        svm_result,
        neural_network_result,
        random_forest_result,
        voting_equal_result,
        voting_weighted_result,
        stacking_result,
    )
    table: pd.DataFrame = pd.DataFrame(rows)
    print(table.to_string(index=False), flush=True)
    if args.output_csv is not None:
        table.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}", flush=True)
    if args.model_output_dir is not None:
        _save_model_artifacts(
            args.model_output_dir,
            elastic_result,
            knn_result,
            xgb_result,
            svm_result,
            neural_network_result,
            random_forest_result,
            voting_equal_result,
            voting_weighted_result,
            stacking_result,
        )


if __name__ == "__main__":
    main()
