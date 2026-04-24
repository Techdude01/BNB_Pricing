"""Train ElasticNet, KNN, and XGBoost with Optuna and print a comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from joblib import dump

from pricing_lab import config
from pricing_lab.data import TrainTestData, load_train_test
from pricing_lab.models.elastic_net import ElasticNetResult, tune_elastic_net
from pricing_lab.models.knn import KnnResult, tune_knn
from pricing_lab.models.xgboost_model import XgboostResult, tune_xgboost

TRAINING_MODES: tuple[str, str] = ("sample", "full")
SAMPLE_TRAIN_FRACTION: float = 0.35
SAMPLE_TRIALS_ELASTIC: int = 10
SAMPLE_TRIALS_KNN: int = 6
SAMPLE_TRIALS_XGB: int = 14


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
    ]
    return rows


def _save_model_artifacts(
    output_dir: str,
    elastic: ElasticNetResult,
    knn: KnnResult,
    xgb: XgboostResult,
) -> None:
    artifacts_dir: Path = Path(output_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Keep artifact naming/order aligned with summary table model order.
    elastic_path: Path = artifacts_dir / "elastic_net.joblib"
    knn_path: Path = artifacts_dir / "knn.joblib"
    xgb_path: Path = artifacts_dir / "xgboost.joblib"
    dump(elastic.pipeline, elastic_path)
    dump(knn.pipeline, knn_path)
    dump(xgb.pipeline, xgb_path)
    print(f"Wrote {elastic_path}", flush=True)
    print(f"Wrote {knn_path}", flush=True)
    print(f"Wrote {xgb_path}", flush=True)


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
        description="Optuna tuning for ElasticNet, KNN, and XGBoost (log1p price target).",
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
    print(
        f"Trials => ElasticNet: {elastic_trials}, KNN: {knn_trials}, XGBoost: {xgb_trials}",
        flush=True,
    )
    print("Tuning ElasticNet...", flush=True)
    elastic_result: ElasticNetResult = tune_elastic_net(train_data, n_trials=elastic_trials)
    print("Tuning KNN...", flush=True)
    knn_result: KnnResult = tune_knn(train_data, n_trials=knn_trials)
    print("Tuning XGBoost...", flush=True)
    xgb_result: XgboostResult = tune_xgboost(train_data, n_trials=xgb_trials)
    # Aggregate and display metrics before any optional persistence.
    rows: list[dict[str, Any]] = _collect_rows(elastic_result, knn_result, xgb_result)
    table: pd.DataFrame = pd.DataFrame(rows)
    print(table.to_string(index=False), flush=True)
    if args.output_csv is not None:
        table.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}", flush=True)
    if args.model_output_dir is not None:
        _save_model_artifacts(args.model_output_dir, elastic_result, knn_result, xgb_result)


if __name__ == "__main__":
    main()
