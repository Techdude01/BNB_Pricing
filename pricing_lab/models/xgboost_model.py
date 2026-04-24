"""XGBoost regressor with Optuna."""

from dataclasses import dataclass

import optuna
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class XgboostResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def build_xgboost_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    max_depth: int = trial.suggest_int("max_depth", 3, 12)
    learning_rate: float = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    n_estimators: int = trial.suggest_int("n_estimators", 100, 800, step=50)
    subsample: float = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree: float = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    min_child_weight: float = trial.suggest_float("min_child_weight", 1.0, 10.0)
    reg_alpha: float = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    reg_lambda: float = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    gamma: float = trial.suggest_float("gamma", 1e-8, 5.0, log=True)
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            (
                "model",
                XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                    objective="reg:squarederror",
                ),
            ),
        ],
    )


def build_xgboost_pipeline_from_params(params: dict[str, float | int]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            (
                "model",
                XGBRegressor(
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    n_estimators=int(params["n_estimators"]),
                    subsample=float(params["subsample"]),
                    colsample_bytree=float(params["colsample_bytree"]),
                    min_child_weight=float(params["min_child_weight"]),
                    reg_alpha=float(params["reg_alpha"]),
                    reg_lambda=float(params["reg_lambda"]),
                    gamma=float(params["gamma"]),
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                    objective="reg:squarederror",
                ),
            ),
        ],
    )


def tune_xgboost(data: TrainTestData, n_trials: int | None = None) -> XgboostResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_XGBOOST
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_xgboost_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    # Refit once on full training data after CV-based hyperparameter selection.
    best_params: dict[str, float | int] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_xgboost_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return XgboostResult(
        name="XGBoost",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
