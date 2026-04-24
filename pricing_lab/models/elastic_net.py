"""ElasticNet with Optuna over alpha, l1_ratio, and max_iter."""

from dataclasses import dataclass

import optuna
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class ElasticNetResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def build_elastic_net_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    alpha: float = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
    l1_ratio: float = trial.suggest_float("l1_ratio", 0.0, 1.0)
    max_iter: int = trial.suggest_int("max_iter", 2000, 15000, step=500)
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ],
    )


def build_elastic_net_pipeline_from_params(params: dict[str, float | int]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=float(params["alpha"]),
                    l1_ratio=float(params["l1_ratio"]),
                    max_iter=int(params["max_iter"]),
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ],
    )


def tune_elastic_net(data: TrainTestData, n_trials: int | None = None) -> ElasticNetResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_ELASTICNET
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_elastic_net_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    # Refit once on full training data after CV-based hyperparameter selection.
    best_params: dict[str, float | int] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_elastic_net_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return ElasticNetResult(
        name="ElasticNet",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
