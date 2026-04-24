"""K-nearest neighbors regression with Optuna."""

from dataclasses import dataclass

import optuna
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class KnnResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int | str]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def build_knn_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    n_neighbors: int = trial.suggest_int("n_neighbors", 3, 80)
    weights: str = trial.suggest_categorical("weights", ["uniform", "distance"])
    p: int = trial.suggest_int("p", 1, 2)
    leaf_size: int = trial.suggest_int("leaf_size", 20, 50)
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p,
                    leaf_size=leaf_size,
                    n_jobs=1,
                ),
            ),
        ],
    )


def build_knn_pipeline_from_params(params: dict[str, float | int | str]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=int(params["n_neighbors"]),
                    weights=str(params["weights"]),
                    p=int(params["p"]),
                    leaf_size=int(params["leaf_size"]),
                    n_jobs=1,
                ),
            ),
        ],
    )


def tune_knn(data: TrainTestData, n_trials: int | None = None) -> KnnResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_KNN
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_knn_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    # Refit once on full training data after CV-based hyperparameter selection.
    best_params: dict[str, float | int | str] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_knn_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return KnnResult(
        name="KNN",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
