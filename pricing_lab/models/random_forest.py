"""Random Forest regressor with Optuna."""

from dataclasses import dataclass

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class RandomForestResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int | str]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def build_random_forest_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    n_estimators: int = trial.suggest_int("n_estimators", 200, 900, step=100)
    max_depth: int = trial.suggest_int("max_depth", 4, 28)
    min_samples_split: int = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf: int = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features: str = trial.suggest_categorical("max_features", ["sqrt", "log2", "None"])
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=None if max_features == "None" else max_features,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ],
    )


def build_random_forest_pipeline_from_params(params: dict[str, float | int | str]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    max_features_value: str = str(params["max_features"])
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    min_samples_split=int(params["min_samples_split"]),
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    max_features=None if max_features_value == "None" else max_features_value,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ],
    )


def tune_random_forest(
    data: TrainTestData,
    n_trials: int | None = None,
) -> RandomForestResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_RANDOM_FOREST
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_random_forest_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params: dict[str, float | int | str] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_random_forest_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return RandomForestResult(
        name="RandomForest",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
