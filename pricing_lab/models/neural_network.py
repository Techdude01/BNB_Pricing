"""MLP regressor with Optuna."""

from dataclasses import dataclass

import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class NeuralNetworkResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int | str]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def _hidden_layers_from_name(hidden_name: str) -> tuple[int, ...]:
    if hidden_name == "64":
        return (64,)
    if hidden_name == "128":
        return (128,)
    if hidden_name == "64_32":
        return (64, 32)
    return (128, 64)


def build_neural_network_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    hidden_layer_name: str = trial.suggest_categorical("hidden_layer_name", ["64", "128", "64_32", "128_64"])
    alpha: float = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    learning_rate_init: float = trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True)
    max_iter: int = trial.suggest_int("max_iter", 300, 900, step=100)
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=_hidden_layers_from_name(hidden_layer_name),
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=True,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ],
    )


def build_neural_network_pipeline_from_params(params: dict[str, float | int | str]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=_hidden_layers_from_name(str(params["hidden_layer_name"])),
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate_init"]),
                    max_iter=int(params["max_iter"]),
                    early_stopping=True,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ],
    )


def tune_neural_network(
    data: TrainTestData,
    n_trials: int | None = None,
) -> NeuralNetworkResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_NEURAL_NETWORK
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_neural_network_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params: dict[str, float | int | str] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_neural_network_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return NeuralNetworkResult(
        name="NeuralNetwork",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
