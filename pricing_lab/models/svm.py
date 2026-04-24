"""Support Vector Regression with Optuna."""

from dataclasses import dataclass

import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pricing_lab import config
from pricing_lab.data import TrainTestData, build_column_transformer
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study, mean_cv_rmse_log


@dataclass(frozen=True)
class SvmResult:
    """Outcome of tuning and test-set evaluation."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | int | str]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def build_svm_pipeline(trial: optuna.Trial) -> Pipeline:
    """Sample hyperparameters from Optuna and return an unfitted pipeline."""
    c_value: float = trial.suggest_float("C", 1e-2, 200.0, log=True)
    epsilon: float = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
    gamma: str = trial.suggest_categorical("gamma", ["scale", "auto"])
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=c_value,
                    epsilon=epsilon,
                    gamma=gamma,
                ),
            ),
        ],
    )


def build_svm_pipeline_from_params(params: dict[str, float | int | str]) -> Pipeline:
    """Rebuild the best pipeline after the study completes."""
    return Pipeline(
        steps=[
            ("prep", build_column_transformer()),
            ("scale", StandardScaler()),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=float(params["C"]),
                    epsilon=float(params["epsilon"]),
                    gamma=str(params["gamma"]),
                ),
            ),
        ],
    )


def tune_svm(data: TrainTestData, n_trials: int | None = None) -> SvmResult:
    """Run Optuna, refit on full training data, evaluate on the held-out test set."""
    trials: int = n_trials if n_trials is not None else config.N_TRIALS_SVM
    study: optuna.Study = create_study()

    def objective(trial: optuna.Trial) -> float:
        pipeline: Pipeline = build_svm_pipeline(trial)
        return mean_cv_rmse_log(pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_params: dict[str, float | int | str] = {k: study.best_params[k] for k in study.best_params}
    best_pipeline: Pipeline = build_svm_pipeline_from_params(best_params)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return SvmResult(
        name="SVM",
        best_cv_rmse_log=float(study.best_value),
        best_params=best_params,
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )
