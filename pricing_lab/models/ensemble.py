"""Ensembling helpers: equal voting, weighted voting, and stacking."""

from dataclasses import dataclass

import optuna
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pricing_lab.data import TrainTestData
from pricing_lab.metrics import DollarMetrics, compute_dollar_metrics
from pricing_lab.tuning import create_study
from pricing_lab.tuning import mean_cv_rmse_log


@dataclass(frozen=True)
class EnsembleResult:
    """Outcome of fitting and evaluating the ensemble model."""

    name: str
    best_cv_rmse_log: float
    best_params: dict[str, float | str]
    test_metrics: DollarMetrics
    pipeline: Pipeline


def select_ensemble_candidates(
    base_pipelines: dict[str, Pipeline],
    base_cv_rmse_log: dict[str, float],
    cv_window: float = 0.03,
) -> dict[str, Pipeline]:
    """Keep models whose CV RMSE is within a relative window of the best model."""
    best_cv: float = min(base_cv_rmse_log.values())
    max_accepted_cv: float = best_cv * (1.0 + cv_window)
    selected: dict[str, Pipeline] = {
        name: pipeline
        for name, pipeline in base_pipelines.items()
        if base_cv_rmse_log[name] <= max_accepted_cv
    }
    # Always keep at least two models to make ensembling meaningful.
    if len(selected) >= 2:
        return selected
    sorted_names: list[str] = sorted(base_cv_rmse_log, key=base_cv_rmse_log.get)
    return {name: base_pipelines[name] for name in sorted_names[:2]}


def build_equal_voting_pipeline(base_pipelines: dict[str, Pipeline]) -> Pipeline:
    """Create an equal-weight VotingRegressor over cloned base pipelines."""
    estimators: list[tuple[str, Pipeline]] = [
        (name, clone(pipeline)) for name, pipeline in base_pipelines.items()
    ]
    return Pipeline(steps=[("model", VotingRegressor(estimators=estimators, n_jobs=-1))])


def _normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    weight_sum: float = float(sum(raw_weights.values()))
    if weight_sum <= 0.0:
        uniform_weight: float = 1.0 / len(raw_weights)
        return {name: uniform_weight for name in raw_weights}
    return {name: weight / weight_sum for name, weight in raw_weights.items()}


def build_weighted_voting_pipeline(
    base_pipelines: dict[str, Pipeline],
    weights_by_name: dict[str, float],
) -> Pipeline:
    """Create a weighted VotingRegressor over cloned base pipelines."""
    estimators: list[tuple[str, Pipeline]] = [
        (name, clone(base_pipelines[name])) for name in base_pipelines
    ]
    normalized_weights: dict[str, float] = _normalize_weights(weights_by_name)
    weights: list[float] = [normalized_weights[name] for name in base_pipelines]
    return Pipeline(
        steps=[
            (
                "model",
                VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1),
            )
        ]
    )


def fit_equal_voting_ensemble(data: TrainTestData, base_pipelines: dict[str, Pipeline]) -> EnsembleResult:
    """Fit an equal-weight voting ensemble and evaluate it with CV + held-out test metrics."""
    ensemble_pipeline: Pipeline = build_equal_voting_pipeline(base_pipelines)
    cv_rmse_log: float = mean_cv_rmse_log(ensemble_pipeline, data.X_train, data.y_train)
    ensemble_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = ensemble_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return EnsembleResult(
        name="VotingEnsembleEqual",
        best_cv_rmse_log=cv_rmse_log,
        best_params={"estimators": ",".join(base_pipelines.keys())},
        test_metrics=test_metrics,
        pipeline=ensemble_pipeline,
    )


def fit_weighted_voting_ensemble(
    data: TrainTestData,
    base_pipelines: dict[str, Pipeline],
    n_trials: int = 20,
) -> EnsembleResult:
    """Tune voting weights with Optuna, then fit and evaluate the weighted ensemble."""
    study: optuna.Study = create_study()
    model_names: list[str] = list(base_pipelines.keys())

    def objective(trial: optuna.Trial) -> float:
        raw_weights: dict[str, float] = {
            name: trial.suggest_float(f"w_{name}", 0.0, 1.0)
            for name in model_names
        }
        voting_pipeline: Pipeline = build_weighted_voting_pipeline(base_pipelines, raw_weights)
        return mean_cv_rmse_log(voting_pipeline, data.X_train, data.y_train)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_raw_weights: dict[str, float] = {
        name: float(study.best_params[f"w_{name}"])
        for name in model_names
    }
    best_weights: dict[str, float] = _normalize_weights(best_raw_weights)
    best_pipeline: Pipeline = build_weighted_voting_pipeline(base_pipelines, best_weights)
    best_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = best_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return EnsembleResult(
        name="VotingEnsembleWeighted",
        best_cv_rmse_log=float(study.best_value),
        best_params={
            **{f"weight_{name}": round(best_weights[name], 6) for name in model_names},
            "estimators": ",".join(model_names),
        },
        test_metrics=test_metrics,
        pipeline=best_pipeline,
    )


def fit_stacking_ensemble(data: TrainTestData, base_pipelines: dict[str, Pipeline]) -> EnsembleResult:
    """Fit a stacking ensemble and evaluate it with CV + held-out test metrics."""
    estimators: list[tuple[str, Pipeline]] = [
        (name, clone(base_pipelines[name])) for name in base_pipelines
    ]
    final_estimator: Ridge = Ridge(alpha=1.0, random_state=42)
    stacking_pipeline: Pipeline = Pipeline(
        steps=[
            (
                "model",
                StackingRegressor(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    n_jobs=-1,
                ),
            )
        ]
    )
    cv_rmse_log: float = mean_cv_rmse_log(stacking_pipeline, data.X_train, data.y_train)
    stacking_pipeline.fit(data.X_train, data.y_train)
    y_pred_log = stacking_pipeline.predict(data.X_test)
    test_metrics: DollarMetrics = compute_dollar_metrics(data.y_test.values, y_pred_log)
    return EnsembleResult(
        name="StackingEnsemble",
        best_cv_rmse_log=cv_rmse_log,
        best_params={
            "estimators": ",".join(base_pipelines.keys()),
            "meta_model": "Ridge(alpha=1.0)",
        },
        test_metrics=test_metrics,
        pipeline=stacking_pipeline,
    )
