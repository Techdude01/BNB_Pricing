"""Shared Optuna + cross-validation helpers for regression on log1p(price)."""

import json
import time
from collections.abc import Callable

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score

from pricing_lab import config


def mean_cv_rmse_log(
    estimator: object,
    X: object,
    y: object,
    cv_splits: int = config.CV_SPLITS,
    random_state: int = config.RANDOM_STATE,
) -> float:
    """Return mean RMSE on the log target across K-fold CV (lower is better)."""
    # This score is for model selection only; final reporting uses held-out test metrics.
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores: np.ndarray = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )
    # sklearn returns negative losses for minimization-compatible scoring APIs.
    return float(-np.mean(scores))


def create_study() -> optuna.Study:
    """TPE sampler with fixed seed for reproducibility across models."""
    sampler: optuna.samplers.TPESampler = optuna.samplers.TPESampler(
        seed=config.OPTUNA_SAMPLER_SEED,
    )
    return optuna.create_study(direction="minimize", sampler=sampler)


def optimize_with_logs(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    model_name: str,
    n_trials: int,
) -> None:
    """Run Optuna with concise start/finish logs for long sequential studies."""

    def logged_objective(trial: optuna.Trial) -> float:
        print(f"[{model_name}] trial {trial.number + 1}/{n_trials} started", flush=True)
        started_at = time.perf_counter()
        try:
            return objective(trial)
        finally:
            trial.set_user_attr("elapsed_seconds", time.perf_counter() - started_at)

    def log_completed_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            value_text = trial.state.name.lower()
        else:
            value_text = f"{trial.value:.6f}"
        elapsed_text = f"{trial.user_attrs.get('elapsed_seconds', 0.0):.1f}s"
        params_text = json.dumps(trial.params, sort_keys=True)
        complete_trials = [
            completed_trial
            for completed_trial in study.trials
            if completed_trial.state == optuna.trial.TrialState.COMPLETE and completed_trial.value is not None
        ]
        best_text = f"{study.best_value:.6f}" if complete_trials else "n/a"
        print(
            f"[{model_name}] trial {trial.number + 1}/{n_trials} finished "
            f"elapsed={elapsed_text} rmse_log={value_text} best={best_text} params={params_text}",
            flush=True,
        )

    study.optimize(
        logged_objective,
        n_trials=n_trials,
        callbacks=[log_completed_trial],
        show_progress_bar=False,
    )
