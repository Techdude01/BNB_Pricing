"""Shared Optuna + cross-validation helpers for regression on log1p(price)."""

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
        n_jobs=-1,
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
