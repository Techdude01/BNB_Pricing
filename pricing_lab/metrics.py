"""Regression metrics on dollar scale from log1p targets."""

from typing import TypedDict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DollarMetrics(TypedDict):
    """Holds MAE, RMSE, and R² on the original price scale."""

    mae: float
    rmse: float
    r2: float


def compute_dollar_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> DollarMetrics:
    """Map predictions from log1p space to dollars and compute MAE, RMSE, R²."""
    actual_price: np.ndarray = np.expm1(np.asarray(y_true_log, dtype=np.float64))
    predicted_price: np.ndarray = np.expm1(np.asarray(y_pred_log, dtype=np.float64))
    if actual_price.shape != predicted_price.shape:
        raise ValueError("Metric arrays must have matching shapes.")
    if not np.all(np.isfinite(predicted_price)):
        raise ValueError("Predictions contain non-finite values after inverse transform.")
    mae_value: float = float(mean_absolute_error(actual_price, predicted_price))
    rmse_value: float = float(np.sqrt(mean_squared_error(actual_price, predicted_price)))
    if rmse_value + 1e-12 < mae_value:
        raise ValueError("Invalid dollar metrics: RMSE must be greater than or equal to MAE.")
    r2_value: float = float(r2_score(actual_price, predicted_price))
    return DollarMetrics(mae=mae_value, rmse=rmse_value, r2=r2_value)
