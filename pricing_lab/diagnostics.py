"""Residual diagnostics for log-target price models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_residual_frame(
    X_test: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> pd.DataFrame:
    """Return one evaluation frame with log, dollar, ratio, and percent errors."""
    actual_log = np.asarray(y_true_log, dtype=np.float64)
    predicted_log = np.asarray(y_pred_log, dtype=np.float64)
    if actual_log.shape != predicted_log.shape:
        raise ValueError("Actual and predicted log arrays must have matching shapes.")
    actual_price = np.expm1(actual_log)
    predicted_price = np.expm1(predicted_log)
    if not np.all(np.isfinite(predicted_price)):
        raise ValueError("Predicted prices contain non-finite values after inverse transform.")

    frame = X_test.reset_index(drop=True).copy()
    frame["actual_log_price"] = actual_log
    frame["predicted_log_price"] = predicted_log
    frame["log_residual"] = actual_log - predicted_log
    frame["absolute_log_residual"] = frame["log_residual"].abs()
    frame["actual_price"] = actual_price
    frame["predicted_price"] = predicted_price
    frame["dollar_residual"] = actual_price - predicted_price
    frame["absolute_error"] = frame["dollar_residual"].abs()
    safe_actual = frame["actual_price"].clip(lower=1.0)
    frame["predicted_to_actual_ratio"] = frame["predicted_price"] / safe_actual
    frame["absolute_percentage_error"] = frame["absolute_error"] / safe_actual
    return frame


def summarize_price_bands(
    residual_frame: pd.DataFrame,
    q: int = 5,
) -> pd.DataFrame:
    """Summarize dollar and log residuals by actual-price quantile bands."""
    frame = residual_frame.copy()
    frame["actual_price_band"] = pd.qcut(frame["actual_price"], q=q, duplicates="drop")
    return summarize_segment(frame, "actual_price_band", min_rows=1)


def summarize_segment(
    residual_frame: pd.DataFrame,
    column_name: str,
    min_rows: int = 30,
) -> pd.DataFrame:
    """Summarize residual magnitude, bias, and ratio error by segment."""
    summary = (
        residual_frame.groupby(column_name, observed=True)
        .agg(
            rows=("absolute_error", "size"),
            actual_median=("actual_price", "median"),
            predicted_median=("predicted_price", "median"),
            mae=("absolute_error", "mean"),
            rmse=("dollar_residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            bias=("dollar_residual", "mean"),
            log_rmse=("log_residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            mean_abs_log_error=("absolute_log_residual", "mean"),
            prediction_ratio=("predicted_to_actual_ratio", "median"),
            mape=("absolute_percentage_error", "mean"),
        )
        .query("rows >= @min_rows")
        .sort_values("rmse", ascending=False)
    )
    return summary.round(3)


def summarize_price_deciles(residual_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize calibration, ratio, and error by actual-price decile."""
    frame = residual_frame.copy()
    frame["actual_price_decile"] = pd.qcut(frame["actual_price"], q=10, duplicates="drop")
    return (
        frame.groupby("actual_price_decile", observed=True)
        .agg(
            rows=("absolute_error", "size"),
            actual_mean=("actual_price", "mean"),
            predicted_mean=("predicted_price", "mean"),
            residual_mean=("dollar_residual", "mean"),
            mae=("absolute_error", "mean"),
            log_rmse=("log_residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            prediction_ratio_median=("predicted_to_actual_ratio", "median"),
            mape=("absolute_percentage_error", "mean"),
        )
        .reset_index()
        .round(3)
    )


def top_price_band_warning(price_band_summary: pd.DataFrame) -> str:
    """Return a concise warning about high-end compression when the top band is underpredicted."""
    if price_band_summary.empty:
        return "No price-band rows available for high-end compression check."
    top_band = price_band_summary.sort_values("actual_median").iloc[-1]
    bias = float(top_band["bias"])
    ratio = float(top_band["prediction_ratio"])
    if bias > 0.0 and ratio < 1.0:
        return (
            "Top actual-price band is underpredicted: "
            f"median predicted/actual ratio is {ratio:.3f}, mean residual is ${bias:,.2f}."
        )
    return "Top actual-price band does not show median underprediction in this run."
