import numpy as np
import pandas as pd
import unittest

from pricing_lab.diagnostics import (
    build_residual_frame,
    summarize_price_bands,
    summarize_price_deciles,
    top_price_band_warning,
)


class DiagnosticsTest(unittest.TestCase):
    def test_build_residual_frame_tracks_log_dollar_ratio_and_percent_errors(self) -> None:
        X_test = pd.DataFrame({"room_type": ["Private room", "Entire home/apt"]})
        y_true_log = np.log1p(np.array([100.0, 200.0]))
        y_pred_log = np.log1p(np.array([90.0, 150.0]))

        frame = build_residual_frame(X_test, y_true_log, y_pred_log)

        np.testing.assert_allclose(frame["dollar_residual"].to_numpy(), [10.0, 50.0])
        np.testing.assert_allclose(frame["absolute_error"].to_numpy(), [10.0, 50.0])
        np.testing.assert_allclose(frame["predicted_to_actual_ratio"].to_numpy(), [0.9, 0.75])
        np.testing.assert_allclose(frame["absolute_percentage_error"].to_numpy(), [0.1, 0.25])
        np.testing.assert_allclose(frame["log_residual"].to_numpy(), y_true_log - y_pred_log)

    def test_price_band_summary_includes_log_error_and_high_end_warning(self) -> None:
        X_test = pd.DataFrame({"room_type": ["A", "A", "B", "B", "B"]})
        actual_prices = np.array([50.0, 75.0, 100.0, 200.0, 300.0])
        predicted_prices = np.array([55.0, 70.0, 95.0, 140.0, 200.0])
        frame = build_residual_frame(X_test, np.log1p(actual_prices), np.log1p(predicted_prices))

        summary = summarize_price_bands(frame, q=2)
        warning = top_price_band_warning(summary)

        assert {"log_rmse", "mean_abs_log_error", "prediction_ratio", "mape"}.issubset(summary.columns)
        assert "underpredicted" in warning

    def test_price_decile_summary_reports_ratio_and_mape(self) -> None:
        X_test = pd.DataFrame({"feature": range(10)})
        actual_prices = np.linspace(50.0, 500.0, 10)
        predicted_prices = actual_prices * 0.9
        frame = build_residual_frame(X_test, np.log1p(actual_prices), np.log1p(predicted_prices))

        summary = summarize_price_deciles(frame)

        assert len(summary) == 10
        assert "prediction_ratio_median" in summary.columns
        assert "mape" in summary.columns
        np.testing.assert_allclose(summary["prediction_ratio_median"].to_numpy(), [0.9] * 10)


if __name__ == "__main__":
    unittest.main()
