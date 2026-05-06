import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from pricing_lab import config
from pricing_lab.models.neural_network import (
    TargetClippedMLPRegressor,
    build_neural_network_pipeline_from_params,
)


class NeuralNetworkModelTest(unittest.TestCase):
    def test_target_clipped_mlp_limits_extreme_log_predictions(self) -> None:
        model = TargetClippedMLPRegressor(
            hidden_layer_sizes=(1,),
            max_iter=50,
            random_state=config.RANDOM_STATE,
        )
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([2.0, 3.0, 4.0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X, y)

        # Force a pathological fitted network output, matching the stale NN artifact failure mode.
        model.coefs_ = [np.zeros_like(model.coefs_[0]), np.zeros_like(model.coefs_[1])]
        model.intercepts_ = [np.zeros_like(model.intercepts_[0]), np.array([100.0])]

        predictions = model.predict(X)

        np.testing.assert_allclose(predictions, np.array([4.0, 4.0, 4.0]))

    def test_rebuilt_pipeline_uses_clipped_mlp(self) -> None:
        pipeline = build_neural_network_pipeline_from_params(
            {
                "hidden_layer_name": "32_16",
                "activation": "relu",
                "alpha": 0.01,
                "batch_size": 128,
                "learning_rate_init": 0.0003,
                "max_iter": 1200,
            }
        )

        assert isinstance(pipeline.named_steps["model"], TargetClippedMLPRegressor)


if __name__ == "__main__":
    unittest.main()
