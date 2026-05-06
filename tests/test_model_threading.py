import unittest

from pricing_lab import config
from pricing_lab.models.random_forest import build_random_forest_pipeline_from_params
from pricing_lab.models.xgboost_model import build_xgboost_pipeline_from_params


class ModelThreadingTest(unittest.TestCase):
    def test_xgboost_uses_configured_model_n_jobs(self) -> None:
        pipeline = build_xgboost_pipeline_from_params(
            {
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
                "gamma": 0.01,
            }
        )

        assert pipeline.named_steps["model"].get_params()["n_jobs"] == config.MODEL_N_JOBS

    def test_random_forest_uses_configured_model_n_jobs(self) -> None:
        pipeline = build_random_forest_pipeline_from_params(
            {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }
        )

        assert pipeline.named_steps["model"].get_params()["n_jobs"] == config.MODEL_N_JOBS


if __name__ == "__main__":
    unittest.main()
