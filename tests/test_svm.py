import unittest

from pricing_lab.models.svm import build_svm_pipeline, build_svm_pipeline_from_params
from pricing_lab.tuning import create_study


class SvmSearchSpaceTest(unittest.TestCase):
    def test_svm_uses_fixed_scale_gamma(self) -> None:
        study = create_study()
        trial = study.ask()
        pipeline = build_svm_pipeline(trial)

        assert pipeline.named_steps["model"].get_params()["gamma"] == "scale"
        assert "gamma" not in trial.params

    def test_svm_search_space_avoids_expensive_rbf_extremes(self) -> None:
        study = create_study()
        for _ in range(20):
            trial = study.ask()
            build_svm_pipeline(trial)
            assert 0.05 <= trial.params["C"] <= 1.0
            assert 0.3 <= trial.params["epsilon"] <= 0.8

    def test_svm_from_params_preserves_gamma(self) -> None:
        pipeline = build_svm_pipeline_from_params(
            {
                "C": 1.0,
                "epsilon": 0.1,
                "gamma": "scale",
            }
        )

        assert pipeline.named_steps["model"].get_params()["gamma"] == "scale"


if __name__ == "__main__":
    unittest.main()
