import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pricing_lab.data import TrainTestData
from pricing_lab.run_all import (
    _load_result_checkpoint,
    _representative_train_sample,
    _save_result_checkpoint,
    _write_results_table,
)


class RunAllSamplingTest(unittest.TestCase):
    def test_representative_train_sample_is_deterministic_and_segmented(self) -> None:
        rows = 100
        X_train = pd.DataFrame(
            {
                "neighbourhood_group": ["Bronx", "Brooklyn", "Manhattan", "Queens"] * 25,
                "room_type": ["Private room", "Entire home/apt"] * 50,
            }
        )
        y_train = pd.Series(np.log1p(np.linspace(50.0, 300.0, rows)))
        data = TrainTestData(
            X_train=X_train,
            X_test=X_train.head(5),
            y_train=y_train,
            y_test=y_train.head(5),
        )

        first_features, first_target = _representative_train_sample(data, target_rows=20)
        second_features, second_target = _representative_train_sample(data, target_rows=20)

        assert len(first_features) == 20
        assert first_features.index.tolist() == second_features.index.tolist()
        np.testing.assert_allclose(first_target.to_numpy(), second_target.to_numpy())
        assert first_features["neighbourhood_group"].nunique() > 1
        assert first_features["room_type"].nunique() > 1


class RunAllCheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_requires_matching_context(self) -> None:
        result = SimpleNamespace(
            name="ElasticNet",
            best_cv_rmse_log=0.123,
            best_params={"alpha": 0.5},
            test_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.3},
            pipeline={"fitted": True},
        )
        context = {
            "mode": "lite",
            "csv_path": "/tmp/AB_NYC_2019.csv",
            "train_rows": 20,
            "test_rows": 5,
            "trial_count": 1,
        }

        with TemporaryDirectory() as temporary_directory:
            _save_result_checkpoint(temporary_directory, result, context)

            loaded = _load_result_checkpoint(temporary_directory, "ElasticNet", context)
            assert loaded is not None
            assert loaded.name == "ElasticNet"
            assert loaded.best_params == {"alpha": 0.5}
            assert loaded.pipeline == {"fitted": True}

            mismatched_context = {**context, "trial_count": 2}
            assert _load_result_checkpoint(temporary_directory, "ElasticNet", mismatched_context) is None

    def test_write_results_table_creates_parent_directory(self) -> None:
        result = SimpleNamespace(
            name="ElasticNet",
            best_cv_rmse_log=0.1234567,
            best_params={"alpha": 0.5},
            test_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.3},
        )

        with TemporaryDirectory() as temporary_directory:
            output_csv = f"{temporary_directory}/nested/metrics.csv"
            table = _write_results_table(output_csv, [result])

            assert table.loc[0, "model"] == "ElasticNet"
            assert pd.read_csv(output_csv).loc[0, "model"] == "ElasticNet"


if __name__ == "__main__":
    unittest.main()
