import unittest

import numpy as np
import pandas as pd

from pricing_lab.data import TrainTestData
from pricing_lab.run_all import _representative_train_sample


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


if __name__ == "__main__":
    unittest.main()
