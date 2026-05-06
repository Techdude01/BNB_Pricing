import unittest

from pricing_lab.data import TrainTestData
from pricing_lab.models.knn import build_knn_pipeline
from pricing_lab.models.knn import build_knn_pipeline_from_params, tune_knn
from pricing_lab.tuning import create_study
import numpy as np
import pandas as pd


class KnnSearchSpaceTest(unittest.TestCase):
    def test_knn_uses_fixed_euclidean_distance(self) -> None:
        study = create_study()
        trial = study.ask()
        pipeline = build_knn_pipeline(trial)

        assert pipeline.named_steps["model"].get_params()["p"] == 2
        assert "p" not in trial.params

    def test_knn_from_params_defaults_to_euclidean_distance(self) -> None:
        pipeline = build_knn_pipeline_from_params(
            {
                "n_neighbors": 5,
                "weights": "uniform",
                "leaf_size": 30,
            }
        )

        assert pipeline.named_steps["model"].get_params()["p"] == 2

    def test_tuned_knn_records_fixed_distance_param(self) -> None:
        X_train = pd.DataFrame(
            {
                "neighbourhood_group": ["Bronx", "Brooklyn", "Manhattan", "Queens"] * 10,
                "neighbourhood": ["A", "B", "C", "D"] * 10,
                "room_type": ["Private room", "Entire home/apt"] * 20,
                "latitude": np.linspace(40.0, 41.0, 40),
                "longitude": np.linspace(-74.0, -73.0, 40),
                "minimum_nights": [1, 2, 3, 4] * 10,
                "number_of_reviews": [0, 1, 2, 3] * 10,
                "reviews_per_month": [0.0, 0.1, 0.2, 0.3] * 10,
                "calculated_host_listings_count": [1, 2, 3, 4] * 10,
                "availability_365": [0, 30, 180, 365] * 10,
                "days_since_last_review": [30, 180, 365, 3650] * 10,
                "has_reviews": [0, 1, 1, 1] * 10,
                "log_minimum_nights": np.log1p([1, 2, 3, 4] * 10),
                "log_host_listing_count": np.log1p([1, 2, 3, 4] * 10),
                "is_professional_host": [0, 1, 1, 1] * 10,
                "is_long_stay": [0, 0, 0, 0] * 10,
                "availability_bucket": ["unavailable", "low", "seasonal", "high"] * 10,
                "minimum_nights_bucket": ["one_night", "two_to_three", "two_to_three", "four_to_seven"] * 10,
                "host_listing_count_bucket": [
                    "single_listing",
                    "small_portfolio",
                    "small_portfolio",
                    "small_portfolio",
                ]
                * 10,
                "review_recency_bucket": ["last_30_days", "last_6_months", "last_year", "older_or_none"] * 10,
                "geo_cell": ["g1", "g2", "g3", "g4"] * 10,
                "room_type_neighbourhood_group": ["r1", "r2", "r3", "r4"] * 10,
                "room_type_availability_bucket": ["a1", "a2", "a3", "a4"] * 10,
                "room_type_minimum_nights_bucket": ["m1", "m2", "m3", "m4"] * 10,
                "room_type_neighbourhood": ["rn1", "rn2", "rn3", "rn4"] * 10,
            }
        )
        y_train = pd.Series(np.log1p(np.linspace(80.0, 200.0, 40)))
        data = TrainTestData(
            X_train=X_train,
            X_test=X_train.head(8),
            y_train=y_train,
            y_test=y_train.head(8),
        )

        result = tune_knn(data, n_trials=1)

        assert result.best_params["p"] == 2


if __name__ == "__main__":
    unittest.main()
