"""Load AB_NYC_2019.csv, apply notebook cleaning, and build sklearn preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pricing_lab import config

COLUMNS_TO_DROP: list[str] = ["id", "name", "host_id", "host_name", "last_review"]
CATEGORICAL_FEATURES: list[str] = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
    "availability_bucket",
    "minimum_nights_bucket",
    "host_listing_count_bucket",
    "review_recency_bucket",
    "geo_cell",
    "room_type_neighbourhood_group",
    "room_type_availability_bucket",
]
TARGET_ENCODED_FEATURES: list[str] = [
    "neighbourhood",
    "geo_cell",
    "room_type_neighbourhood_group",
]
NUMERIC_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "minimum_nights",
    "log_minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "log_host_listing_count",
    "availability_365",
    "has_reviews",
    "days_since_last_review",
    "is_long_stay",
]


@dataclass(frozen=True)
class TrainTestData:
    """Train and test tensors for log1p(price) regression."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical columns with smoothed target means inside CV folds."""

    def __init__(self, smoothing: float = 10.0) -> None:
        self.smoothing = smoothing

    def fit(self, X: object, y: object) -> "TargetMeanEncoder":
        data_frame: pd.DataFrame = self._to_frame(X)
        target_series: pd.Series = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(target_series.mean())
        self.feature_maps_: dict[str, dict[object, float]] = {}
        for column_name in data_frame.columns:
            grouped = target_series.groupby(data_frame[column_name], dropna=False).agg(["mean", "count"])
            smoothed = (
                grouped["mean"] * grouped["count"] + self.global_mean_ * self.smoothing
            ) / (grouped["count"] + self.smoothing)
            self.feature_maps_[column_name] = smoothed.to_dict()
        return self

    def transform(self, X: object) -> np.ndarray:
        data_frame: pd.DataFrame = self._to_frame(X)
        encoded_columns: list[np.ndarray] = []
        for column_name in data_frame.columns:
            feature_map = self.feature_maps_.get(column_name, {})
            encoded = data_frame[column_name].map(feature_map).fillna(self.global_mean_)
            encoded_columns.append(encoded.to_numpy(dtype=np.float64))
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        feature_names = list(input_features) if input_features is not None else list(self.feature_maps_)
        return np.asarray([f"{feature_name}_target_mean" for feature_name in feature_names], dtype=object)

    @staticmethod
    def _to_frame(X: object) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        return pd.DataFrame(X)


def _build_review_recency_features(model_data_frame: pd.DataFrame) -> pd.DataFrame:
    review_dates: pd.Series = pd.to_datetime(model_data_frame["last_review"], errors="coerce")
    latest_review_date = review_dates.max()
    fallback_days = 3650
    if pd.isna(latest_review_date):
        model_data_frame["days_since_last_review"] = fallback_days
    else:
        days_since_review = (latest_review_date - review_dates).dt.days
        model_data_frame["days_since_last_review"] = days_since_review.fillna(fallback_days).clip(lower=0)
    model_data_frame["has_reviews"] = (model_data_frame["number_of_reviews"] > 0).astype(int)
    model_data_frame["review_recency_bucket"] = pd.cut(
        model_data_frame["days_since_last_review"],
        bins=[-1, 30, 180, 365, fallback_days + 1],
        labels=["last_30_days", "last_6_months", "last_year", "older_or_none"],
    ).astype(str)
    return model_data_frame


def add_engineered_features(model_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe listing features derived only from listing attributes."""
    model_data_frame = _build_review_recency_features(model_data_frame)
    model_data_frame["log_minimum_nights"] = np.log1p(model_data_frame["minimum_nights"])
    model_data_frame["is_long_stay"] = (model_data_frame["minimum_nights"] >= 30).astype(int)
    model_data_frame["minimum_nights_bucket"] = pd.cut(
        model_data_frame["minimum_nights"],
        bins=[0, 1, 3, 7, 30, np.inf],
        labels=["one_night", "two_to_three", "four_to_seven", "eight_to_thirty", "over_thirty"],
    ).astype(str)
    model_data_frame["availability_bucket"] = pd.cut(
        model_data_frame["availability_365"],
        bins=[-1, 0, 30, 180, 365],
        labels=["unavailable", "low", "seasonal", "high"],
    ).astype(str)
    model_data_frame["log_host_listing_count"] = np.log1p(model_data_frame["calculated_host_listings_count"])
    model_data_frame["host_listing_count_bucket"] = pd.cut(
        model_data_frame["calculated_host_listings_count"],
        bins=[0, 1, 5, 20, np.inf],
        labels=["single_listing", "small_portfolio", "medium_portfolio", "large_portfolio"],
    ).astype(str)
    model_data_frame["geo_cell"] = (
        model_data_frame["latitude"].round(2).astype(str) + "_" + model_data_frame["longitude"].round(2).astype(str)
    )
    model_data_frame["room_type_neighbourhood_group"] = (
        model_data_frame["room_type"].astype(str) + "__" + model_data_frame["neighbourhood_group"].astype(str)
    )
    model_data_frame["room_type_availability_bucket"] = (
        model_data_frame["room_type"].astype(str) + "__" + model_data_frame["availability_bucket"].astype(str)
    )
    return model_data_frame


def clean_listings_dataframe(raw_data_frame: pd.DataFrame) -> pd.DataFrame:
    """Replicate cleaning from Linear Train.ipynb and DSB EDA."""
    model_data_frame: pd.DataFrame = raw_data_frame.copy()
    # Normalize key nullable review fields before row-level filtering.
    model_data_frame["reviews_per_month"] = model_data_frame["reviews_per_month"].fillna(0)
    model_data_frame["last_review"] = model_data_frame["last_review"].fillna("No reviews")
    model_data_frame = model_data_frame.dropna(subset=["name", "host_name"])
    model_data_frame = model_data_frame[model_data_frame["price"] > 0].copy()
    # Clip only high-end price outliers before feature preparation.
    first_quartile: float = float(model_data_frame["price"].quantile(0.25))
    third_quartile: float = float(model_data_frame["price"].quantile(0.75))
    interquartile_range: float = third_quartile - first_quartile
    upper_bound: float = third_quartile + 1.5 * interquartile_range
    model_data_frame = model_data_frame[model_data_frame["price"] <= upper_bound].copy()
    model_data_frame = add_engineered_features(model_data_frame)
    existing_columns_to_drop: list[str] = [
        column_name for column_name in COLUMNS_TO_DROP if column_name in model_data_frame.columns
    ]
    # Drop identifier-heavy columns before broad numeric imputations.
    model_data_frame = model_data_frame.drop(columns=existing_columns_to_drop)
    for column_name in model_data_frame.columns:
        if column_name == "price":
            continue
        # Handle pandas string/categorical extension dtypes safely during numeric imputation.
        if not is_numeric_dtype(model_data_frame[column_name]):
            continue
        median_value: float = float(model_data_frame[column_name].median())
        model_data_frame[column_name] = model_data_frame[column_name].fillna(median_value)
    return model_data_frame


def load_train_test(
    csv_path: str | None = None,
    random_state: int = config.RANDOM_STATE,
    test_size: float = config.TEST_SIZE,
) -> TrainTestData:
    """Load CSV, clean, split, and return features with log1p(price) target."""
    path: str = str(csv_path) if csv_path is not None else str(config.DATA_PATH)
    data_frame: pd.DataFrame = pd.read_csv(path)
    cleaned_frame: pd.DataFrame = clean_listings_dataframe(data_frame)
    # Build log target before split so all folds/splits share the same transform.
    target_series: pd.Series = np.log1p(cleaned_frame["price"])
    feature_frame: pd.DataFrame = cleaned_frame.drop(columns=["price"])
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target_series,
        test_size=test_size,
        random_state=random_state,
    )
    return TrainTestData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def build_column_transformer() -> ColumnTransformer:
    """One-hot categoricals, target-encode high-cardinality groups, and pass numerics."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers: list = [
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ("target_mean", TargetMeanEncoder(), TARGET_ENCODED_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
