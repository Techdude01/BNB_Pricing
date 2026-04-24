"""Load AB_NYC_2019.csv, apply notebook cleaning, and build sklearn preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pricing_lab import config

COLUMNS_TO_DROP: list[str] = ["id", "name", "host_id", "host_name", "last_review"]
CATEGORICAL_FEATURES: list[str] = ["neighbourhood_group", "neighbourhood", "room_type"]
NUMERIC_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]


@dataclass(frozen=True)
class TrainTestData:
    """Train and test tensors for log1p(price) regression."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


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
    """One-hot categoricals; pass numeric columns through unchanged."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers: list = [
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
