"""Project-wide constants for the pricing_lab package."""

import os
from pathlib import Path


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if value == 0 or value < -1:
        raise ValueError(f"{name} must be -1 or a positive integer, got {value}.")
    return value


RANDOM_STATE: int = 42
CV_SPLITS: int = 5
TEST_SIZE: float = 0.2
DATA_FILE_NAME: str = "AB_NYC_2019.csv"
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / DATA_FILE_NAME

N_TRIALS_ELASTICNET: int = 35
N_TRIALS_KNN: int = 30
N_TRIALS_XGBOOST: int = 45
N_TRIALS_SVM: int = 8
N_TRIALS_NEURAL_NETWORK: int = 25
N_TRIALS_RANDOM_FOREST: int = 35

OPTUNA_SAMPLER_SEED: int = RANDOM_STATE
MODEL_N_JOBS: int = _int_from_env("BNB_MODEL_N_JOBS", 6)
