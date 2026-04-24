"""Project-wide constants for the pricing_lab package."""

from pathlib import Path

RANDOM_STATE: int = 42
CV_SPLITS: int = 5
TEST_SIZE: float = 0.2
DATA_FILE_NAME: str = "AB_NYC_2019.csv"
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / DATA_FILE_NAME

N_TRIALS_ELASTICNET: int = 35
N_TRIALS_KNN: int = 30
N_TRIALS_XGBOOST: int = 45

OPTUNA_SAMPLER_SEED: int = RANDOM_STATE
