"""Central project paths and settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_RAW_FILE = RAW_DATA_DIR / "fraudTrain.csv"

USERS_FILE = PROCESSED_DATA_DIR / "users.parquet"
TRANSACTIONS_FILE = PROCESSED_DATA_DIR / "transactions.parquet"
FEATURES_FILE = PROCESSED_DATA_DIR / "features.parquet"
