"""Load the raw Sparkov data and split it into users and transactions tables."""
from pathlib import Path

import pandas as pd

from fraud_detection.config import TRAIN_RAW_FILE

USER_COLUMNS = [
    "cc_num", "first", "last", "gender", "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob",
]

TRANSACTION_COLUMNS = [
    "trans_num", "cc_num", "trans_date_trans_time", "category", "merchant",
    "amt", "merch_lat", "merch_long", "is_fraud",
]


def load_raw(path: Path = TRAIN_RAW_FILE) -> pd.DataFrame:
    """Read a raw Sparkov CSV with proper dtypes, dropping unusable columns."""
    df = pd.read_csv(path, index_col=0, parse_dates=["trans_date_trans_time", "dob"])
    # unix_time is offset by ~7 years from trans_date_trans_time: redundant and unreliable
    return df.drop(columns=["unix_time"])


def split_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw rows into (users, transactions), validating both primary keys."""
    users = df[USER_COLUMNS].drop_duplicates().reset_index(drop=True)
    if not users["cc_num"].is_unique:
        raise ValueError("Some cc_num values map to more than one user profile")

    transactions = (
        df[TRANSACTION_COLUMNS]
        # trans_num breaks ties (same timestamp), so the order is always the same
        .sort_values(["trans_date_trans_time", "trans_num"])
        .reset_index(drop=True)
    )
    if not transactions["trans_num"].is_unique:
        raise ValueError("Duplicate trans_num values found")

    return users, transactions
