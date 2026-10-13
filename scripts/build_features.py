"""Build the model-ready features table from the processed users and transactions tables."""
import pandas as pd

from fraud_detection.config import FEATURES_FILE, TRANSACTIONS_FILE, USERS_FILE
from fraud_detection.features import build_features


def main() -> None:
    transactions = pd.read_parquet(TRANSACTIONS_FILE)
    users = pd.read_parquet(USERS_FILE)

    features = build_features(transactions, users)
    features.to_parquet(FEATURES_FILE, index=False)

    print(f"features: {len(features):>9,} rows x {features.shape[1]} columns -> {FEATURES_FILE}")
    print(f"frauds:   {features['is_fraud'].sum():>9,}")


if __name__ == "__main__":
    main()
