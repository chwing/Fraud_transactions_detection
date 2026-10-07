"""Build the processed users and transactions tables from the raw training CSV."""
from fraud_detection.config import PROCESSED_DATA_DIR, TRANSACTIONS_FILE, USERS_FILE
from fraud_detection.data import load_raw, split_tables


def main() -> None:
    raw = load_raw()
    users, transactions = split_tables(raw)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    users.to_parquet(USERS_FILE, index=False)
    transactions.to_parquet(TRANSACTIONS_FILE, index=False)

    print(f"users:        {len(users):>9,} rows -> {USERS_FILE}")
    print(f"transactions: {len(transactions):>9,} rows -> {TRANSACTIONS_FILE}")


if __name__ == "__main__":
    main()
