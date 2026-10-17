"""Train the logistic regression baseline on group A features and evaluate it on validation."""
import pandas as pd

from fraud_detection.config import FEATURES_FILE
from fraud_detection.evaluate import evaluate
from fraud_detection.features import LABEL, TRANSACTION_FEATURES
from fraud_detection.models import make_baseline
from fraud_detection.split import time_split


def main() -> None:
    features = pd.read_parquet(FEATURES_FILE)
    train, val = time_split(features)

    model = make_baseline()
    model.fit(train[TRANSACTION_FEATURES], train[LABEL])

    scores = model.predict_proba(val[TRANSACTION_FEATURES])[:, 1]
    for name, value in evaluate(val[LABEL], scores).items():
        print(f"{name:>9}: {value:.4f}")


if __name__ == "__main__":
    main()
