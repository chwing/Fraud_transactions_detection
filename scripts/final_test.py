"""Final, one-time evaluation on the held-out fraudTest.csv.

The model and threshold are chosen exactly as in train_xgboost.py (train / validation split
of fraudTrain only); the test set is used for nothing but this final score.
"""
import pandas as pd

from fraud_detection.config import TEST_RAW_FILE, TRAIN_RAW_FILE
from fraud_detection.data import load_raw, split_tables
from fraud_detection.evaluate import evaluate, threshold_for_recall
from fraud_detection.features import LABEL, build_features
from fraud_detection.models import XGB_FEATURES, make_xgboost
from fraud_detection.split import time_split


def main() -> None:
    train_raw, test_raw = load_raw(TRAIN_RAW_FILE), load_raw(TEST_RAW_FILE)

    # Features on both files together, so card history continues from train into test
    # (features only look at the past, so test rows never leak into train rows).
    # split_tables also checks that each card's profile is identical in both files.
    users, transactions = split_tables(pd.concat([train_raw, test_raw], ignore_index=True))
    features = build_features(transactions, users)

    is_test = features["trans_num"].isin(test_raw["trans_num"])
    train, val = time_split(features[~is_test])
    test = features[is_test]

    # Same model and threshold selection as train_xgboost.py
    frauds = train[LABEL].sum()
    model = make_xgboost((len(train) - frauds) / frauds).fit(train[XGB_FEATURES], train[LABEL])
    threshold = threshold_for_recall(
        val[LABEL], model.predict_proba(val[XGB_FEATURES])[:, 1], target_recall=0.9
    )

    scores = model.predict_proba(test[XGB_FEATURES])[:, 1]
    flagged = scores >= threshold
    print(f"test: {len(test):,} rows, {test[LABEL].sum():,} frauds  |  threshold {threshold:.4f}")
    for name, value in evaluate(test[LABEL], scores, threshold).items():
        print(f"{name:>9}: {value:.4f}")
    print(f"  flagged: {flagged.sum():,}  (false alarms: {(flagged & (test[LABEL] == 0)).sum():,}, "
          f"missed frauds: {(~flagged & (test[LABEL] == 1)).sum():,})")


if __name__ == "__main__":
    main()
