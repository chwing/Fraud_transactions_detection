"""Train XGBoost on the default features (XGB_FEATURES) and evaluate it on validation."""
import pandas as pd

from fraud_detection.config import FEATURES_FILE
from fraud_detection.evaluate import evaluate, threshold_for_recall
from fraud_detection.features import LABEL
from fraud_detection.models import XGB_FEATURES, make_xgboost
from fraud_detection.split import time_split


def main() -> None:
    features = pd.read_parquet(FEATURES_FILE)
    train, val = time_split(features)

    # legitimate / fraud ratio, from TRAIN only
    frauds = train[LABEL].sum()
    scale_pos_weight = (len(train) - frauds) / frauds

    model = make_xgboost(scale_pos_weight)
    model.fit(train[XGB_FEATURES], train[LABEL])

    scores = model.predict_proba(val[XGB_FEATURES])[:, 1]
    threshold = threshold_for_recall(val[LABEL], scores, target_recall=0.9)
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")
    print(f"threshold (recall >= 0.90): {threshold:.4f}")
    for name, value in evaluate(val[LABEL], scores, threshold).items():
        print(f"{name:>9}: {value:.4f}")


if __name__ == "__main__":
    main()
