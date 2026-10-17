"""Ablation: retrain XGBoost on different feature sets and compare them on validation."""
import pandas as pd

from fraud_detection.config import FEATURES_FILE
from fraud_detection.evaluate import evaluate
from fraud_detection.features import (
    CARD_FEATURES, LABEL, TRANSACTION_FEATURES,
)
from fraud_detection.models import make_xgboost
from fraud_detection.split import time_split

VARIANTS = {
    "A (transaction)": TRANSACTION_FEATURES,
    "A+B (default)": TRANSACTION_FEATURES + CARD_FEATURES,
    "A+B + city_pop": TRANSACTION_FEATURES + CARD_FEATURES + ["city_pop"],
    "A+B + city_pop + age": TRANSACTION_FEATURES + CARD_FEATURES + ["city_pop", "age"],
}


def main() -> None:
    features = pd.read_parquet(FEATURES_FILE)
    train, val = time_split(features)
    frauds = train[LABEL].sum()
    scale_pos_weight = (len(train) - frauds) / frauds

    rows = []
    for name, cols in VARIANTS.items():
        model = make_xgboost(scale_pos_weight, cols).fit(train[cols], train[LABEL])
        scores = model.predict_proba(val[cols])[:, 1]
        rows.append({"variant": name, **evaluate(val[LABEL], scores)})

    print(pd.DataFrame(rows).set_index("variant").round(4).to_string())


if __name__ == "__main__":
    main()
