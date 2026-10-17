import pandas as pd

from fraud_detection.features import TRANSACTION_FEATURES
from fraud_detection.models import XGB_FEATURES, make_baseline, make_xgboost


def make_train() -> pd.DataFrame:
    # Tiny training set: big online night-time amounts are fraud
    return pd.DataFrame({
        "amt": [900.0, 850.0, 20.0, 35.0, 15.0, 40.0],
        "category": ["shopping_net", "misc_net", "grocery_pos", "home", "grocery_pos", "home"],
        "hour": [23, 2, 10, 14, 9, 18],
        "day_of_week": [4, 5, 0, 1, 2, 3],
        "is_online": [1, 1, 0, 0, 0, 0],
        "is_fraud": [1, 1, 0, 0, 0, 0],
    })


def test_baseline_fits_and_predicts_probabilities():
    train = make_train()
    model = make_baseline().fit(train[TRANSACTION_FEATURES], train["is_fraud"])

    proba = model.predict_proba(train[TRANSACTION_FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba[:2].min() > proba[2:].max()     # frauds ranked above legitimate rows


def test_unknown_category_does_not_crash():
    train = make_train()
    model = make_baseline().fit(train[TRANSACTION_FEATURES], train["is_fraud"])

    new = train[TRANSACTION_FEATURES].head(1).assign(category="travel")   # never seen in training
    assert model.predict_proba(new).shape == (1, 2)


def test_xgboost_fits_with_missing_history():
    train = make_train().assign(
        hours_since_prev_tx=[0.2, 0.1, 5.0, None, 8.0, 3.0],   # None = card's first transaction
        tx_count_1h=[3, 2, 0, 0, 0, 0],
        tx_count_24h=[6, 5, 2, 0, 1, 3],
        amt_vs_card_mean=[12.0, 9.0, 0.8, None, 1.1, 0.9],
        city_pop=[149, 2000, 149, 5000, 149, 2000],
    )
    model = make_xgboost(scale_pos_weight=2.0).fit(train[XGB_FEATURES], train["is_fraud"])

    proba = model.predict_proba(train[XGB_FEATURES])[:, 1]
    assert proba[:2].min() > proba[2:].max()
