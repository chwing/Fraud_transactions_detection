"""Model definitions: each function returns an untrained scikit-learn Pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.features import CARD_FEATURES, TRANSACTION_FEATURES, USER_FEATURES

# Baseline inputs = group A (TRANSACTION_FEATURES), split by how each column is prepared.
# hour and day_of_week are one-hot encoded: fraud peaks at BOTH ends of the 0-23 range,
# which a linear model cannot learn from a single number.
BASELINE_ONE_HOT = ["category", "hour", "day_of_week"]
BASELINE_SCALED = ["amt"]
BASELINE_AS_IS = ["is_online"]  # already 0/1


def make_baseline() -> Pipeline:
    """Logistic regression on the transaction features (the score XGBoost must beat)."""
    preprocess = ColumnTransformer([
        # handle_unknown="ignore": a value never seen in training becomes all zeros, not a crash
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), BASELINE_ONE_HOT),
        ("scale", StandardScaler(), BASELINE_SCALED),
        ("as_is", "passthrough", BASELINE_AS_IS),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        # balanced: each fraud weighs ~170x more, so the model does not just learn "always 0"
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


# XGBoost inputs = groups A + B (+ C, currently empty). Only category needs encoding: trees split on raw numbers,
# so no scaling, and missing values (a card's first transaction) are handled natively.
# Dropped after the ablation (no PR-AUC change over 3 seeds): is_online is derived from
# category, and day_of_week carries almost no signal. The baseline still uses both.
XGB_EXCLUDED = ["day_of_week", "is_online"]
XGB_FEATURES = [c for c in TRANSACTION_FEATURES + CARD_FEATURES + USER_FEATURES if c not in XGB_EXCLUDED]


def make_xgboost(scale_pos_weight: float, features: list[str] = XGB_FEATURES) -> Pipeline:
    """Gradient-boosted trees on `features` (default: XGB_FEATURES).

    scale_pos_weight = legitimate count / fraud count in the TRAINING data (~170 here):
    the tree equivalent of class_weight="balanced".
    """
    preprocess = ColumnTransformer([
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), [c for c in features if c == "category"]),
        ("as_is", "passthrough", [c for c in features if c != "category"]),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", n_jobs=-1, random_state=0,
        )),
    ])
