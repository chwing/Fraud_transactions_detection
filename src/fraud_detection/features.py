"""Feature engineering shared by offline training and real-time scoring."""
import pandas as pd

TIME = "trans_date_trans_time"

# Group A: computed from the transaction row alone
TRANSACTION_FEATURES = ["amt", "category", "hour", "day_of_week", "is_online"]
# Group B: computed from the card's PAST transactions only (never the current or future ones)
CARD_FEATURES = ["hours_since_prev_tx", "tx_count_1h", "tx_count_24h", "amt_vs_card_mean"]
# Group C: looked up in the users table
USER_FEATURES = ["city_pop"]
# Computed but excluded by default; kept only if the Step 4 ablation shows it helps
OPTIONAL_FEATURES = ["age"]

# Not features: identify the row (cc_num is kept for per-card analysis, never as model input)
ID_COLUMNS = ["trans_num", TIME, "cc_num"]
LABEL = "is_fraud"


def add_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `tx` with the group A features added."""
    out = tx.copy()
    ts = out[TIME]

    out["hour"] = ts.dt.hour  # 0-23; fraud concentrates at 22:00-03:59
    out["day_of_week"] = ts.dt.dayofweek  # Monday = 0 ... Sunday = 6
    # "_net" categories are online (card not present), "_pos" are in store
    out["is_online"] = out["category"].str.endswith("_net").astype(int)

    return out


def add_card_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `tx` with the group B (card history) features added.

    Every value describes the card's history BEFORE the transaction, so a row's
    features never depend on itself or on later rows. A card's first transaction
    has no history: hours_since_prev_tx and amt_vs_card_mean are NaN there.
    """
    if not tx.index.is_unique:
        raise ValueError("tx index must be unique (results are aligned on it)")

    out = tx.copy()
    # Each card's transactions in time order; "stable" keeps the input order for equal timestamps
    by_time = out.sort_values(["cc_num", TIME], kind="stable")
    by_card = by_time.groupby("cc_num", sort=False)

    # Hours since the card's previous transaction
    out["hours_since_prev_tx"] = by_card[TIME].diff().dt.total_seconds() / 3600

    # Number of the card's transactions in the previous 1h / 24h.
    # The rolling window (t - window, t] includes the current row, hence the "- 1".
    for window in ("1h", "24h"):
        counts = by_card.rolling(window, on=TIME)["amt"].count()
        # rolling(on=...) drops the original index; rows come back in by_time order
        out[f"tx_count_{window}"] = pd.Series(counts.to_numpy() - 1, index=by_time.index).astype(int)

    # Amount relative to the mean of all the card's previous amounts (expanding mean)
    past_sum = by_card["amt"].cumsum() - by_time["amt"]
    past_count = by_card.cumcount()  # 0 for the first transaction -> 0/0 = NaN
    out["amt_vs_card_mean"] = by_time["amt"] / (past_sum / past_count)

    return out


def add_user_features(tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `tx` with the group C features (and optional age) added."""
    out = tx.copy()
    profile = users.set_index("cc_num")

    unknown = ~out["cc_num"].isin(profile.index)
    if unknown.any():
        raise ValueError(f"{unknown.sum()} transactions have a cc_num missing from users")

    out["city_pop"] = out["cc_num"].map(profile["city_pop"])
    # Age in whole years at the time of the transaction
    dob = out["cc_num"].map(profile["dob"])
    out["age"] = ((out[TIME] - dob).dt.days // 365.25).astype(int)

    return out


def build_features(transactions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Run groups A -> B -> C and keep only ids, features and label, in time order."""
    tx = add_transaction_features(transactions)
    tx = add_card_features(tx)
    tx = add_user_features(tx, users)

    columns = ID_COLUMNS + TRANSACTION_FEATURES + CARD_FEATURES + USER_FEATURES + OPTIONAL_FEATURES + [LABEL]
    return tx[columns].sort_values(TIME, kind="stable").reset_index(drop=True)
