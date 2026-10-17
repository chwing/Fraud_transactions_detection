import pandas as pd
import pytest

from fraud_detection.features import (
    CARD_FEATURES,
    TRANSACTION_FEATURES,
    OPTIONAL_FEATURES,
    add_card_features,
    add_transaction_features,
    add_user_features,
    build_features,
)


def make_tx() -> pd.DataFrame:
    # Two hand-made transactions with known answers:
    # 2019-01-07 is a Monday (0), 2019-01-12 is a Saturday (5)
    return pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(["2019-01-07 23:15:00", "2019-01-12 08:00:00"]),
        "category": ["shopping_net", "grocery_pos"],
        "amt": [900.0, 20.0],
    })


def test_transaction_features_values():
    out = add_transaction_features(make_tx())

    assert out["hour"].tolist() == [23, 8]
    assert out["day_of_week"].tolist() == [0, 5]
    assert out["is_online"].tolist() == [1, 0]   # _net -> online, _pos -> in store


def test_all_declared_features_exist():
    # Every name in the feature list must really be produced
    out = add_transaction_features(make_tx())
    assert set(TRANSACTION_FEATURES) <= set(out.columns)


def test_input_is_not_modified():
    tx = make_tx()
    add_transaction_features(tx)
    assert "hour" not in tx.columns


def make_card_tx() -> pd.DataFrame:
    # Card 1: three transactions (10:00, 10:30, 12:00). Card 2: one transaction in between.
    # Rows are deliberately NOT sorted by card, to check the grouping.
    return pd.DataFrame({
        "cc_num": [1, 2, 1, 1],
        "trans_date_trans_time": pd.to_datetime(
            ["2019-01-01 10:00", "2019-01-01 10:10", "2019-01-01 10:30", "2019-01-01 12:00"]
        ),
        "amt": [10.0, 50.0, 30.0, 100.0],
    })


def test_card_features_values():
    out = add_card_features(make_card_tx())

    # Card 2's transaction is between card 1's, but must not count as card 1's history
    assert out["hours_since_prev_tx"].tolist()[2:] == [0.5, 1.5]
    assert out["tx_count_1h"].tolist() == [0, 0, 1, 0]      # 12:00 is > 1h after 10:30
    assert out["tx_count_24h"].tolist() == [0, 0, 1, 2]
    # 30 / mean(10) = 3.0 ; 100 / mean(10, 30) = 5.0
    assert out["amt_vs_card_mean"].tolist()[2:] == [3.0, 5.0]
    # First transaction of each card: no history -> NaN
    assert out.loc[[0, 1], ["hours_since_prev_tx", "amt_vs_card_mean"]].isna().all().all()
    assert set(CARD_FEATURES) <= set(out.columns)


def test_card_features_do_not_use_the_future():
    # Leakage test: changing a LATER transaction must not change an earlier row's features
    tx = make_card_tx()
    changed = tx.copy()
    changed.loc[3, "amt"] = 99_999.0                         # card 1's last transaction
    changed.loc[3, "trans_date_trans_time"] = pd.Timestamp("2019-01-01 10:31")

    before = add_card_features(tx).loc[[0, 1, 2], CARD_FEATURES]
    after = add_card_features(changed).loc[[0, 1, 2], CARD_FEATURES]
    pd.testing.assert_frame_equal(before, after)


def test_card_features_duplicate_index_raises():
    with pytest.raises(ValueError):
        add_card_features(make_card_tx().set_axis([0, 0, 1, 2]))


def make_users() -> pd.DataFrame:
    return pd.DataFrame({
        "cc_num": [1, 2],
        "city_pop": [149, 2_000_000],
        "dob": pd.to_datetime(["1978-06-21", "2000-01-01"]),
    })


def test_user_features_values():
    out = add_user_features(make_card_tx(), make_users())

    assert out["city_pop"].tolist() == [149, 2_000_000, 149, 149]
    # Card 1 born 1978-06-21 -> 40 on 2019-01-01 ; card 2 born 2000-01-01 -> 19
    assert out["age"].tolist() == [40, 19, 40, 40]
    assert set(OPTIONAL_FEATURES) <= set(out.columns)


def test_unknown_card_raises():
    users = make_users().iloc[:1]                             # card 2 missing
    with pytest.raises(ValueError):
        add_user_features(make_card_tx(), users)


def test_build_features_output():
    tx = make_card_tx()
    tx["trans_num"] = ["t1", "t2", "t3", "t4"]
    tx["category"] = "grocery_pos"
    tx["merchant"] = "fraud_Shop"            # raw column that must NOT reach the output
    tx["is_fraud"] = [0, 0, 0, 1]

    out = build_features(tx, make_users())

    assert len(out) == len(tx)               # no row lost or duplicated
    assert out.columns[-1] == "is_fraud"
    assert "merchant" not in out.columns
    assert out["trans_date_trans_time"].is_monotonic_increasing
