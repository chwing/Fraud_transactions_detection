"""Unit tests for fraud_detection.data.split_tables.

Each test builds a tiny hand-made DataFrame (no real CSV needed), runs split_tables
on it, and checks a property the split must always guarantee.
"""
import pandas as pd
import pytest

from fraud_detection.data import split_tables


def make_row(cc_num: int, trans_num: str, city: str = "Orient") -> dict:
    """Build one fake raw row with every column split_tables expects.

    Only cc_num, trans_num and city vary between tests; all other values are
    constant, so two rows with the same cc_num describe the same user.
    """
    return {
        "cc_num": cc_num, "first": "Ann", "last": "Lee", "gender": "F",
        "street": "1 Main St", "city": city, "state": "WA", "zip": 99160,
        "lat": 48.9, "long": -118.2, "city_pop": 149, "job": "Teacher",
        "dob": pd.Timestamp("1978-06-21"),
        "trans_num": trans_num, "trans_date_trans_time": pd.Timestamp("2019-01-01 10:00"),
        "category": "grocery_pos", "merchant": "fraud_Shop", "amt": 10.0,
        "merch_lat": 49.0, "merch_long": -118.1, "is_fraud": 0,
    }


def test_split_keeps_all_rows_and_valid_keys():
    # Input: 3 transactions from 2 cards (card 1 has two transactions).
    raw = pd.DataFrame([make_row(1, "t1"), make_row(1, "t2"), make_row(2, "t3")])

    users, transactions = split_tables(raw)

    # Card 1's repeated profile must collapse into one user -> 2 users total.
    assert len(users) == 2
    # No transaction may be lost or duplicated by the split.
    assert len(transactions) == 3
    # Foreign key check: every transaction's cc_num points to an existing user.
    assert set(transactions["cc_num"]) <= set(users["cc_num"])


def test_inconsistent_user_profile_raises():
    # Input: the same card (cc_num=1) appears with two different cities.
    # drop_duplicates keeps both profiles, so cc_num is no longer unique in users.
    raw = pd.DataFrame([make_row(1, "t1"), make_row(1, "t2", city="Boulder")])

    # The split must refuse to guess which profile is right and raise instead.
    with pytest.raises(ValueError):
        split_tables(raw)


def test_duplicate_trans_num_raises():
    # Input: two different cards share the transaction ID "t1",
    # which breaks the transactions primary key.
    raw = pd.DataFrame([make_row(1, "t1"), make_row(2, "t1")])

    # The split must detect the duplicate key and raise.
    with pytest.raises(ValueError):
        split_tables(raw)


def test_equal_timestamps_are_ordered_by_trans_num():
    # Same card, same second: the order must not depend on the input order
    raw = pd.DataFrame([make_row(1, "t2"), make_row(1, "t1")])

    _, transactions = split_tables(raw)
    assert transactions["trans_num"].tolist() == ["t1", "t2"]
