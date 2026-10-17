import pandas as pd
import pytest

from fraud_detection.split import time_split


def make_df() -> pd.DataFrame:
    # 10 rows, deliberately shuffled. Sorted positions 7 and 8 share the same timestamp,
    # so a plain 80% row cut (after position 7) would put one in train and one in validation.
    days = pd.to_datetime([f"2019-01-0{d}" for d in range(1, 8)] + ["2019-01-08"] * 2 + ["2019-01-09"])
    return pd.DataFrame({"trans_date_trans_time": days, "amt": range(10)}).sample(frac=1, random_state=0)


def test_train_is_strictly_before_validation():
    train, val = time_split(make_df(), val_fraction=0.2)

    assert train["trans_date_trans_time"].max() < val["trans_date_trans_time"].min()
    assert len(train) + len(val) == 10           # no row lost or duplicated


def test_proportions():
    df = pd.DataFrame({"trans_date_trans_time": pd.date_range("2019-01-01", periods=100, freq="h")})
    train, val = time_split(df, val_fraction=0.2)
    assert (len(train), len(val)) == (80, 20)


def test_equal_timestamps_stay_on_one_side():
    # The two 2019-01-08 rows straddle the 80% position: both must land in validation
    train, val = time_split(make_df(), val_fraction=0.2)
    assert (val["trans_date_trans_time"] == pd.Timestamp("2019-01-08")).sum() == 2
    assert len(train) == 7


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        time_split(make_df(), val_fraction=1.5)
