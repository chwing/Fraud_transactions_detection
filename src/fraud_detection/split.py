"""Chronological train / validation split (never shuffled: train on the past, validate on the future)."""
import pandas as pd

from fraud_detection.features import TIME


def time_split(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` at a cut-off time: the earliest rows go to train, the latest to validation.

    The cut-off is the timestamp of the row at position (1 - val_fraction) in time order.
    Rows AT the cut-off time all go to validation, so no timestamp is on both sides.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    ordered = df.sort_values(TIME, kind="stable").reset_index(drop=True)
    cutoff = ordered[TIME].iloc[int(len(ordered) * (1 - val_fraction))]

    is_train = ordered[TIME] < cutoff
    return ordered[is_train].reset_index(drop=True), ordered[~is_train].reset_index(drop=True)
