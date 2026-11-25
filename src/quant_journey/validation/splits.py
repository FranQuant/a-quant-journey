"""Chronological partitions for one-session forward targets."""

import pandas as pd


def split_chronologically(
    dataset: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by feature date and purge labels crossing boundaries."""
    if not 0 < train_fraction < 1:
        raise ValueError("Invalid training fraction")
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("No test allocation")
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if not dataset.index.is_unique or not dataset.index.is_monotonic_increasing:
        raise ValueError("Invalid chronological index")
    if dataset.isna().any().any():
        raise ValueError("Missing dataset values")

    n = len(dataset)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))

    if train_end < 2 or validation_end - train_end < 2:
        raise ValueError("Insufficient partition size")
    if n - validation_end < 1:
        raise ValueError("Insufficient test size")

    train = dataset.iloc[:train_end - 1].copy()
    validation = dataset.iloc[train_end:validation_end - 1].copy()
    test = dataset.iloc[validation_end:].copy()

    return train, validation, test
