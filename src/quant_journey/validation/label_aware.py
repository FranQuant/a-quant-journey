"""Chronological splitting with explicit label-availability guards."""

import pandas as pd


def split_train_validation_by_dates(
    dataset: pd.DataFrame,
    validation_start: str | pd.Timestamp,
    reserved_test_start: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split only approved train/validation labels, without a test partition.

    The supplied dataset must end before the reserved test feature date. Both
    partitions apply the strict label-maturity purges used by A6.
    """
    if "label_available_at" not in dataset.columns:
        raise ValueError("Missing label availability metadata")
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise ValueError("Expected feature-date DatetimeIndex")
    if (dataset.index.hasnans or not dataset.index.is_unique
            or not dataset.index.is_monotonic_increasing):
        raise ValueError("Invalid feature-date index")
    validation_start = pd.Timestamp(validation_start)
    reserved_test_start = pd.Timestamp(reserved_test_start)
    if (pd.isna(validation_start) or pd.isna(reserved_test_start)
            or validation_start >= reserved_test_start):
        raise ValueError("Invalid partition boundaries")
    if len(dataset) == 0 or dataset.index.max() >= reserved_test_start:
        raise ValueError("Dataset reaches the reserved test partition")
    availability = dataset["label_available_at"]
    if (not pd.api.types.is_datetime64_any_dtype(availability)
            or availability.isna().any()
            or not (availability > dataset.index).all()):
        raise ValueError("Invalid label availability dates")

    train = dataset.loc[dataset.index < validation_start].copy()
    validation = dataset.loc[
        (dataset.index >= validation_start)
        & (dataset.index < reserved_test_start)
    ].copy()
    train = train.loc[train["label_available_at"] < validation_start]
    validation = validation.loc[
        validation["label_available_at"] < reserved_test_start
    ]
    if train.empty or validation.empty:
        raise ValueError("Empty train or validation partition after label purging")
    return train, validation


def split_by_label_availability(
    dataset: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition by feature date and purge unavailable boundary labels."""
    if not 0 < train_fraction < 1:
        raise ValueError("Invalid training fraction")
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("No test allocation")

    if "label_available_at" not in dataset.columns:
        raise ValueError("Missing label availability metadata")
    if not isinstance(dataset.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if (
        dataset.index.hasnans
        or not dataset.index.is_unique
        or not dataset.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid feature-date index")

    availability = dataset["label_available_at"]
    if not pd.api.types.is_datetime64_any_dtype(availability):
        raise ValueError("Label availability must be datetime")
    if availability.isna().any():
        raise ValueError("Missing label availability dates")
    if not (availability > dataset.index).all():
        raise ValueError("Labels must mature after their feature dates")

    n = len(dataset)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))

    if not (0 < train_end < validation_end < n):
        raise ValueError("Empty chronological partition")

    train_start_next = dataset.index[train_end]
    validation_start_next = dataset.index[validation_end]

    train = dataset.iloc[:train_end]
    validation = dataset.iloc[train_end:validation_end]
    test = dataset.iloc[validation_end:]

    train = train.loc[
        train["label_available_at"] < train_start_next
    ].copy()
    validation = validation.loc[
        validation["label_available_at"] < validation_start_next
    ].copy()
    test = test.copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Empty partition after label purging")

    return train, validation, test
