"""Aligned supervised dataset for SPY research."""

import pandas as pd

from .causal import make_causal_features
from .targets import make_forward_targets


def make_supervised_dataset(prices: pd.Series) -> pd.DataFrame:
    """Align close-t features with the next-session return target."""
    features = make_causal_features(prices)
    targets = make_forward_targets(prices)

    dataset = features.join(targets, how="inner")

    if dataset.empty or dataset.isna().any().any():
        raise ValueError("Invalid supervised dataset")

    return dataset
