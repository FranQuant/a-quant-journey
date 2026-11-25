"""Execution-aligned supervised dataset construction."""

import numpy as np
import pandas as pd

from ..features.causal import make_causal_features
from .targets import make_execution_targets


def make_execution_dataset(prices: pd.Series) -> pd.DataFrame:
    """Join after-close-t features to the later executable-return target.

    The target measures close(t+1) to close(t+2), while all features
    are available after close(t). No same-close execution is assumed.
    """
    features = make_causal_features(prices)
    targets = make_execution_targets(prices)

    dataset = features.join(targets, how="inner")

    required = [
        "return_1d",
        "momentum_5d",
        "volatility_5d",
        "execution_target_return",
        "execution_target_up",
        "label_available_at",
    ]

    if dataset.empty or dataset[required].isna().any().any():
        raise ValueError("Invalid execution-aligned dataset")

    numeric = dataset[required[:-1]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite execution dataset")

    if not (
        dataset["label_available_at"].to_numpy()
        > dataset.index.to_numpy()
    ).all():
        raise ValueError("Execution labels must mature after features")

    return dataset
