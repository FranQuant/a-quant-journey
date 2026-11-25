"""Targets for explicitly delayed, close-only execution research."""

import numpy as np
import pandas as pd


def make_execution_targets(prices: pd.Series) -> pd.DataFrame:
    """Label hypothetical close(t+1) to close(t+2) returns.

    Features are assumed available after close(t).
    The target becomes observable after close(t+2).
    This is not evidence of an achievable execution price.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if (
        not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or prices.index.hasnans
    ):
        raise ValueError("Invalid chronological index")
    if len(prices) < 3:
        raise ValueError("At least three prices required")

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")

    execution_return = prices.shift(-2) / prices.shift(-1) - 1

    return pd.DataFrame({
        "execution_target_return": execution_return.iloc[:-2],
        "execution_target_up": (
            execution_return.iloc[:-2] > 0
        ).astype(int),
        "label_available_at": prices.index[2:],
    })
