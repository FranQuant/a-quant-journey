"""Forward targets for chronological financial research."""

import numpy as np
import pandas as pd


def make_forward_targets(prices: pd.Series) -> pd.DataFrame:
    """Label the next observed trading-session close-to-close return."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have unique, increasing dates")
    if len(prices) < 2:
        raise ValueError("At least two prices are required")
    if prices.isna().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Prices must be finite and nonmissing")
    if (prices <= 0).any():
        raise ValueError("Prices must be positive")

    forward_return = prices.shift(-1) / prices - 1

    result = pd.DataFrame({
        "forward_return": forward_return.iloc[:-1],
        "target_up": (forward_return.iloc[:-1] > 0).astype(int),
    })

    return result
