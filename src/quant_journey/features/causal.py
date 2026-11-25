"""Causal features based on historical SPY closes."""

import numpy as np
import pandas as pd


def make_causal_features(prices: pd.Series) -> pd.DataFrame:
    """Compute features available after each session's close."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Invalid chronological index")
    if prices.empty or prices.isna().any():
        raise ValueError("Missing or empty price series")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive")

    daily_returns = prices.pct_change()

    return pd.DataFrame({
        "return_1d": daily_returns,
        "momentum_5d": prices.pct_change(5),
        "volatility_5d": daily_returns.rolling(5).std(),
    }).dropna()
