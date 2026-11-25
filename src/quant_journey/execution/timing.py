"""Explicit close-only forecast and execution-proxy horizons."""

import numpy as np
import pandas as pd


def make_timing_contract(prices: pd.Series) -> pd.DataFrame:
    """Align close-t forecasts with later hypothetical close executions.

    Index t:
      forecast_return: close(t) -> close(t+1)
      execution_proxy_return: close(t+1) -> close(t+2)

    Execution is hypothetical, not a verified fill.
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

    forecast = prices.shift(-1) / prices - 1
    execution = prices.shift(-2) / prices.shift(-1) - 1

    return pd.DataFrame({
        "forecast_return": forecast.iloc[:-1],
        "execution_proxy_return": execution.iloc[:-1],
    })
