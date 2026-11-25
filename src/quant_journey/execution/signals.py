"""Map post-close signals to later hypothetical close executions."""

import numpy as np
import pandas as pd


def map_signals_to_executions(
    prices: pd.Series,
    signals: pd.Series,
) -> pd.DataFrame:
    """Map each signal to the next observed price session.

    Signals are assumed available after their indexed session close.
    Execution at the next observed close is hypothetical, not a fill.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected price DatetimeIndex")
    if (
        prices.empty
        or prices.index.hasnans
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid price index")

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid observed prices")

    if not isinstance(signals.index, pd.DatetimeIndex):
        raise ValueError("Expected signal DatetimeIndex")
    if (
        signals.index.hasnans
        or not signals.index.is_unique
        or not signals.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid signal index")
    if not signals.index.isin(prices.index).all():
        raise ValueError("Signals must reference observed sessions")

    signal_values = signals.to_numpy(dtype=float)
    if not np.isfinite(signal_values).all():
        raise ValueError("Nonfinite signals")
    if not np.isin(signal_values, [0.0, 1.0]).all():
        raise ValueError("Only long/cash signals supported")

    records = []

    for signal_date, position in signals.items():
        signal_loc = prices.index.get_loc(signal_date)

        if signal_loc + 1 >= len(prices):
            continue

        execution_date = prices.index[signal_loc + 1]

        records.append({
            "signal_date": signal_date,
            "execution_date": execution_date,
            "execution_price": float(prices.iloc[signal_loc + 1]),
            "target_position": int(position),
        })

    if not records:
        return pd.DataFrame(columns=[
            "signal_date",
            "execution_date",
            "execution_price",
            "target_position",
        ])

    return pd.DataFrame.from_records(records).set_index(
        "execution_date", drop=False
    )
