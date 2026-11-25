"""Map hypothetical executions into persistent long/cash targets."""

import numpy as np
import pandas as pd


def make_execution_positions(
    prices: pd.Series,
    executions: pd.DataFrame,
) -> pd.Series:
    """Construct post-execution target positions on observed sessions.

    Initial position is cash. Positions persist until another
    hypothetical execution changes the target.

    Execution prices and signal timestamps are validated against
    the observed price calendar.
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
        raise ValueError("Invalid prices")

    required = {
        "signal_date",
        "execution_date",
        "execution_price",
        "target_position",
    }
    if not required.issubset(executions.columns):
        raise ValueError("Missing execution fields")

    positions = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float,
        name="target_position",
    )

    if not executions.empty:
        if not executions.index.equals(
            pd.DatetimeIndex(executions["execution_date"])
        ):
            raise ValueError("Execution index mismatch")
        if not executions.index.is_unique:
            raise ValueError("Duplicate execution dates")
        if not executions.index.is_monotonic_increasing:
            raise ValueError("Unsorted execution dates")

        for _, row in executions.iterrows():
            signal_date = pd.Timestamp(row["signal_date"])
            execution_date = pd.Timestamp(row["execution_date"])

            if signal_date not in prices.index:
                raise ValueError("Signal date outside price calendar")
            if execution_date not in prices.index:
                raise ValueError("Execution date outside price calendar")

            signal_pos = prices.index.get_loc(signal_date)
            execution_pos = prices.index.get_loc(execution_date)

            if execution_pos != signal_pos + 1:
                raise ValueError("Execution must follow signal")

            observed_price = float(prices.loc[execution_date])
            if not np.isclose(
                float(row["execution_price"]),
                observed_price,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Execution price mismatch")

            target = float(row["target_position"])
            if not np.isfinite(target) or target not in (0.0, 1.0):
                raise ValueError("Invalid target position")

            positions.loc[execution_date] = target

    return positions.ffill().fillna(0.0).astype(int)
