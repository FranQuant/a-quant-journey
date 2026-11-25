"""Self-financing long/cash portfolio accounting."""

import numpy as np
import pandas as pd


def run_long_cash_ledger(
    prices: pd.Series,
    target_positions: pd.Series,
    initial_cash: float = 1000.0,
    cost_bps: float = 5.0,
    liquidate_final: bool = False,
) -> pd.DataFrame:
    """Execute hypothetical target positions at supplied prices.

    Position 1: fully invested in the asset.
    Position 0: entirely in cash.

    Trades occur at the price attached to each target-position date.
    This function does not establish when signals were generated.
    """

    if not prices.index.equals(target_positions.index):
        raise ValueError("Prices and positions must share an index")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if (
        prices.empty
        or prices.index.hasnans
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Invalid chronological index")

    values = prices.to_numpy(dtype=float)
    positions = target_positions.to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if not np.isfinite(positions).all():
        raise ValueError("Positions must be finite")
    if not np.isin(positions, [0.0, 1.0]).all():
        raise ValueError("Only long/cash positions are supported")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Invalid initial cash")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Invalid transaction cost")

    cost_rate = cost_bps / 10_000
    cash = float(initial_cash)
    shares = 0.0
    records = []

    for i, (date, price, target) in enumerate(
        zip(prices.index, values, positions)
    ):
        pre_trade_equity = cash + shares * price
        trade_notional = 0.0
        execution_count = 0
        signed_shares_traded = 0.0

        if target == 1.0 and shares == 0.0:
            purchased = cash / (price * (1 + cost_rate))
            trade_notional = purchased * price
            cash -= trade_notional * (1 + cost_rate)
            shares += purchased
            execution_count += 1
            signed_shares_traded += purchased

        elif target == 0.0 and shares > 0.0:
            trade_notional = shares * price
            cash += trade_notional * (1 - cost_rate)
            signed_shares_traded -= shares
            execution_count += 1
            shares = 0.0

        if liquidate_final and i == len(prices) - 1 and shares > 0.0:
            liquidation_notional = shares * price
            cash += liquidation_notional * (1 - cost_rate)
            trade_notional += liquidation_notional
            signed_shares_traded -= shares
            execution_count += 1
            shares = 0.0

        if abs(cash) < 1e-10:
            cash = 0.0

        equity = cash + shares * price

        records.append({
            "date": date,
            "price": price,
            "target_position": int(target),
            "pre_trade_equity": pre_trade_equity,
            "trade_notional": trade_notional,
            "execution_count": execution_count,
            "signed_shares_traded": signed_shares_traded,
            "transaction_cost": trade_notional * cost_rate,
            "cash": cash,
            "shares": shares,
            "equity": equity,
        })

    result = pd.DataFrame.from_records(records).set_index("date")
    result["net_return"] = (
        result["equity"].div(
            result["equity"].shift(1).fillna(initial_cash)
        ) - 1
    )

    return result
