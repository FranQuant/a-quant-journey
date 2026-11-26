"""Validate the teaching handoff and simulate independent long/cash portfolios."""
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from ..data.tpq import load_spy_prices_through
from ..execution.dataset import make_execution_dataset
from ..validation.label_aware import split_train_validation_by_dates

from ..execution.prediction_handoff import (
    BENCHMARK, CUTOFF, INITIAL_CASH, RESERVED_START, SCENARIOS, VALIDATION_START,
    read_predictions,
)

SHARE_FIELDS = {"shares", "signed_shares_traded", "net_return", "turnover"}


def load_handoff(directory, data_path):
    prices = load_spy_prices_through(data_path, CUTOFF)
    train, validation = split_train_validation_by_dates(
        make_execution_dataset(prices), VALIDATION_START, RESERVED_START)
    frame = read_predictions(directory, train, validation)
    frame[BENCHMARK] = 1
    return prices.loc[frame.index[0]:validation.label_available_at.max()], frame


def simulate(prices, signal, bps):
    """Return native engine balances, dated instructions and executed flows."""
    state, _ = simulate_with_portfolio(prices, signal, bps)
    return state


def simulate_with_portfolio(prices, signal, bps):
    """Return the validated dated state and the native portfolio that produced it."""
    for index in (prices.index, signal.index):
        if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
                or not index.is_unique or not index.is_monotonic_increasing):
            raise ValueError("Invalid calendar")
    if (prices.empty or prices.index[-1] > pd.Timestamp(CUTOFF)
            or not np.isfinite(prices).all() or not (prices > 0).all()
            or not signal.isin([0, 1]).all() or bps not in SCENARIOS.values()):
        raise ValueError("Invalid prices, cutoff, signal or cost")
    loc = prices.index.get_indexer(signal.index)
    if (loc < 0).any() or (loc + 2 >= len(prices)).any():
        raise ValueError("Signal requires an observed close, execution and valuation")
    dates = prices.index[loc + 1]
    state = pd.DataFrame(index=prices.index)
    state["signal_date"] = pd.Series(signal.index, index=dates)
    state["target_position"] = pd.Series(signal.to_numpy(), index=dates).reindex(
        prices.index).ffill().fillna(0).astype(int)
    delta = state.target_position.diff().fillna(state.target_position)
    orders = delta.where(delta != 0)
    pf = vbt.Portfolio.from_orders(
        close=prices, price=prices, size=orders, size_type="percent", direction="longonly",
        init_cash=INITIAL_CASH, fees=bps / 10000, fixed_fees=0., slippage=0., cash_sharing=False,
        min_size=0., max_size=np.inf, size_granularity=np.nan, reject_prob=0.,
        allow_partial=False, raise_reject=True, log=True)
    fills = pf.orders.records
    expected = np.flatnonzero(orders.notna())
    if not np.array_equal(fills.idx.to_numpy(), expected):
        raise ValueError(f"Unexpected fill rows: expected {expected}, received {fills.idx.to_numpy()}")
    signed = fills["size"].to_numpy() * np.where(fills.side == 0, 1., -1.)
    state["cash"], state["shares"], state["equity"] = pf.cash(), pf.assets(), pf.value()
    opening_cash = state.cash.shift(fill_value=INITIAL_CASH)
    opening_shares = state.shares.shift(fill_value=0.)
    expected_qty = np.where(delta > 0, opening_cash / (prices * (1 + bps / 10000)),
                            -opening_shares)[expected]
    bad = ~np.isclose(signed, expected_qty, rtol=1e-12, atol=1e-12)
    if bad.any():
        raise ValueError(f"Incorrect signed fill quantity at {prices.index[expected[bad][0]]}")
    state["execution_count"] = orders.notna().astype(int)
    for field, values in [("signed_shares_traded", signed), ("transaction_cost", fills.fees),
                          ("trade_notional", fills["size"] * fills.price), ("fill_price", fills.price)]:
        state[field] = np.nan if field == "fill_price" else 0.
        state.loc[prices.index[expected], field] = np.asarray(values)
    state["pre_trade_equity"] = opening_cash + opening_shares * prices
    state["turnover"] = state.trade_notional / state.pre_trade_equity
    state["net_return"] = state.equity / state.equity.shift(fill_value=INITIAL_CASH) - 1
    return state, pf


def compare(actual, expected, exact=()):
    """Report absolute and relative errors. Raise on the earliest failing row."""
    pd.testing.assert_index_equal(actual.index, expected.index, check_names=False)
    errors, failures = {}, []
    for field in expected:
        a, e = actual[field], expected[field]
        if field in exact:
            ok = a.eq(e) | (a.isna() & e.isna())
            errors[field] = (0., 0.)
        else:
            atol = 1e-12 if field in SHARE_FIELDS or field in {"return", "max_drawdown"} else 1e-9
            ok = np.isclose(a, e, atol=atol, rtol=1e-12, equal_nan=True)
            diff = (a - e).abs()
            relative = diff / e.abs().replace(0, np.nan)
            errors[field] = (diff.fillna(0).max(), relative.fillna(0).max())
        bad = np.flatnonzero(~ok)
        if len(bad):
            i = bad[0]
            failures.append((i, field, a.iloc[i], e.iloc[i]))
    if failures:
        i, field, got, want = min(failures)
        raise AssertionError(f"{actual.index[i]} {field}: engine={got}, expected={want}")
    return pd.DataFrame(errors, index=["max_abs", "max_rel_nonzero"]).T


def reconcile(directory, prices, signals, names):
    """Oracle-only comparison after simulation, including same-run saved artifacts."""
    from ..execution.signals import map_signals_to_executions
    from ..execution.positions import make_execution_positions
    from .ledger import run_long_cash_ledger

    directory = Path(directory)
    saved = pd.read_csv(directory / "portfolio_equity_curves.csv", header=[0, 1],
                        index_col=0, parse_dates=True)
    summary = pd.read_csv(directory / "portfolio_summary.csv").set_index(["portfolio", "scenario"])
    states, discrepancies, rows = {}, {}, []
    pd.testing.assert_index_equal(saved.index, prices.index, check_names=False)
    for name in names:
        for scenario, bps in SCENARIOS.items():
            key = (name, scenario)
            state = simulate(prices, signals[name], bps)
            executions = map_signals_to_executions(prices, signals[name])
            positions = make_execution_positions(prices, executions)
            oracle = run_long_cash_ledger(prices, positions, cost_bps=bps).drop(columns="price")
            oracle["signal_date"] = executions.signal_date
            oracle["fill_price"] = prices.where(oracle.execution_count > 0)
            oracle["turnover"] = oracle.trade_notional / oracle.pre_trade_equity
            audit = state.copy()
            audit["saved_equity"] = state.equity
            oracle["saved_equity"] = saved[key]
            discrepancies[key] = compare(audit, oracle, {"signal_date","target_position","execution_count"})
            metrics = {"ending_equity": state.equity.iloc[-1],
                       "return": state.equity.iloc[-1] / INITIAL_CASH - 1,
                       "max_drawdown": (state.equity / state.equity.cummax() - 1).min(),
                       "trades": int(state.execution_count.sum())}
            end = pd.DataFrame([metrics], index=prices.index[-1:])
            reference = end.copy()
            reference.iloc[0] = summary.loc[key, list(metrics)].to_numpy()
            discrepancies[key] = pd.concat([discrepancies[key], compare(end, reference, {"trades"})])
            states[key] = state
            rows.append(dict(portfolio=name, scenario=scenario, **metrics,
                             fees=state.transaction_cost.sum(), turnover=state.turnover.sum()))
    return states, pd.DataFrame(rows), pd.concat(discrepancies, names=["portfolio","scenario","field"])
