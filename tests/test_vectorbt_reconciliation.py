"""Optional parity gate. Set AQJ_ARTIFACT_DIR to the explicit 03/04 handoff."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("vectorbt")
from src.quant_journey.backtesting import vectorbt_adapter as adapter


def series(values):
    dates = pd.to_datetime(["2020-01-02","2020-01-08","2020-01-09","2020-01-17","2020-01-21"])
    return pd.Series(values, index=dates)


@pytest.mark.parametrize("bps", [0, 5])
def test_fractional_round_trip(bps):
    p = series([2000.] * 5)
    signal = pd.Series([1, 1, 0], index=p.index[:3])
    result = adapter.simulate(p, signal, bps)
    fee = bps / 10000
    qty = 1000 / (2000 * (1 + fee))
    np.testing.assert_allclose(result.shares, [0, qty, qty, 0, 0])
    np.testing.assert_allclose(result.cash, [1000, 0, 0, qty*2000*(1-fee), qty*2000*(1-fee)])
    np.testing.assert_allclose(result.transaction_cost, np.array([0,1,0,1,0])*qty*2000*fee)
    assert result.execution_count.tolist() == [0,1,0,1,0]


def test_delayed_entry_gap_and_terminal_mark():
    prices = series([50., 100., 200., 300., 400.])
    sig = pd.Series([1], index=prices.index[:1])
    result = adapter.simulate(prices, sig, 0)
    assert result.equity.tolist() == [1000,1000,2000,3000,4000]
    assert result.shares.iloc[-1] == 10
    cash = adapter.simulate(prices, sig * 0, 5)
    assert (cash.equity == 1000).all() and (cash.turnover == 0).all()
    assert (cash.cash == 1000).all() and cash.execution_count.sum() == 0


@pytest.mark.parametrize("fault", ["ignored","partial","rejected"])
def test_fill_negative_controls(monkeypatch, fault):
    original = adapter.vbt.Portfolio.from_orders
    def inject(**kwargs):
        if fault == "ignored":
            kwargs["size"] = np.nan
        elif fault == "partial":
            kwargs["size"] = kwargs["size"] * 0.5
        else:
            kwargs["min_size"] = 1e6
        return original(**kwargs)
    from vectorbt.portfolio.enums import RejectedOrderError
    monkeypatch.setattr(adapter.vbt.Portfolio, "from_orders", inject)
    p = series([100.] * 5)
    with pytest.raises(RejectedOrderError if fault == "rejected" else ValueError,
                       match={"ignored":"fill rows","partial":"fill quantity","rejected":".+"}[fault]):
        adapter.simulate(p, pd.Series([1], index=p.index[:1]), 5)


@pytest.mark.parametrize("fault", ["cutoff","last","price","unknown","duplicate","binary"])
def test_invalid_calendar(fault):
    p = series([100.] * 5)
    s = pd.Series([1], index=p.index[:1])
    if fault == "cutoff":
        p.loc["2021-12-29"] = 1
    elif fault == "last":
        s.index = p.index[-2:-1]
    elif fault == "price":
        p.iloc[2] = np.nan
    elif fault == "unknown":
        s.index = s.index + pd.Timedelta(days=1)
    elif fault == "duplicate":
        s = pd.concat([s,s])
    else:
        s.iloc[0] = -1
    with pytest.raises(ValueError):
        adapter.simulate(p,s,5)


@pytest.mark.parametrize("stage", ["pilot","all"])
def test_real_validation(stage):
    if "AQJ_ARTIFACT_DIR" not in os.environ:
        pytest.skip("Set AQJ_ARTIFACT_DIR for the explicit canonical run")
    directory = Path(os.environ["AQJ_ARTIFACT_DIR"])
    prices, signals = adapter.load_handoff(directory, "data/raw/tpq_eod_2014_2023.csv")
    assert len(prices) == 502 and len(signals) == 500
    names = ["GaussianNB", adapter.BENCHMARK] if stage == "pilot" else signals.columns
    states, _, _ = adapter.reconcile(directory, prices, signals, names)
    if stage == "all":
        assert sum(map(len,states.values())) == 9036
        for scenario in adapter.SCENARIOS:
            for name in ["Logistic regression","Training majority baseline"]:
                pd.testing.assert_series_equal(states[name,scenario].equity, states[adapter.BENCHMARK,scenario].equity)


def test_native_portfolio_wrapper():
    prices = series([100., 99., 103., 102., 101.])
    signal = pd.Series([1, 0, 1], index=prices.index[:3])
    state, pf = adapter.simulate_with_portfolio(prices, signal, 5)
    assert isinstance(pf, adapter.vbt.Portfolio)
    pd.testing.assert_frame_equal(state, adapter.simulate(prices, signal, 5), check_exact=True)


@pytest.fixture(scope="module")
def analytics():
    """Exercise the actual teaching cells on the bounded handoff with network calls denied."""
    if "AQJ_ARTIFACT_DIR" not in os.environ:
        pytest.skip("Set AQJ_ARTIFACT_DIR for the explicit canonical run")
    import socket
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def deny_connection(*args, **kwargs):
        pytest.fail("Analytics attempted network access")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", deny_connection)
        directory = Path(os.environ["AQJ_ARTIFACT_DIR"])
        prices, signals = adapter.load_handoff(directory, "data/raw/tpq_eod_2014_2023.csv")
        states, _, _ = adapter.reconcile(directory, prices, signals, ["GaussianNB", adapter.BENCHMARK])
        env = dict(prices=prices, signals=signals, states=states, pd=pd, plt=plt,
                   BENCHMARK=adapter.BENCHMARK, audit=states["GaussianNB", "illustrative_5bps"],
                   display=lambda *args, **kwargs: None)
        notebook = Path("notebooks/teaching/05_vectorbt_portfolio_reconciliation.ipynb")
        exercised = []
        for cell in json.loads(notebook.read_text())["cells"]:
            if cell["cell_type"] == "code" and cell["id"].startswith("native-"):
                exercised.append(cell["id"])
                exec(compile("".join(cell["source"]), cell["id"], "exec"), env)
        assert exercised == ["native-statistics", "native-risk", "native-table", "native-records",
                             "native-dashboard", "native-display", "native-snapshot", "native-heatmap"]
        yield env


def test_native_accounting_and_risk(analytics):
    a = analytics
    r, acc, state = a["returns"], a["acc"], a["state"]
    annual = (1 + r).prod() ** (252 / len(r)) - 1
    volatility = r.std(ddof=1) * np.sqrt(252)
    mean = r.mean() * 252
    downside = np.sqrt(252 * (r.clip(upper=0) ** 2).mean())
    dd = (state.equity / state.equity.cummax() - 1).min()
    np.testing.assert_allclose(
        [acc.annualized(), acc.annualized_volatility(), acc.sharpe_ratio(),
         acc.sortino_ratio(), acc.calmar_ratio(), a["stats"]["Max Drawdown [%]"]],
        [annual, volatility, mean / volatility, mean / downside, annual / abs(dd), -100 * dd],
        atol=1e-12, rtol=1e-12)
    assert a["matched"].cash.iloc[0] == 1000 and a["bm"].index.equals(r.index)
    assert a["matched"].execution_count.iloc[:2].tolist() == [0, 1]
    np.testing.assert_allclose(a["bm"].iloc[:2], [0, 1 / 1.0005 - 1], atol=1e-12)
    assert a["matched"].execution_count.sum() == 1
    assert a["open_positions"]["Exit Fees"].item() == 0


def test_native_dashboard_traces(analytics):
    a = analytics
    traces = {t.name: t for t in a["dashboard"].data}
    def check(name, dates, values, row):
        t = traces[name]
        assert t.yaxis == row and pd.DatetimeIndex(t.x).equals(pd.DatetimeIndex(dates))
        np.testing.assert_allclose(t.y, values, atol=1e-12, rtol=1e-12)

    prices = a["prices"]
    check("Close", prices.index, prices, "y")
    for side in ["Buy", "Sell"]:
        fills = a["orders"].loc[a["orders"].Side == side]
        check(side, fills.Timestamp, fills.Price, "y")
    trades = a["trades"]
    for name, mask in [("Closed - Profit", trades.Status.eq("Closed") & trades.PnL.gt(0)),
                       ("Closed - Loss", trades.Status.eq("Closed") & trades.PnL.lt(0)),
                       ("Open", trades.Status.eq("Open"))]:
        check(name, trades.loc[mask, "Exit Timestamp"], trades.loc[mask, "PnL"], "y2")
    for name, state in [("GaussianNB", a["state"]), ("Matched SPY", a["matched"])]:
        check(name, state.index, state.equity / 1000 - 1, "y3")


def test_native_quantstats_figures(analytics):
    # The teaching cells assert each of the 502 coordinates per panel and every heatmap value.
    a = analytics
    assert len(a["snapshot"].axes) == 3 and a["matrix"].shape == (2, 12)
    ax = a["heatmap"].axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["2020", "2021"]
    assert a["returns"].index[-1] < a["monthly"].index[-1].end_time
