"""Synthetic end-to-end signal, execution, and accounting tests."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.execution.signals import (
    map_signals_to_executions,
)
from src.quant_journey.execution.positions import (
    make_execution_positions,
)
from src.quant_journey.backtesting.ledger import (
    run_long_cash_ledger,
)


class TestSignalLedgerIntegration(unittest.TestCase):

    def test_always_long_and_matched_buy_hold_reconcile(self):
        dates = pd.to_datetime([
            "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07",
        ])
        prices = pd.Series([324.87, 322.41, 323.64, 325.22], index=dates)
        signals = pd.Series(1, index=dates[:2], dtype=int)

        executions = map_signals_to_executions(prices, signals)
        model_positions = make_execution_positions(prices, executions)
        # Independent buy-and-hold reference: cash until the mapped entry
        # close, then hold through terminal valuation without liquidation.
        benchmark_positions = pd.Series(
            [0, 1, 1, 1], index=dates, dtype=int, name="target_position"
        )

        self.assertEqual(executions["signal_date"].tolist(), dates[:2].tolist())
        self.assertEqual(executions["execution_date"].tolist(), dates[1:3].tolist())
        self.assertEqual(executions["execution_price"].tolist(), prices.iloc[1:3].tolist())
        pd.testing.assert_series_equal(model_positions, benchmark_positions)

        initial_cash = 1000.0
        entry_price = prices.iloc[1]
        for cost_bps in (0.0, 5.0):
            with self.subTest(cost_bps=cost_bps):
                rate = cost_bps / 10_000
                shares = initial_cash / (entry_price * (1 + rate))
                entry_fee = shares * entry_price * rate
                ledger = run_long_cash_ledger(
                    prices, model_positions, initial_cash=initial_cash, cost_bps=cost_bps
                )
                pd.testing.assert_index_equal(ledger.index, dates, check_names=False)
                self.assertEqual(ledger["execution_count"].tolist(), [0, 1, 0, 0])
                expected = {
                    "cash": [initial_cash, 0, 0, 0],
                    "shares": [0, shares, shares, shares],
                    "transaction_cost": [0, entry_fee, 0, 0],
                    "equity": [initial_cash, *(shares * prices.iloc[1:])],
                }
                # Allow float64 roundoff around zero and on the $1,000 account.
                for column, values in expected.items():
                    np.testing.assert_allclose(
                        ledger[column], values, rtol=1e-12, atol=1e-12, err_msg=column
                    )

    def test_timing_and_financial_conservation(self):
        dates = pd.to_datetime([
            "2023-01-02",
            "2023-01-03",
            "2023-01-05",
            "2023-01-06",
            "2023-01-09",
        ])

        prices = pd.Series(
            [100., 102., 101., 103., 105.],
            index=dates,
        )
        signals = pd.Series(
            [1, 0],
            index=dates[[0, 2]],
        )

        executions = map_signals_to_executions(
            prices, signals
        )
        positions = make_execution_positions(
            prices, executions
        )
        ledger = run_long_cash_ledger(
            prices,
            positions,
            initial_cash=1000.,
            cost_bps=5.,
        )

        self.assertEqual(
            positions.tolist(), [0, 1, 1, 0, 0]
        )
        self.assertEqual(
            executions["execution_date"].tolist(),
            [dates[1], dates[3]],
        )
        self.assertTrue(
            (executions["signal_date"]
             < executions["execution_date"]).all()
        )

        # No exposure on the signal-generation date.
        self.assertEqual(ledger.iloc[0]["shares"], 0.)
        self.assertEqual(ledger.iloc[0]["equity"], 1000.)

        # Purchase at the next observed close.
        shares = 1000. / (102. * 1.0005)
        self.assertAlmostEqual(
            ledger.iloc[1]["shares"], shares
        )

        # Sale after the second signal, not before.
        self.assertAlmostEqual(
            ledger.iloc[2]["shares"], shares
        )
        self.assertEqual(ledger.iloc[3]["shares"], 0.)

        expected_final = shares * 103. * 0.9995
        self.assertAlmostEqual(
            ledger.iloc[-1]["equity"], expected_final
        )

        np.testing.assert_allclose(
            ledger["equity"],
            ledger["cash"] + ledger["shares"] * ledger["price"],
        )
        np.testing.assert_allclose(
            ledger["pre_trade_equity"]
            - ledger["transaction_cost"],
            ledger["equity"],
            atol=1e-9,
        )

    def test_no_signals_no_trades(self):
        dates = pd.bdate_range("2023-01-02", periods=4)
        prices = pd.Series(
            [100., 105., 95., 110.], index=dates
        )
        signals = pd.Series(
            dtype=int, index=pd.DatetimeIndex([])
        )

        executions = map_signals_to_executions(
            prices, signals
        )
        positions = make_execution_positions(
            prices, executions
        )
        ledger = run_long_cash_ledger(
            prices, positions
        )

        self.assertTrue(executions.empty)
        self.assertEqual(positions.tolist(), [0, 0, 0, 0])
        np.testing.assert_allclose(
            ledger["equity"], 1000.
        )
        self.assertEqual(
            ledger["execution_count"].sum(), 0
        )


if __name__ == "__main__":
    unittest.main()
