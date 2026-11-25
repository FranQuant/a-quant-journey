"""Synthetic regression tests for self-financing portfolio accounting."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.backtesting.ledger import run_long_cash_ledger


class TestLongCashLedger(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=4)
        self.prices = pd.Series(
            [100.0, 102.0, 101.0, 103.0],
            index=self.dates,
        )

    def test_buy_hold_sell(self):
        positions = pd.Series([1, 1, 0, 0], index=self.dates)

        result = run_long_cash_ledger(
            self.prices, positions,
            initial_cash=1000.0, cost_bps=5.0,
        )

        shares = 1000.0 / (100.0 * 1.0005)
        final_cash = shares * 101.0 * 0.9995

        self.assertAlmostEqual(result.iloc[0]["shares"], shares)
        self.assertAlmostEqual(result.iloc[0]["cash"], 0.0)
        self.assertAlmostEqual(result.iloc[1]["shares"], shares)
        self.assertAlmostEqual(result.iloc[2]["shares"], 0.0)
        self.assertAlmostEqual(result.iloc[2]["cash"], final_cash)
        self.assertAlmostEqual(result.iloc[3]["equity"], final_cash)

        np.testing.assert_allclose(
            result["equity"],
            result["cash"] + result["shares"] * result["price"],
        )

    def test_all_cash_has_no_cost(self):
        positions = pd.Series([0, 0, 0, 0], index=self.dates)
        result = run_long_cash_ledger(self.prices, positions)

        np.testing.assert_allclose(result["equity"], 1000.0)
        np.testing.assert_allclose(result["transaction_cost"], 0.0)

    def test_terminal_liquidation(self):
        positions = pd.Series([1, 1, 1, 1], index=self.dates)
        result = run_long_cash_ledger(
            self.prices, positions, liquidate_final=True,
        )

        self.assertEqual(result.iloc[-1]["shares"], 0.0)
        self.assertAlmostEqual(
            result.iloc[-1]["equity"],
            result.iloc[-1]["cash"],
        )
        self.assertGreater(
            result.iloc[-1]["transaction_cost"], 0.0,
        )

    def test_invalid_position_rejected(self):
        positions = pd.Series([0, 1, -1, 0], index=self.dates)

        with self.assertRaises(ValueError):
            run_long_cash_ledger(self.prices, positions)

    def test_single_date_terminal_round_trip(self):
        date = self.dates[:1]
        prices = pd.Series([100.0], index=date)
        positions = pd.Series([1], index=date)

        result = run_long_cash_ledger(
            prices, positions,
            initial_cash=1000.0,
            cost_bps=5.0,
            liquidate_final=True,
        )

        expected = 1000.0 * (1 - 0.0005) / (1 + 0.0005)

        self.assertAlmostEqual(
            result.iloc[0]["equity"], expected,
        )
        self.assertEqual(result.iloc[0]["shares"], 0.0)
        self.assertAlmostEqual(
            result.iloc[0]["transaction_cost"],
            1000.0 - expected,
        )


if __name__ == "__main__":
    unittest.main()
