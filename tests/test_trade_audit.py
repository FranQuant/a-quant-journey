"""Adversarial tests for auditable trade accounting."""

import unittest

import pandas as pd

from src.quant_journey.backtesting.ledger import run_long_cash_ledger


class TestTradeAudit(unittest.TestCase):

    def test_purchase_and_terminal_sale_are_distinct(self):
        dates = pd.bdate_range("2023-01-02", periods=1)
        prices = pd.Series([100.0], index=dates)
        positions = pd.Series([1], index=dates)

        result = run_long_cash_ledger(
            prices,
            positions,
            initial_cash=1000.0,
            cost_bps=5.0,
            liquidate_final=True,
        )

        self.assertIn("execution_count", result.columns)
        self.assertIn("signed_shares_traded", result.columns)

        self.assertEqual(result.iloc[0]["execution_count"], 2)
        self.assertAlmostEqual(
            result.iloc[0]["signed_shares_traded"], 0.0
        )
        self.assertGreater(
            result.iloc[0]["trade_notional"], 0.0
        )

    def test_no_trade_has_zero_execution_count(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        prices = pd.Series([100., 101., 102.], index=dates)
        positions = pd.Series([0, 0, 0], index=dates)

        result = run_long_cash_ledger(prices, positions)

        self.assertEqual(
            result["execution_count"].tolist(), [0, 0, 0]
        )
        self.assertEqual(
            result["signed_shares_traded"].tolist(),
            [0.0, 0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
