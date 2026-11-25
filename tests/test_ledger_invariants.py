"""Financial conservation invariants for the synthetic ledger."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.backtesting.ledger import run_long_cash_ledger


class TestLedgerInvariants(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=6)
        self.prices = pd.Series(
            [100., 105., 102., 108., 103., 110.],
            index=self.dates,
        )

    def test_equity_conservation_each_date(self):
        positions = pd.Series(
            [1, 1, 0, 1, 1, 0],
            index=self.dates,
        )

        result = run_long_cash_ledger(
            self.prices,
            positions,
            initial_cash=1000.,
            cost_bps=5.,
        )

        np.testing.assert_allclose(
            result["pre_trade_equity"]
            - result["transaction_cost"],
            result["equity"],
            rtol=1e-12,
            atol=1e-9,
        )

        np.testing.assert_allclose(
            result["cash"] + result["shares"] * result["price"],
            result["equity"],
            rtol=1e-12,
            atol=1e-9,
        )

        self.assertTrue((result["cash"] >= -1e-9).all())
        self.assertTrue((result["shares"] >= 0).all())

    def test_signed_trades_reconcile_to_holdings(self):
        positions = pd.Series(
            [1, 1, 0, 1, 0, 1],
            index=self.dates,
        )

        result = run_long_cash_ledger(
            self.prices,
            positions,
            liquidate_final=True,
        )

        share_changes = (
            result["shares"]
            - result["shares"].shift(1).fillna(0)
        )

        np.testing.assert_allclose(
            result["signed_shares_traded"],
            share_changes,
            atol=1e-10,
        )

        self.assertEqual(result.iloc[-1]["execution_count"], 2)
        self.assertEqual(result.iloc[-1]["shares"], 0.0)

    def test_cost_equals_traded_notional_times_rate(self):
        positions = pd.Series(
            [1, 0, 1, 0, 1, 1],
            index=self.dates,
        )

        result = run_long_cash_ledger(
            self.prices,
            positions,
            cost_bps=5.,
            liquidate_final=True,
        )

        np.testing.assert_allclose(
            result["transaction_cost"],
            result["trade_notional"] * 0.0005,
            rtol=1e-12,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
