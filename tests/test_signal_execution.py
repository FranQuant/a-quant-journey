"""Synthetic tests for timestamp-safe signal execution mapping."""

import unittest

import pandas as pd

from src.quant_journey.execution.signals import (
    map_signals_to_executions,
)


class TestSignalExecution(unittest.TestCase):

    def setUp(self):
        # Deliberately skip January 4.
        self.dates = pd.to_datetime([
            "2023-01-02",
            "2023-01-03",
            "2023-01-05",
            "2023-01-06",
        ])

        self.prices = pd.Series(
            [100.0, 102.0, 101.0, 103.0],
            index=self.dates,
        )

    def test_signal_executes_next_observed_session(self):
        signals = pd.Series(
            [1, 0],
            index=self.dates[:2],
        )

        result = map_signals_to_executions(
            self.prices, signals
        )

        self.assertEqual(len(result), 2)

        self.assertEqual(
            result["execution_date"].tolist(),
            list(self.dates[1:3]),
        )

        self.assertEqual(
            result["signal_date"].tolist(),
            list(self.dates[:2]),
        )

        self.assertEqual(
            result["execution_price"].tolist(),
            [102.0, 101.0],
        )

        self.assertTrue(
            (result["execution_date"] > result["signal_date"]).all()
        )

    def test_missing_session_not_fabricated(self):
        signals = pd.Series(
            [1], index=self.dates[1:2]
        )

        result = map_signals_to_executions(
            self.prices, signals
        )

        self.assertEqual(
            result.iloc[0]["execution_date"],
            pd.Timestamp("2023-01-05"),
        )
        self.assertEqual(
            result.iloc[0]["execution_price"], 101.0
        )

    def test_final_signal_has_no_execution(self):
        signals = pd.Series(
            [1], index=self.dates[-1:]
        )

        result = map_signals_to_executions(
            self.prices, signals
        )

        self.assertTrue(result.empty)

    def test_invalid_signal_date_rejected(self):
        signals = pd.Series(
            [1],
            index=pd.to_datetime(["2023-01-04"]),
        )

        with self.assertRaises(ValueError):
            map_signals_to_executions(
                self.prices, signals
            )

    def test_invalid_position_rejected(self):
        signals = pd.Series(
            [-1], index=self.dates[:1]
        )

        with self.assertRaises(ValueError):
            map_signals_to_executions(
                self.prices, signals
            )


if __name__ == "__main__":
    unittest.main()
