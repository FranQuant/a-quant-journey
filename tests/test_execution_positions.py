"""Adversarial tests for signal-to-position mapping."""

import unittest

import pandas as pd

from src.quant_journey.execution.signals import (
    map_signals_to_executions,
)
from src.quant_journey.execution.positions import (
    make_execution_positions,
)


class TestExecutionPositions(unittest.TestCase):

    def setUp(self):
        self.dates = pd.to_datetime([
            "2023-01-02",
            "2023-01-03",
            "2023-01-05",
            "2023-01-06",
            "2023-01-09",
        ])
        self.prices = pd.Series(
            [100., 102., 101., 103., 105.],
            index=self.dates,
        )

    def test_persistence_and_initial_cash(self):
        signals = pd.Series(
            [1, 0],
            index=self.dates[[0, 2]],
        )

        executions = map_signals_to_executions(
            self.prices, signals
        )
        positions = make_execution_positions(
            self.prices, executions
        )

        self.assertEqual(
            positions.tolist(), [0, 1, 1, 0, 0]
        )

    def test_empty_executions_remain_cash(self):
        executions = map_signals_to_executions(
            self.prices,
            pd.Series(dtype=int, index=pd.DatetimeIndex([])),
        )
        positions = make_execution_positions(
            self.prices, executions
        )

        self.assertEqual(positions.tolist(), [0] * 5)

    def test_final_signal_is_not_executed(self):
        signals = pd.Series(
            [1], index=self.dates[-1:]
        )

        executions = map_signals_to_executions(
            self.prices, signals
        )
        positions = make_execution_positions(
            self.prices, executions
        )

        self.assertEqual(positions.tolist(), [0] * 5)

    def test_same_close_execution_rejected(self):
        signals = pd.Series(
            [1], index=self.dates[:1]
        )
        executions = map_signals_to_executions(
            self.prices, signals
        ).copy()

        executions["signal_date"] = executions["execution_date"]

        with self.assertRaises(ValueError):
            make_execution_positions(self.prices, executions)

    def test_fabricated_execution_price_rejected(self):
        signals = pd.Series(
            [1], index=self.dates[:1]
        )
        executions = map_signals_to_executions(
            self.prices, signals
        ).copy()

        executions["execution_price"] = 999.

        with self.assertRaises(ValueError):
            make_execution_positions(self.prices, executions)

    def test_duplicate_execution_rejected(self):
        signals = pd.Series(
            [1], index=self.dates[:1]
        )
        executions = map_signals_to_executions(
            self.prices, signals
        )

        duplicated = pd.concat([executions, executions])

        with self.assertRaises(ValueError):
            make_execution_positions(self.prices, duplicated)


if __name__ == "__main__":
    unittest.main()
