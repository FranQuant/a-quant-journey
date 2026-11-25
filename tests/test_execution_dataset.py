"""Synthetic validation of the execution-aligned supervised dataset."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.execution.dataset import (
    make_execution_dataset,
)
from src.quant_journey.features.dataset import (
    make_supervised_dataset,
)


class TestExecutionDataset(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=30)
        self.prices = pd.Series(
            100.0 + np.arange(30) * 0.7
            + np.sin(np.arange(30) * 0.8) * 2,
            index=self.dates,
        )

    def test_execution_horizon_and_label_maturity(self):
        data = make_execution_dataset(self.prices)

        self.assertEqual(len(data), 23)
        self.assertEqual(data.index[0], self.dates[5])
        self.assertEqual(data.index[-1], self.dates[-3])

        expected = (
            self.prices.shift(-2)
            / self.prices.shift(-1) - 1
        ).loc[data.index]

        np.testing.assert_allclose(
            data["execution_target_return"], expected
        )

        self.assertEqual(
            data["label_available_at"].tolist(),
            list(self.dates[7:]),
        )

        np.testing.assert_array_equal(
            data["execution_target_up"],
            (expected > 0).astype(int),
        )

    def test_future_price_changes_do_not_rewrite_prior_features(self):
        baseline = make_execution_dataset(self.prices)

        modified = self.prices.copy()
        modified.iloc[20:] *= 1.25

        changed = make_execution_dataset(modified)

        columns = [
            "return_1d",
            "momentum_5d",
            "volatility_5d",
        ]

        pd.testing.assert_frame_equal(
            baseline.loc[:self.dates[19], columns],
            changed.loc[:self.dates[19], columns],
        )

    def test_forecast_and_execution_targets_remain_distinct(self):
        forecast = make_supervised_dataset(self.prices)
        execution = make_execution_dataset(self.prices)

        common = forecast.index.intersection(execution.index)

        self.assertEqual(
            len(forecast), len(execution) + 1
        )

        expected_forecast = (
            self.prices.shift(-1) / self.prices - 1
        ).loc[common]

        expected_execution = (
            self.prices.shift(-2)
            / self.prices.shift(-1) - 1
        ).loc[common]

        np.testing.assert_allclose(
            forecast.loc[common, "forward_return"],
            expected_forecast,
        )
        np.testing.assert_allclose(
            execution.loc[common, "execution_target_return"],
            expected_execution,
        )

        self.assertTrue(
            np.any(
                np.abs(
                    expected_forecast.to_numpy()
                    - expected_execution.to_numpy()
                ) > 1e-8
            )
        )

    def test_invalid_price_history_rejected(self):
        invalid = self.prices.copy()
        invalid.iloc[10] = np.nan

        with self.assertRaises(ValueError):
            make_execution_dataset(invalid)

    def test_insufficient_history_rejected(self):
        with self.assertRaises(ValueError):
            make_execution_dataset(self.prices.iloc[:5])


if __name__ == "__main__":
    unittest.main()
