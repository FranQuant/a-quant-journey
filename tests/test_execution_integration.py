"""Integration checks for execution-horizon research data."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.features.causal import make_causal_features
from src.quant_journey.execution.targets import make_execution_targets
from src.quant_journey.validation.label_aware import (
    split_by_label_availability,
)


class TestExecutionIntegration(unittest.TestCase):

    def test_alignment_and_label_maturity(self):
        dates = pd.bdate_range("2023-01-02", periods=25)
        prices = pd.Series(
            100.0 + np.arange(25) + np.sin(np.arange(25)),
            index=dates,
        )

        features = make_causal_features(prices)
        targets = make_execution_targets(prices)
        data = features.join(targets, how="inner")

        self.assertEqual(len(data), 18)
        self.assertEqual(data.index[0], dates[5])
        self.assertEqual(data.index[-1], dates[22])

        expected_returns = (
            prices.shift(-2) / prices.shift(-1) - 1
        ).loc[data.index]

        np.testing.assert_allclose(
            data["execution_target_return"],
            expected_returns,
        )

        expected_availability = dates[7:25]
        self.assertEqual(
            data["label_available_at"].tolist(),
            list(expected_availability),
        )

    def test_split_respects_actual_label_dates(self):
        dates = pd.bdate_range("2023-01-02", periods=35)
        prices = pd.Series(
            100.0 + np.arange(35),
            index=dates,
        )

        data = make_causal_features(prices).join(
            make_execution_targets(prices),
            how="inner",
        )

        train, validation, test = split_by_label_availability(data)

        n = len(data)
        train_boundary = data.index[int(n * 0.6)]
        validation_boundary = data.index[int(n * 0.8)]

        self.assertTrue(
            (train["label_available_at"] < train_boundary).all()
        )
        self.assertTrue(
            (
                validation["label_available_at"]
                < validation_boundary
            ).all()
        )

        self.assertEqual(
            len(data) - len(train) - len(validation) - len(test),
            4,
        )


if __name__ == "__main__":
    unittest.main()
