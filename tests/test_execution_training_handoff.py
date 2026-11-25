"""Synthetic integration of execution targets, splits, and model fitting."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.execution.dataset import (
    make_execution_dataset,
)
from src.quant_journey.validation.label_aware import (
    split_by_label_availability,
)
from src.quant_journey.models.execution_benchmark import (
    fit_execution_benchmark,
    predict_execution_benchmark,
)
from src.quant_journey.models.preprocessing import FEATURE_COLUMNS


class TestExecutionTrainingHandoff(unittest.TestCase):

    def test_training_and_validation_alignment(self):
        rng = np.random.default_rng(42)
        n = 160
        dates = pd.bdate_range("2023-01-02", periods=n)

        returns = rng.normal(0.0002, 0.012, n)
        prices = pd.Series(
            100.0 * np.cumprod(1.0 + returns),
            index=dates,
        )

        dataset = make_execution_dataset(prices)

        train, validation, test = split_by_label_availability(
            dataset
        )

        cutoff = validation.index.min()

        self.assertTrue(
            (train["label_available_at"] < cutoff).all()
        )
        self.assertTrue(
            (train.index < cutoff).all()
        )
        self.assertTrue(
            (validation.index < test.index.min()).all()
        )

        model = fit_execution_benchmark(train, cutoff)

        np.testing.assert_allclose(
            model["scaler"].mean_,
            train[list(FEATURE_COLUMNS)].mean().to_numpy(),
            atol=1e-12,
        )

        predictions = predict_execution_benchmark(
            model, validation
        )

        pd.testing.assert_index_equal(
            predictions.index,
            validation.index,
        )

        self.assertTrue(predictions.isin([0, 1]).all())
        self.assertEqual(len(predictions), len(validation))

        # Test partition is structurally present but never supplied
        # to either model fitting or prediction.
        self.assertGreater(len(test), 0)


if __name__ == "__main__":
    unittest.main()
