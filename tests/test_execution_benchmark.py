"""Synthetic tests for the execution-aligned logistic benchmark."""

import unittest

import numpy as np
import pandas as pd

from src.quant_journey.models.execution_benchmark import (
    fit_execution_benchmark,
    predict_execution_benchmark,
)


class TestExecutionBenchmark(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 80
        dates = pd.bdate_range("2023-01-02", periods=n + 3)

        self.train = pd.DataFrame({
            "return_1d": rng.normal(0, 0.01, n),
            "momentum_5d": rng.normal(0, 0.02, n),
            "volatility_5d": rng.uniform(0.002, 0.03, n),
            "execution_target_up": np.array([0, 1] * (n // 2)),
            "label_available_at": dates[2:n + 2],
        }, index=dates[:n])

        self.cutoff = dates[n + 2]

    def test_correct_target_used(self):
        train = self.train.copy()

        # Deliberately contradictory legacy labels.
        train["target_up"] = 1 - train["execution_target_up"]

        first = fit_execution_benchmark(train, self.cutoff)
        second = fit_execution_benchmark(
            train.drop(columns="target_up"), self.cutoff
        )

        np.testing.assert_allclose(
            first["classifier"].coef_,
            second["classifier"].coef_,
        )

    def test_unavailable_label_rejected(self):
        train = self.train.copy()
        train.loc[train.index[-1], "label_available_at"] = self.cutoff

        with self.assertRaises(ValueError):
            fit_execution_benchmark(train, self.cutoff)

    def test_predictions_do_not_refit(self):
        model = fit_execution_benchmark(self.train, self.cutoff)

        coefficients = model["classifier"].coef_.copy()
        scaler_mean = model["scaler"].mean_.copy()

        changed = self.train.copy()
        changed["execution_target_up"] = (
            1 - changed["execution_target_up"]
        )

        original = predict_execution_benchmark(model, self.train)
        altered = predict_execution_benchmark(model, changed)

        pd.testing.assert_series_equal(original, altered)

        np.testing.assert_array_equal(
            coefficients, model["classifier"].coef_
        )
        np.testing.assert_array_equal(
            scaler_mean, model["scaler"].mean_
        )

    def test_deterministic_fit(self):
        first = fit_execution_benchmark(self.train, self.cutoff)
        second = fit_execution_benchmark(self.train, self.cutoff)

        np.testing.assert_allclose(
            first["classifier"].coef_,
            second["classifier"].coef_,
        )

    def test_single_class_rejected(self):
        train = self.train.copy()
        train["execution_target_up"] = 1

        with self.assertRaises(ValueError):
            fit_execution_benchmark(train, self.cutoff)


if __name__ == "__main__":
    unittest.main()
