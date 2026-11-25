import unittest

import numpy as np
import pandas as pd

from src.quant_journey.models.benchmark import (
    fit_logistic_benchmark,
    predict_benchmark,
)


class TestLogisticBenchmark(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 100

        self.train = pd.DataFrame({
            "return_1d": rng.normal(0, 0.01, n),
            "momentum_5d": rng.normal(0, 0.02, n),
            "volatility_5d": rng.uniform(0.002, 0.03, n),
            "target_up": np.array([0, 1] * (n // 2)),
        })

    def test_deterministic_fit(self):
        first = fit_logistic_benchmark(self.train)
        second = fit_logistic_benchmark(self.train)

        np.testing.assert_allclose(
            first["classifier"].coef_,
            second["classifier"].coef_,
        )

    def test_prediction_does_not_refit(self):
        model = fit_logistic_benchmark(self.train)

        coefficients = model["classifier"].coef_.copy()
        scaler_mean = model["scaler"].mean_.copy()

        validation = self.train.copy()
        validation["return_1d"] *= 100
        validation["target_up"] = 1 - validation["target_up"]

        predictions = predict_benchmark(model, validation)

        self.assertEqual(len(predictions), len(validation))
        np.testing.assert_array_equal(
            model["classifier"].coef_, coefficients
        )
        np.testing.assert_array_equal(
            model["scaler"].mean_, scaler_mean
        )

    def test_validation_labels_do_not_affect_predictions(self):
        model = fit_logistic_benchmark(self.train)

        validation = self.train.copy()
        altered = validation.copy()
        altered["target_up"] = 1 - altered["target_up"]

        pd.testing.assert_series_equal(
            predict_benchmark(model, validation),
            predict_benchmark(model, altered),
        )

    def test_single_class_rejected(self):
        train = self.train.copy()
        train["target_up"] = 1

        with self.assertRaises(ValueError):
            fit_logistic_benchmark(train)


if __name__ == "__main__":
    unittest.main()
