import unittest

import numpy as np
import pandas as pd

from src.quant_journey.models.preprocessing import (
    FEATURE_COLUMNS,
    fit_training_scaler,
    transform_features,
)


class TestPreprocessingIsolation(unittest.TestCase):

    def setUp(self):
        self.train = pd.DataFrame({
            "return_1d": [0.01, -0.02, 0.03, 0.00],
            "momentum_5d": [0.04, -0.01, 0.02, 0.05],
            "volatility_5d": [0.01, 0.02, 0.03, 0.04],
            "forward_return": [0.03, -0.01, 0.02, -0.02],
            "target_up": [1, 0, 1, 0],
        })

    def test_training_scaler_statistics(self):
        scaler = fit_training_scaler(self.train)

        np.testing.assert_allclose(
            scaler.mean_,
            self.train[list(FEATURE_COLUMNS)].mean().to_numpy(),
        )

    def test_labels_cannot_affect_scaler(self):
        original = fit_training_scaler(self.train)

        changed = self.train.copy()
        changed["forward_return"] *= 1000
        changed["target_up"] = 1 - changed["target_up"]

        revised = fit_training_scaler(changed)

        np.testing.assert_allclose(original.mean_, revised.mean_)
        np.testing.assert_allclose(original.scale_, revised.scale_)

    def test_validation_does_not_refit_scaler(self):
        scaler = fit_training_scaler(self.train)
        mean_before = scaler.mean_.copy()
        scale_before = scaler.scale_.copy()

        validation = self.train.copy()
        validation[list(FEATURE_COLUMNS)] *= 100

        transformed = transform_features(validation, scaler)

        self.assertEqual(transformed.shape, (4, 3))
        np.testing.assert_allclose(scaler.mean_, mean_before)
        np.testing.assert_allclose(scaler.scale_, scale_before)


if __name__ == "__main__":
    unittest.main()
