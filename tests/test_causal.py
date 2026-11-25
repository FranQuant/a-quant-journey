import unittest

import numpy as np
import pandas as pd

from src.quant_journey.features.causal import make_causal_features


class TestCausalFeatures(unittest.TestCase):

    def test_future_price_invariance(self):
        dates = pd.bdate_range("2023-01-02", periods=20)
        prices = pd.Series(
            100 + np.arange(20) + np.sin(np.arange(20)),
            index=dates,
        )

        original = make_causal_features(prices)

        changed = prices.copy()
        changed.iloc[15:] *= 3
        revised = make_causal_features(changed)

        pd.testing.assert_frame_equal(
            original.loc[:dates[14]],
            revised.loc[:dates[14]],
        )

    def test_warmup_and_formulas(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        prices = pd.Series(
            [100., 102., 101., 105., 107.,
             106., 109., 111., 110., 114.],
            index=dates,
        )

        features = make_causal_features(prices)

        self.assertEqual(len(features), 5)
        self.assertEqual(features.index[0], dates[5])
        self.assertAlmostEqual(
            features.iloc[0]["momentum_5d"], 0.06
        )
        self.assertAlmostEqual(
            features.iloc[0]["return_1d"], 106 / 107 - 1
        )

    def test_invalid_prices(self):
        dates = pd.bdate_range("2023-01-02", periods=6)
        prices = pd.Series(
            [100., 101., np.nan, 103., 104., 105.],
            index=dates,
        )
        with self.assertRaises(ValueError):
            make_causal_features(prices)


if __name__ == "__main__":
    unittest.main()
