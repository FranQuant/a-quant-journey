import unittest

import numpy as np
import pandas as pd

from src.quant_journey.execution.timing import make_timing_contract


class TestTimingContract(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=5)
        self.prices = pd.Series(
            [100., 110., 121., 108.9, 119.79],
            index=self.dates,
        )

    def test_explicit_return_horizons(self):
        result = make_timing_contract(self.prices)

        self.assertEqual(len(result), 4)
        self.assertEqual(result.index[0], self.dates[0])
        self.assertEqual(result.index[-1], self.dates[3])

        np.testing.assert_allclose(
            result["forecast_return"].to_numpy(),
            [0.10, 0.10, -0.10, 0.10],
        )
        np.testing.assert_allclose(
            result["execution_proxy_return"].iloc[:3],
            [0.10, -0.10, 0.10],
        )
        self.assertTrue(
            pd.isna(result["execution_proxy_return"].iloc[-1])
        )

    def test_future_price_invariance(self):
        original = make_timing_contract(self.prices)

        altered = self.prices.copy()
        altered.iloc[-1] *= 2

        revised = make_timing_contract(altered)

        pd.testing.assert_frame_equal(
            original.iloc[:2],
            revised.iloc[:2],
        )

    def test_invalid_input_rejected(self):
        invalid = self.prices.copy()
        invalid.iloc[2] = np.nan

        with self.assertRaises(ValueError):
            make_timing_contract(invalid)


if __name__ == "__main__":
    unittest.main()
