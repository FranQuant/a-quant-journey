import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.quant_journey.data.tpq import load_spy_prices, calculate_returns


DATA = Path("data/raw/tpq_eod_2014_2023.csv")


class TestTPQ(unittest.TestCase):

    def test_historical_dataset(self):
        prices = load_spy_prices(DATA)
        self.assertEqual(len(prices), 2516)
        self.assertTrue(prices.index.is_unique)
        self.assertTrue(prices.index.is_monotonic_increasing)
        self.assertFalse(prices.isna().any())

        returns = calculate_returns(prices)
        self.assertEqual(len(returns), 2515)
        self.assertFalse(returns.isna().any().any())

    def test_return_formulas(self):
        prices = pd.Series([100., 110., 99.])
        result = calculate_returns(prices)

        np.testing.assert_allclose(
            result["return_simple"], [0.10, -0.10]
        )
        np.testing.assert_allclose(
            result["return_log"],
            [np.log(1.10), np.log(0.90)]
        )

    def test_invalid_inputs(self):
        cases = [
            pd.DataFrame({"OTHER": [100., 101.]}),
            pd.DataFrame({"SPY": [100., 0.]}),
            pd.DataFrame({"SPY": [100., -1.]}),
            pd.DataFrame({"SPY": [np.inf, 101.]}),
        ]

        for frame in cases:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "prices.csv"
                frame.to_csv(path)
                with self.assertRaises(ValueError):
                    load_spy_prices(path)

    def test_missing_price_not_imputed(self):
        frame = pd.DataFrame(
            {"SPY": [100., np.nan, 110.]},
            index=pd.date_range("2023-01-02", periods=3)
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            frame.to_csv(path)
            result = load_spy_prices(path)

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[-1], 110.)


if __name__ == "__main__":
    unittest.main()


class TestDateIntegrity(unittest.TestCase):

    def test_invalid_date_indexes(self):
        cases = [
            ["2023-01-02", "2023-01-02"],  # Duplicate
            ["2023-01-03", "2023-01-02"],  # Unsorted
            ["2023-01-02", "not-a-date"],  # Malformed
        ]

        for dates in cases:
            with self.subTest(dates=dates):
                frame = pd.DataFrame(
                    {"SPY": [100.0, 101.0]},
                    index=dates,
                )

                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "prices.csv"
                    frame.to_csv(path)

                    with self.assertRaises(ValueError):
                        load_spy_prices(path)
