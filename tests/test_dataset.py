import unittest

import numpy as np
import pandas as pd

from src.quant_journey.features.dataset import make_supervised_dataset


class TestSupervisedDataset(unittest.TestCase):

    def test_alignment_and_last_label_exclusion(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        prices = pd.Series(
            [100., 102., 101., 105., 107.,
             106., 109., 111., 110., 114.],
            index=dates,
        )

        data = make_supervised_dataset(prices)

        self.assertEqual(len(data), 4)
        self.assertEqual(data.index[0], dates[5])
        self.assertEqual(data.index[-1], dates[8])

        np.testing.assert_allclose(
            data["forward_return"],
            prices.shift(-1).div(prices).sub(1).loc[data.index],
        )

        expected = (data["forward_return"] > 0).astype(int)
        pd.testing.assert_series_equal(
            data["target_up"], expected.rename("target_up")
        )

    def test_future_price_does_not_change_prior_rows(self):
        dates = pd.bdate_range("2023-01-02", periods=20)
        prices = pd.Series(
            100. + np.arange(20), index=dates
        )

        original = make_supervised_dataset(prices)
        changed = prices.copy()
        changed.iloc[15:] *= 2
        revised = make_supervised_dataset(changed)

        pd.testing.assert_frame_equal(
            original.loc[:dates[13]],
            revised.loc[:dates[13]],
        )


if __name__ == "__main__":
    unittest.main()
