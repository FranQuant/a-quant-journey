import unittest

import numpy as np
import pandas as pd

from src.quant_journey.features.targets import make_forward_targets


class TestForwardTargets(unittest.TestCase):

    def test_forward_alignment(self):
        dates = pd.date_range("2023-01-02", periods=4)
        prices = pd.Series([100., 110., 99., 99.], index=dates)
        result = make_forward_targets(prices)

        self.assertEqual(result.index.tolist(), dates[:-1].tolist())
        np.testing.assert_allclose(
            result["forward_return"], [0.10, -0.10, 0.0]
        )
        self.assertEqual(result["target_up"].tolist(), [1, 0, 0])

    def test_future_changes_do_not_rewrite_past_labels(self):
        dates = pd.date_range("2023-01-02", periods=5)
        prices = pd.Series([100., 101., 102., 103., 104.], index=dates)

        original = make_forward_targets(prices)
        changed = prices.copy()
        changed.iloc[-1] = 500.
        revised = make_forward_targets(changed)

        pd.testing.assert_frame_equal(
            original.iloc[:-1], revised.iloc[:-1]
        )

    def test_insufficient_history(self):
        prices = pd.Series(
            [100.], index=pd.to_datetime(["2023-01-02"])
        )
        with self.assertRaises(ValueError):
            make_forward_targets(prices)


if __name__ == "__main__":
    unittest.main()
