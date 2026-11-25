import unittest

import numpy as np
import pandas as pd

from src.quant_journey.execution.targets import make_execution_targets


class TestExecutionTargets(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=5)
        self.prices = pd.Series(
            [100., 110., 121., 108.9, 119.79],
            index=self.dates,
        )

    def test_return_and_label_availability(self):
        result = make_execution_targets(self.prices)

        self.assertEqual(len(result), 3)
        pd.testing.assert_index_equal(
            result.index,
            self.dates[:-2],
        )

        np.testing.assert_allclose(
            result["execution_target_return"],
            [0.10, -0.10, 0.10],
        )
        self.assertEqual(
            result["execution_target_up"].tolist(),
            [1, 0, 1],
        )
        self.assertEqual(
            result["label_available_at"].tolist(),
            list(self.dates[2:]),
        )

    def test_future_changes_do_not_rewrite_mature_labels(self):
        original = make_execution_targets(self.prices)

        altered = self.prices.copy()
        altered.iloc[-1] *= 2

        revised = make_execution_targets(altered)

        pd.testing.assert_frame_equal(
            original.iloc[:-1],
            revised.iloc[:-1],
        )

    def test_invalid_prices_rejected(self):
        altered = self.prices.copy()
        altered.iloc[2] = np.nan

        with self.assertRaises(ValueError):
            make_execution_targets(altered)


if __name__ == "__main__":
    unittest.main()
