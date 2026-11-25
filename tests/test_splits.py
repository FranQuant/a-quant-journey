import unittest
import pandas as pd

from src.quant_journey.validation.splits import split_chronologically


class TestChronologicalSplits(unittest.TestCase):

    def setUp(self):
        dates = pd.bdate_range("2023-01-02", periods=20)
        self.data = pd.DataFrame(
            {"feature": range(20), "target_up": [0, 1] * 10},
            index=dates,
        )

    def test_sizes_and_boundaries(self):
        train, val, test = split_chronologically(self.data)

        self.assertEqual((len(train), len(val), len(test)), (11, 3, 4))
        self.assertEqual(train.index[-1], self.data.index[10])
        self.assertEqual(val.index[0], self.data.index[12])
        self.assertEqual(val.index[-1], self.data.index[14])
        self.assertEqual(test.index[0], self.data.index[16])

    def test_boundary_purging(self):
        train, val, test = split_chronologically(self.data)

        for earlier, later in ((train, val), (val, test)):
            position = self.data.index.get_loc(earlier.index[-1])
            label_end = self.data.index[position + 1]
            self.assertLess(label_end, later.index[0])

    def test_invalid_fractions(self):
        with self.assertRaises(ValueError):
            split_chronologically(
                self.data, train_fraction=0.9,
                validation_fraction=0.2,
            )


if __name__ == "__main__":
    unittest.main()
