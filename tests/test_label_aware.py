import unittest

import pandas as pd

from src.quant_journey.validation.label_aware import (
    split_by_label_availability,
)


class TestLabelAwareSplits(unittest.TestCase):

    def setUp(self):
        dates = pd.bdate_range("2023-01-02", periods=22)

        self.data = pd.DataFrame({
            "feature": range(20),
            "label_available_at": dates[2:],
        }, index=dates[:20])

    def test_two_session_label_purging(self):
        train, validation, test = split_by_label_availability(
            self.data
        )

        self.assertEqual(
            (len(train), len(validation), len(test)),
            (10, 2, 4),
        )

        train_boundary = self.data.index[12]
        validation_boundary = self.data.index[16]

        self.assertTrue(
            (train["label_available_at"] < train_boundary).all()
        )
        self.assertTrue(
            (
                validation["label_available_at"]
                < validation_boundary
            ).all()
        )

    def test_delayed_labels_are_purged(self):
        altered = self.data.copy()
        altered.loc[altered.index[5], "label_available_at"] = (
            altered.index[18]
        )

        train, _, _ = split_by_label_availability(altered)

        self.assertNotIn(altered.index[5], train.index)

    def test_missing_availability_rejected(self):
        invalid = self.data.drop(columns=["label_available_at"])

        with self.assertRaises(ValueError):
            split_by_label_availability(invalid)

    def test_nonfuture_label_rejected(self):
        invalid = self.data.copy()
        invalid.loc[
            invalid.index[3], "label_available_at"
        ] = invalid.index[3]

        with self.assertRaises(ValueError):
            split_by_label_availability(invalid)


if __name__ == "__main__":
    unittest.main()
