"""Focused safeguards for the authorized A11 date-bounded evaluation."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.quant_journey.data.tpq import (
    load_spy_observations_through,
    load_spy_prices_through,
    load_tpq_columns_through,
)
from src.quant_journey.validation.label_aware import (
    split_train_validation_by_dates,
)


class TestA11Boundaries(unittest.TestCase):
    def test_date_scan_stops_before_first_reserved_price_field(self):
        eligible = (
            b"Date,SPY,AAA,BBB\n"
            b"2020-01-01,100,10,20\n"
            b"2020-01-02,101,11,21\n"
        )
        reserved = b"2020-01-03,RESERVED_PRICE_SENTINEL,0,0\n"

        class GuardedStream:
            def __init__(self):
                self.stream = io.BytesIO(eligible + reserved)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                self.stream.close()

            def read(self, size=-1):
                if size < 0 or self.stream.tell() + size > len(eligible):
                    raise AssertionError("Read requested beyond the cutoff row")
                return self.stream.read(size)

        path = Path("synthetic.csv")
        with patch.object(Path, "open", autospec=True, return_value=GuardedStream()) as opener:
            frame = load_tpq_columns_through(path, ["AAA", "SPY"], "2020-01-02")
        opener.assert_called_once_with(path, "rb", buffering=0)

        expected = pd.DataFrame(
            {"AAA": [10.0, 11.0], "SPY": [100.0, 101.0]},
            index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="Date"),
        )
        pd.testing.assert_frame_equal(frame, expected)

    def test_bounded_loader_does_not_parse_prices_after_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prices.csv"
            path.write_text(
                "Date,SPY\n"
                "2020-01-01,100\n"
                "2020-01-02,101\n"
                "RESERVED_DATE_SENTINEL,RESERVED_PRICE_SENTINEL\n",
                encoding="utf-8",
            )
            prices = load_spy_prices_through(path, "2020-01-02")
        self.assertEqual(prices.index.max(), pd.Timestamp("2020-01-02"))
        self.assertEqual(prices.tolist(), [100.0, 101.0])

    def test_bounded_observations_preserve_blank_prices_and_ignore_later_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prices.csv"
            path.write_text(
                "Date,SPY\n"
                "2020-01-01,100\n"
                "2020-01-02,\n"
                "2020-01-03,RESERVED_PRICE_SENTINEL\n",
                encoding="utf-8",
            )
            observations = load_spy_observations_through(path, "2020-01-02")
        self.assertEqual(observations.index.strftime("%Y-%m-%d").tolist(), [
            "2020-01-01", "2020-01-02",
        ])
        self.assertEqual(observations["SPY"].iloc[0], 100.0)
        self.assertTrue(pd.isna(observations["SPY"].iloc[1]))

    def test_bounded_multi_column_loader_reads_only_requested_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prices.csv"
            path.write_text(
                "Date,SPY,AAA,BBB\n"
                "2020-01-01,100,10,20\n"
                "2020-01-02,101,11,21\n"
                "RESERVED_DATE_SENTINEL,RESERVED_PRICE_SENTINEL,0,0\n",
                encoding="utf-8",
            )
            frame = load_tpq_columns_through(path, ["SPY", "AAA"], "2020-01-02")
        self.assertEqual(frame.columns.tolist(), ["SPY", "AAA"])
        self.assertEqual(frame.index.max(), pd.Timestamp("2020-01-02"))
        self.assertEqual(frame.iloc[-1].tolist(), [101.0, 11.0])

    @staticmethod
    def split_fixture():
        dates = pd.to_datetime([
            "2019-12-27", "2019-12-30", "2020-01-02", "2020-01-03",
            "2021-12-23", "2021-12-27",
        ])
        return pd.DataFrame({
            "label_available_at": pd.to_datetime([
                "2019-12-31", "2020-01-02", "2020-01-06", "2021-12-28",
                "2021-12-29", "2021-12-30",
            ]),
            "feature": range(len(dates)),
        }, index=dates)

    def test_date_split_purges_labels_without_returning_test(self):
        data = self.split_fixture()
        boundaries = ("2020-01-02", "2021-12-29")
        train, validation = split_train_validation_by_dates(data, *boundaries)

        pd.testing.assert_index_equal(train.index, data.index[:1])
        pd.testing.assert_index_equal(validation.index, data.index[2:4])
        self.assertNotIn(data.index[1], train.index)  # Matures at validation start.
        for date in data.index[4:]:  # Mature at and after reserved start.
            self.assertNotIn(date, validation.index)
        self.assertLess(train.index.max(), validation.index.min())
        self.assertLess(train["label_available_at"].max(), pd.Timestamp(boundaries[0]))
        self.assertLess(validation["label_available_at"].max(), pd.Timestamp(boundaries[1]))

        repeat_train, repeat_validation = split_train_validation_by_dates(data, *boundaries)
        pd.testing.assert_frame_equal(train, repeat_train)
        pd.testing.assert_frame_equal(validation, repeat_validation)

    def test_date_split_rejects_reserved_test_feature_date(self):
        for date in pd.to_datetime(["2021-12-29", "2021-12-30"]):
            with self.subTest(date=date):
                data = self.split_fixture()
                data.loc[date, "label_available_at"] = date + pd.Timedelta(days=1)
                with self.assertRaisesRegex(ValueError, "reserved test"):
                    split_train_validation_by_dates(data, "2020-01-02", "2021-12-29")

    def test_date_split_rejects_invalid_inputs(self):
        data = self.split_fixture()
        maturity = data["label_available_at"]
        cases = {
            "missing metadata": data.drop(columns="label_available_at"),
            "non-datetime maturity": data.assign(label_available_at=maturity.astype(str)),
            "missing maturity": data.assign(label_available_at=maturity.mask(data.index == data.index[0])),
            "nonfuture maturity": data.assign(label_available_at=data.index),
            "duplicate dates": data.iloc[[0, 0, 2, 3]],
            "unsorted dates": data.iloc[::-1],
            "empty input": data.iloc[:0],
            "all training labels purged": data.iloc[1:],
            "all validation labels purged": data.iloc[[0, 1, 4, 5]],
        }
        for name, invalid in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                split_train_validation_by_dates(invalid, "2020-01-02", "2021-12-29")

    def test_date_split_rejects_invalid_cutoffs(self):
        data = self.split_fixture()
        validation, reserved = "2020-01-02", "2021-12-29"
        for start, end in [
            ("invalid", reserved), (validation, "invalid"),
            (None, reserved), (validation, pd.NaT),
            (validation, validation), (reserved, validation),
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                split_train_validation_by_dates(data, start, end)


if __name__ == "__main__":
    unittest.main()
