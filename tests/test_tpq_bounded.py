from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from src.quant_journey.data.tpq import (
    load_spy_observations_through,
    load_spy_prices_through,
    load_tpq_columns_through,
)


def guard_csv(monkeypatch, allowed, remainder):
    class GuardedCSV(BytesIO):
        def read(self, size=-1):
            assert size == 1 and self.tell() < len(allowed)
            return super().read(size)

        def readline(self, size=-1):
            raise AssertionError("LF-only framing can cross a bare CR")

    def open_fixture(path, mode, *, buffering):
        assert path == Path("fixture.csv")
        assert mode == "rb" and buffering == 0
        return GuardedCSV(allowed + remainder)

    monkeypatch.setattr(Path, "open", open_fixture)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("cutoff,suffix", [
    ("2020-01-02", b""),
    ("2020-01-03", b"2020-01-04,"),
])
def test_read_boundary(monkeypatch, ending, cutoff, suffix):
    prefix = ending.join([b'Date,"PEER,A"', b'2020-01-02,"12"', suffix])
    guard_csv(monkeypatch, prefix, b"forbidden\n")
    frame = load_tpq_columns_through("fixture.csv", ["PEER,A"], cutoff)
    assert frame["PEER,A"].tolist() == [12.0]


@pytest.mark.parametrize("record", [b"Date,PEER", b"Date,PEER\n2020-01-02,12"])
@pytest.mark.parametrize("lookahead,remainder", [
    (b"2", b"020-01-04,forbidden\n"),
    (b"\r", b"\n2020-01-04,forbidden\n"),
])
def test_invalid_record_ending(monkeypatch, record, lookahead, remainder):
    guard_csv(monkeypatch, record + b"\r" + lookahead, remainder)
    with pytest.raises(ValueError, match="record ending"):
        load_tpq_columns_through("fixture.csv", ["PEER"], "2020-01-02")


def test_order_and_independent_missing_prices(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,SPY,B\n2020-01-02,,3\n2020-01-03,2,")
    cutoff = pd.Timestamp("2020-01-04")
    frame = load_tpq_columns_through(path, ["B", "SPY"], cutoff)
    expected = pd.DataFrame(
        {"B": [3., float("nan")], "SPY": [float("nan"), 2.]},
        index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="Date"),
    )
    pd.testing.assert_frame_equal(frame, expected, check_index_type=False)
    pd.testing.assert_frame_equal(load_spy_observations_through(path, cutoff), frame[["SPY"]])
    pd.testing.assert_series_equal(load_spy_prices_through(path, cutoff), frame["SPY"].dropna())


@pytest.mark.parametrize("header,columns", [
    ("Date,A", ["SPY"]), ("Date,A", []), ("Date,A", ["Date"]),
    ("Date,A,B,B", ["A"]), ("A,Date", ["A"]),
])
def test_invalid_selection_or_header(tmp_path, header, columns):
    path = tmp_path / "prices.csv"
    path.write_text(header + "\n2020-01-02,1\n")
    with pytest.raises(ValueError):
        load_tpq_columns_through(path, columns, "2020-01-02")


@pytest.mark.parametrize("cutoff", ["not-a-date", "2020-02-30", pd.NaT, "2019-12-31"])
def test_invalid_or_empty_cutoff(tmp_path, cutoff):
    path = tmp_path / "prices.csv"
    path.write_text("Date,A\n2020-01-02,123\n")
    with pytest.raises(ValueError):
        load_tpq_columns_through(path, ["A"], cutoff)


@pytest.mark.parametrize("row", [
    ",2", "\n", "bad,2", "2020-02-30,2", "2020-01", "2020-01-03",
    "2020-01-02,2", "2020-01-01,2", "2020-01-03,abc",
    "2020-01-03,inf", "2020-01-03,0", "2020-01-03,-2",
])
def test_invalid_eligible_record(tmp_path, row):
    path = tmp_path / "prices.csv"
    path.write_text("Date,A\n2020-01-02,1\n" + row)
    with pytest.raises(ValueError):
        load_tpq_columns_through(path, ["A"], "2020-12-31")


def test_entirely_blank_requested_peer(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,SPY,B\n2020-01-02,1,\n2020-01-03,2,4\n")
    with pytest.raises(ValueError, match="Entirely blank"):
        load_tpq_columns_through(path, ["SPY", "B"], "2020-01-02")
