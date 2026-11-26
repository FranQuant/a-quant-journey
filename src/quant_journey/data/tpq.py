"""Validated historical TPQ market-data loader."""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


INSTRUMENTS = (
    "AAPL.O", "MSFT.O", "INTC.O", "AMZN.O", "GS.N", ".SPX",
    ".VIX", "SPY", "EUR=", "XAU=", "GDX", "GLD",
)


def observed_sessions(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Omit inferred shared US calendar blanks. Preserve unexplained gaps."""
    us_sessions = panel["SPY"].notna() | panel[".SPX"].notna()
    return {
        name: series if name in ("EUR=", "XAU=") else series.loc[us_sessions]
        for name, series in panel.items()
    }


def _read_tpq_date(source) -> date | None:
    """Read the leading ISO date and its comma, returning None only at EOF."""
    field = bytearray()
    while True:
        char = source.read(1)
        if not char and not field:
            return None
        if char == b",":
            break
        if len(field) == 10 or not (char.isdigit() or char == b"-"):
            raise ValueError("Invalid TPQ date field")
        field.extend(char)
    text = field.decode("ascii")
    day = date.fromisoformat(text)
    if day.isoformat() != text or day.year < 1900:
        raise ValueError("Invalid TPQ date field")
    return day


def _read_csv_fields(source) -> list[str]:
    """Read one single-line record (or its price suffix) without prefetching."""
    line = bytearray()
    while char := source.read(1):
        if char == b"\n":
            break
        if char == b"\r":
            if source.read(1) != b"\n":
                raise ValueError("Invalid TPQ record ending")
            break
        line.extend(char)
    try:
        return next(csv.reader([line.decode("utf-8")], strict=True))
    except csv.Error as exc:
        raise ValueError("Invalid TPQ CSV record") from exc


def load_spy_prices(path: str | Path) -> pd.Series:
    """Return valid SPY session closes without price imputation."""
    data = pd.read_csv(path, index_col=0)

    try:
        data.index = pd.DatetimeIndex(
            pd.to_datetime(data.index, format="%Y-%m-%d", errors="raise")
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid date index") from exc

    if "SPY" not in data.columns:
        raise ValueError("Missing SPY column")

    if data.index.hasnans or not data.index.is_unique:
        raise ValueError("Invalid or duplicate dates")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Dates are not increasing")

    prices = pd.to_numeric(data["SPY"], errors="raise")
    prices = prices.dropna().astype(float)

    if prices.empty or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Invalid SPY prices")

    if (prices <= 0).any():
        raise ValueError("SPY prices must be positive")

    return prices.rename("SPY")


def load_spy_observations_through(
    path: str | Path,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return bounded session observations, including missing SPY prices."""
    return load_tpq_columns_through(path, ["SPY"], end_date)


def load_tpq_columns_through(
    path: str | Path,
    columns: list[str] | tuple[str, ...],
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Read requested prices through the cutoff from ordered daily TPQ records."""
    requested = list(columns)
    if not requested or "Date" in requested:
        raise ValueError("Request one or more price columns")
    if isinstance(end_date, str):
        cutoff = date.fromisoformat(end_date)
        if cutoff.isoformat() != end_date:
            raise ValueError("Invalid TPQ end date")
    elif isinstance(end_date, pd.Timestamp) and not pd.isna(end_date):
        cutoff = end_date.date()
    else:
        raise ValueError("Invalid TPQ end date")

    dates = []
    rows = []
    try:
        with Path(path).open("rb", buffering=0) as source:
            header = _read_csv_fields(source)
            if len(set(header)) != len(header):
                raise ValueError("Duplicate TPQ source headers")
            if header[:1] != ["Date"] or not set(requested).issubset(header):
                raise ValueError("Missing requested TPQ columns or leading Date")
            positions = [header.index(name) - 1 for name in requested]
            while (day := _read_tpq_date(source)) is not None:
                if dates and day <= dates[-1]:
                    raise ValueError("TPQ dates are not strictly increasing")
                if day > cutoff:
                    break
                fields = _read_csv_fields(source)
                # A trailing date delimiter alone represents one empty price.
                if not fields:
                    fields = [""]
                if len(fields) != len(header) - 1:
                    raise ValueError("Invalid TPQ row width")
                rows.append([fields[i].strip() for i in positions])
                dates.append(day)
                if day == cutoff:
                    break
    except OSError as exc:
        raise ValueError("Unable to read TPQ price file") from exc

    if not dates:
        raise ValueError("No TPQ rows through requested date")
    data = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"), columns=requested)
    data = data.apply(pd.to_numeric, errors="raise").astype(float)
    if data.isna().all().any():
        raise ValueError("Entirely blank requested TPQ series")
    if np.isinf(data.to_numpy()).any() or (data <= 0).any().any():
        raise ValueError("TPQ prices must be finite and positive")
    return data


def load_spy_prices_through(
    path: str | Path,
    end_date: str | pd.Timestamp,
) -> pd.Series:
    """Load valid SPY close observations only through an approved date."""
    return load_spy_observations_through(path, end_date)["SPY"].dropna()


def calculate_returns(prices: pd.Series) -> pd.DataFrame:
    """Calculate returns between consecutive valid sessions."""
    simple = prices.pct_change()
    log = np.log(prices / prices.shift(1))

    return pd.DataFrame({
        "return_simple": simple,
        "return_log": log,
    }).dropna()
