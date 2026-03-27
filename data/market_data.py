"""Utilities for downloading and transforming market ETF data."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import yfinance as yf

DEFAULT_ETFS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EFA")


class MarketDataError(ValueError):
    """Raised when market data cannot be transformed safely."""


def _validate_datetime_index(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise MarketDataError(f"{name} must use a DatetimeIndex.")
    if frame.index.has_duplicates:
        dupes = frame.index[frame.index.duplicated()].unique()
        raise MarketDataError(f"{name} contains duplicate timestamps: {dupes.tolist()}")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataError(f"{name} index must be sorted in ascending order.")


def download_adjusted_close_prices(
    tickers: Iterable[str] = DEFAULT_ETFS,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted-close prices for a list of ETFs via yfinance."""
    tickers = tuple(dict.fromkeys(tickers))
    if not tickers:
        raise MarketDataError("At least one ticker is required.")

    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="column",
    )

    if data.empty:
        raise MarketDataError("No market data returned by yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise MarketDataError("yfinance output does not include 'Adj Close'.")
        adj_close = data["Adj Close"].copy()
    else:
        adj_close = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    adj_close.index = pd.DatetimeIndex(adj_close.index).tz_localize(None)
    adj_close = adj_close.sort_index().dropna(how="all")
    _validate_datetime_index(adj_close, name="Adjusted-close prices")
    return adj_close


def resample_to_monthly_end(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end observations."""
    _validate_datetime_index(prices, name="Daily prices")
    monthly = prices.resample("M").last().dropna(how="all")
    _validate_datetime_index(monthly, name="Monthly prices")
    return monthly


def compute_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple monthly returns from month-end prices."""
    _validate_datetime_index(monthly_prices, name="Monthly prices")
    returns = monthly_prices.pct_change().dropna(how="all")
    _validate_datetime_index(returns, name="Monthly returns")
    return returns


def build_market_monthly_returns(
    tickers: Sequence[str] = DEFAULT_ETFS,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper that downloads prices and returns monthly ETF returns."""
    prices = download_adjusted_close_prices(tickers=tickers, start=start, end=end)
    monthly_prices = resample_to_monthly_end(prices)
    return compute_monthly_returns(monthly_prices)
