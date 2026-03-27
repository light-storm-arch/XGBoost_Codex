"""FRED data access and series-catalog utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from fredapi import Fred


class FredDataError(ValueError):
    """Raised when FRED data cannot be loaded or transformed safely."""


@dataclass(frozen=True)
class FredSeriesConfig:
    """Metadata for an individual FRED series."""

    series_id: str
    description: str
    native_frequency: str
    release_lag_months: int


DEFAULT_FRED_CATALOG: tuple[FredSeriesConfig, ...] = (
    FredSeriesConfig(
        series_id="UNRATE",
        description="Civilian unemployment rate",
        native_frequency="monthly",
        release_lag_months=1,
    ),
    FredSeriesConfig(
        series_id="CPIAUCSL",
        description="Consumer Price Index for All Urban Consumers",
        native_frequency="monthly",
        release_lag_months=1,
    ),
    FredSeriesConfig(
        series_id="INDPRO",
        description="Industrial production index",
        native_frequency="monthly",
        release_lag_months=1,
    ),
    FredSeriesConfig(
        series_id="GDPC1",
        description="Real gross domestic product",
        native_frequency="quarterly",
        release_lag_months=1,
    ),
)


def get_fred_client(api_key_env: str = "FRED_API_KEY") -> Fred:
    """Create a Fred client using an API key loaded from environment variables."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise FredDataError(f"Missing FRED API key in environment variable: {api_key_env}")
    return Fred(api_key=api_key)


def catalog_to_dataframe(catalog: Sequence[FredSeriesConfig]) -> pd.DataFrame:
    """Represent the configured catalog as a dataframe for inspection/logging."""
    rows = [
        {
            "series_id": item.series_id,
            "description": item.description,
            "native_frequency": item.native_frequency,
            "release_lag_months": item.release_lag_months,
        }
        for item in catalog
    ]
    return pd.DataFrame(rows)


def fetch_fred_series(
    catalog: Iterable[FredSeriesConfig] = DEFAULT_FRED_CATALOG,
    *,
    start: str | None = None,
    end: str | None = None,
    api_key_env: str = "FRED_API_KEY",
) -> pd.DataFrame:
    """Fetch configured FRED series and return a raw macro dataframe indexed by period end."""
    client = get_fred_client(api_key_env=api_key_env)
    series_frames: list[pd.Series] = []

    for config in catalog:
        values = client.get_series(config.series_id, observation_start=start, observation_end=end)
        if values.empty:
            continue

        series = values.rename(config.series_id)
        series.index = pd.to_datetime(series.index).to_period("M").to_timestamp("M")
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()
        series_frames.append(series)

    if not series_frames:
        raise FredDataError("No FRED series were returned for the configured catalog/date range.")

    macro = pd.concat(series_frames, axis=1).sort_index()
    if macro.index.has_duplicates:
        raise FredDataError("Macro dataframe contains duplicated dates.")
    if not macro.index.is_monotonic_increasing:
        raise FredDataError("Macro dataframe index is not monotonic increasing.")
    return macro
