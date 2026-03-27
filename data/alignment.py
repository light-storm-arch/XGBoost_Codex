"""Lagging and alignment utilities for leak-safe monthly feature datasets."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from data.fred_data import FredSeriesConfig


class AlignmentError(ValueError):
    """Raised when feature alignment fails validation checks."""


def validate_time_index(frame: pd.DataFrame | pd.Series, *, name: str) -> None:
    """Validate duplicated timestamps and monotonic order."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise AlignmentError(f"{name} must use a DatetimeIndex.")
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique().tolist()
        raise AlignmentError(f"{name} contains duplicated dates: {dupes}")
    if not index.is_monotonic_increasing:
        raise AlignmentError(f"{name} index must be monotonic increasing.")


def shift_macro_series_by_release_lag(
    macro_raw: pd.DataFrame,
    series_catalog: Sequence[FredSeriesConfig],
) -> dict[str, pd.DataFrame]:
    """Shift macro observations to their first known month-end based on release lag."""
    validate_time_index(macro_raw, name="Raw macro data")
    lag_map = {item.series_id: item.release_lag_months for item in series_catalog}

    lagged: dict[str, pd.DataFrame] = {}
    for series_id in macro_raw.columns:
        if series_id not in lag_map:
            continue

        values = macro_raw[series_id].dropna().sort_index()
        release_dates = values.index + MonthEnd(lag_map[series_id])

        frame = pd.DataFrame(
            {
                "available_date": release_dates,
                "source_date": values.index,
                "value": values.to_numpy(),
            }
        ).sort_values("available_date")
        frame = frame.drop_duplicates(subset=["available_date"], keep="last")
        lagged[series_id] = frame

    return lagged


def _asof_align_single_series(
    target_index: pd.DatetimeIndex,
    lagged_series: pd.DataFrame,
    *,
    series_name: str,
) -> pd.DataFrame:
    """Asof-merge one lagged series into the target monthly index."""
    target = pd.DataFrame(index=target_index).reset_index(names="feature_date")

    right = lagged_series[["available_date", "source_date", "value"]].copy()
    right = right.sort_values("available_date")

    merged = pd.merge_asof(
        target.sort_values("feature_date"),
        right,
        left_on="feature_date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.set_index("feature_date")
    merged = merged.rename(columns={"value": series_name})
    return merged


def validate_no_forward_fill_leakage(
    aligned_with_lineage: pd.DataFrame,
    *,
    value_columns: Iterable[str],
) -> None:
    """Ensure each feature timestamp only uses data released at or before that timestamp."""
    for column in value_columns:
        source_col = f"{column}__source_date"
        available_col = f"{column}__available_date"
        if source_col not in aligned_with_lineage or available_col not in aligned_with_lineage:
            raise AlignmentError(f"Missing lineage columns for {column}.")

        available = aligned_with_lineage[available_col]
        feature_idx = aligned_with_lineage.index.to_series(index=aligned_with_lineage.index)

        invalid = (available.notna()) & (available > feature_idx)
        if invalid.any():
            raise AlignmentError(
                f"Forward-fill leakage detected in {column}: {int(invalid.sum())} rows use future releases."
            )


def align_monthly_features(
    market_monthly_features: pd.DataFrame,
    macro_raw: pd.DataFrame,
    series_catalog: Sequence[FredSeriesConfig],
) -> pd.DataFrame:
    """Return a leak-safe aligned monthly dataframe for feature engineering."""
    validate_time_index(market_monthly_features, name="Market monthly features")
    validate_time_index(macro_raw, name="Raw macro data")

    target_index = market_monthly_features.index
    lagged_map = shift_macro_series_by_release_lag(macro_raw, series_catalog)

    aligned = market_monthly_features.copy()
    lineage = pd.DataFrame(index=target_index)

    for series_id, lagged_frame in lagged_map.items():
        merged = _asof_align_single_series(target_index, lagged_frame, series_name=series_id)
        aligned[series_id] = merged[series_id]
        lineage[f"{series_id}__source_date"] = merged["source_date"]
        lineage[f"{series_id}__available_date"] = merged["available_date"]

    validate_time_index(aligned, name="Aligned feature dataset")
    validate_no_forward_fill_leakage(aligned_with_lineage=lineage, value_columns=lagged_map.keys())
    return aligned
