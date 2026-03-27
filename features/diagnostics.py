"""Diagnostics for feature/target dataset quality checks."""

from __future__ import annotations

import pandas as pd


def feature_coverage_by_date(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return row-wise feature availability summary."""

    coverage = feature_df.notna().sum(axis=1)
    ratio = coverage / max(feature_df.shape[1], 1)
    return pd.DataFrame(
        {
            "available_feature_count": coverage,
            "available_feature_ratio": ratio,
        },
        index=feature_df.index,
    )


def na_counts(feature_df: pd.DataFrame) -> pd.Series:
    """Count NA values by feature column."""

    return feature_df.isna().sum().sort_values(ascending=False)


def first_valid_timestamp_per_feature(feature_df: pd.DataFrame) -> pd.Series:
    """Get first non-null timestamp for each feature."""

    return feature_df.apply(lambda s: s.first_valid_index())


def build_lag_audit_columns(
    feature_df: pd.DataFrame,
    lagged_macro_df: pd.DataFrame | None = None,
    macro_lag: int = 1,
) -> pd.DataFrame:
    """Build audit metadata columns for lag assumptions.

    Adds explicit markers that can be merged into modeling datasets.
    """

    audit = pd.DataFrame(index=feature_df.index)
    audit["audit__feature_timestamp"] = feature_df.index
    audit["audit__macro_lag_periods"] = macro_lag
    audit["audit__has_any_missing_feature"] = feature_df.isna().any(axis=1)

    if lagged_macro_df is not None:
        audit["audit__lagged_macro_complete"] = lagged_macro_df.notna().all(axis=1)

    return audit
