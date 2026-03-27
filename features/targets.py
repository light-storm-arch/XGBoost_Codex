"""Target generation and dataset alignment utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .feature_builder import PairConfig

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 6, 12)


def forward_return(price: pd.Series, horizon: int) -> pd.Series:
    """Compute forward return over `horizon` periods."""

    return price.shift(-horizon) / price - 1.0


def build_pair_targets(
    price_df: pd.DataFrame,
    pair: PairConfig,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build pair-relative forward targets for a single pair.

    For each horizon h, creates:
      target_h = fwd_ret(etf_a, h) - fwd_ret(etf_b, h)
    """

    missing = [col for col in (pair.etf_a, pair.etf_b) if col not in price_df.columns]
    if missing:
        raise KeyError(f"Missing required price columns for target creation: {missing}")

    out = pd.DataFrame(index=price_df.index)
    for h in horizons:
        target_name = f"{pair.pair_name}__target_{h}"
        out[target_name] = forward_return(price_df[pair.etf_a], h) - forward_return(
            price_df[pair.etf_b], h
        )

    return out


def build_targets(
    price_df: pd.DataFrame,
    pairs: Sequence[PairConfig],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build pair-relative forward targets across all provided pairs."""

    parts = [build_pair_targets(price_df=price_df, pair=pair, horizons=horizons) for pair in pairs]
    return pd.concat(parts, axis=1)


def align_features_and_targets(
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align features and targets at current timestamp with strict completeness.

    Drops rows where any feature is missing and rows where any target is incomplete,
    which naturally removes trailing rows with unavailable forward windows.
    """

    if not feature_df.index.equals(target_df.index):
        target_df = target_df.reindex(feature_df.index)

    merged = pd.concat({"feature": feature_df, "target": target_df}, axis=1)
    merged = merged.dropna(axis=0, how="any")

    aligned_features = merged["feature"].copy()
    aligned_targets = merged["target"].copy()
    return aligned_features, aligned_targets
