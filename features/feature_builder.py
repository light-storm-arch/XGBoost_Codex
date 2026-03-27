"""Feature engineering utilities for ETF-pair relative modeling.

This module builds two classes of features:
1) Pair-relative market features from ETF prices.
2) Macro-transformed features built strictly from lagged macro observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_MOMENTUM_WINDOWS: tuple[int, ...] = (1, 3, 6, 12)
DEFAULT_VOL_WINDOWS: tuple[int, ...] = (3, 6, 12)


@dataclass(frozen=True)
class PairConfig:
    """Configuration describing a tradable ETF pair."""

    etf_a: str
    etf_b: str
    name: str | None = None

    @property
    def pair_name(self) -> str:
        return self.name or f"{self.etf_a}__{self.etf_b}"


def _validate_price_columns(price_df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in price_df.columns]
    if missing:
        raise KeyError(f"Missing required price columns: {missing}")


def build_pair_relative_features(
    price_df: pd.DataFrame,
    pair: PairConfig,
    momentum_windows: Sequence[int] = DEFAULT_MOMENTUM_WINDOWS,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
) -> pd.DataFrame:
    """Create pair-relative features from ETF prices.

    Parameters
    ----------
    price_df:
        Price dataframe indexed by timestamp and containing columns for `pair.etf_a`
        and `pair.etf_b`.
    pair:
        ETF pair metadata.
    momentum_windows:
        Windows (in rows) for return momentum differential features.
    vol_windows:
        Rolling windows (in rows) for return volatility differential features.
    """

    _validate_price_columns(price_df, [pair.etf_a, pair.etf_b])

    out = pd.DataFrame(index=price_df.index)
    ret_a = price_df[pair.etf_a].pct_change()
    ret_b = price_df[pair.etf_b].pct_change()

    prefix = pair.pair_name
    out[f"{prefix}__spread_ret_1"] = ret_a - ret_b

    for window in momentum_windows:
        mom_a = price_df[pair.etf_a].pct_change(window)
        mom_b = price_df[pair.etf_b].pct_change(window)
        out[f"{prefix}__mom_diff_{window}"] = mom_a - mom_b

    for window in vol_windows:
        vol_a = ret_a.rolling(window=window, min_periods=window).std()
        vol_b = ret_b.rolling(window=window, min_periods=window).std()
        out[f"{prefix}__vol_diff_{window}"] = vol_a - vol_b

    return out


def build_macro_transformed_features(
    macro_df: pd.DataFrame,
    yoy_periods: int = 12,
    mom_periods: int = 1,
) -> pd.DataFrame:
    """Build macro transformations using only lagged macro inputs.

    The routine first lags all macro columns by one step to prevent using same-period
    information. It then computes:
      * YoY changes
      * MoM changes
      * term spread (long yield - short yield) when available
      * risk spread (corporate yield - treasury yield) when available
    """

    if macro_df.empty:
        return pd.DataFrame(index=macro_df.index)

    lagged = macro_df.shift(1)
    out = pd.DataFrame(index=macro_df.index)

    for col in lagged.columns:
        out[f"macro__{col}__mom_{mom_periods}"] = lagged[col].pct_change(mom_periods)
        out[f"macro__{col}__yoy_{yoy_periods}"] = lagged[col].pct_change(yoy_periods)

    # Heuristic spread features when matching columns exist.
    lower = {c.lower(): c for c in lagged.columns}

    long_key = next((k for k in lower if "10y" in k or "long" in k), None)
    short_key = next((k for k in lower if "2y" in k or "3m" in k or "short" in k), None)
    if long_key and short_key:
        long_col, short_col = lower[long_key], lower[short_key]
        out["macro__term_spread"] = lagged[long_col] - lagged[short_col]

    corp_key = next((k for k in lower if "corp" in k or "baa" in k), None)
    tsy_key = next((k for k in lower if "treasury" in k or "10y" in k), None)
    if corp_key and tsy_key:
        corp_col, tsy_col = lower[corp_key], lower[tsy_key]
        out["macro__risk_spread"] = lagged[corp_col] - lagged[tsy_col]

    return out


def build_feature_matrix(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    pairs: Sequence[PairConfig],
    momentum_windows: Sequence[int] = DEFAULT_MOMENTUM_WINDOWS,
    vol_windows: Sequence[int] = DEFAULT_VOL_WINDOWS,
) -> pd.DataFrame:
    """Build complete feature matrix for all pairs and macro transforms."""

    parts = [
        build_pair_relative_features(
            price_df=price_df,
            pair=pair,
            momentum_windows=momentum_windows,
            vol_windows=vol_windows,
        )
        for pair in pairs
    ]
    parts.append(build_macro_transformed_features(macro_df=macro_df))

    feature_df = pd.concat(parts, axis=1)
    return feature_df
