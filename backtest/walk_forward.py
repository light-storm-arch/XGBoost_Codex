"""Walk-forward training utilities for rolling/expanding backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import compute_group_metrics
from modeling.xgb_model import XGBModelConfig, XGBRegressorWrapper


WindowType = Literal["rolling", "expanding"]


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtesting."""

    train_window_length: int
    test_window_length: int
    rebalance_frequency: int
    selected_horizons: Sequence[int]
    window_type: WindowType = "rolling"


def _split_indices(
    n_rows: int,
    *,
    train_window_length: int,
    test_window_length: int,
    rebalance_frequency: int,
    window_type: WindowType,
):
    if train_window_length <= 0 or test_window_length <= 0 or rebalance_frequency <= 0:
        raise ValueError("Window lengths and rebalance_frequency must be positive integers.")
    if n_rows < train_window_length + test_window_length:
        return

    train_start = 0
    train_end = train_window_length

    while train_end + test_window_length <= n_rows:
        test_start = train_end
        test_end = test_start + test_window_length
        yield train_start, train_end, test_start, test_end

        train_end += rebalance_frequency
        if window_type == "rolling":
            train_start += rebalance_frequency
        elif window_type == "expanding":
            train_start = 0
        else:
            raise ValueError("window_type must be either 'rolling' or 'expanding'.")


def _predict_pair_horizon(
    data: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    date_col: str,
    pair: str,
    horizon: int,
    config: WalkForwardConfig,
    model_config: XGBModelConfig,
) -> tuple[list[dict], list[dict]]:
    subset = data.loc[(data["pair"] == pair) & (data["horizon_m"] == horizon)].copy()
    subset = subset.sort_values(date_col).reset_index(drop=True)

    oos_records: list[dict] = []
    is_records: list[dict] = []

    for split_id, (train_start, train_end, test_start, test_end) in enumerate(
        _split_indices(
            len(subset),
            train_window_length=config.train_window_length,
            test_window_length=config.test_window_length,
            rebalance_frequency=config.rebalance_frequency,
            window_type=config.window_type,
        )
    ):
        train_df = subset.iloc[train_start:train_end]
        test_df = subset.iloc[test_start:test_end]

        model = XGBRegressorWrapper(config=model_config)
        model.fit(train_df.loc[:, feature_cols], train_df[target_col])

        y_pred_test = model.predict(test_df.loc[:, feature_cols])
        y_pred_train = model.predict(train_df.loc[:, feature_cols])

        train_start_date = train_df[date_col].iloc[0]
        train_end_date = train_df[date_col].iloc[-1]

        for row_idx, pred in zip(test_df.index, y_pred_test):
            oos_records.append(
                {
                    "date": subset.at[row_idx, date_col],
                    "pair": pair,
                    "horizon_m": horizon,
                    "y_pred": float(pred),
                    "y_true": float(subset.at[row_idx, target_col]),
                    "train_start": train_start_date,
                    "train_end": train_end_date,
                }
            )

        metrics = compute_group_metrics(
            pd.DataFrame({"y_true": train_df[target_col].to_numpy(), "y_pred": y_pred_train})
        )
        metrics.update(
            {
                "scope": "in_sample",
                "pair": pair,
                "horizon_m": horizon,
                "split_id": split_id,
                "train_start": train_start_date,
                "train_end": train_end_date,
            }
        )
        is_records.append(metrics)

    return oos_records, is_records


def run_walk_forward_backtest(
    data: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    config: WalkForwardConfig,
    model_config: XGBModelConfig | None = None,
    date_col: str = "date",
    pair_col: str = "pair",
    horizon_col: str = "horizon_m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtest and return OOS predictions + IS diagnostics.

    Returns:
        Tuple of ``(oos_predictions, in_sample_diagnostics)`` where:
        - ``oos_predictions`` includes columns:
          ``date, pair, horizon_m, y_pred, y_true, train_start, train_end``.
        - ``in_sample_diagnostics`` includes per-split diagnostics, explicitly
          labeled with ``scope='in_sample'`` to avoid conflating them with OOS
          performance.
    """
    required_cols = set(feature_cols) | {target_col, date_col, pair_col, horizon_col}
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {sorted(missing_cols)}")

    if config.window_type not in ("rolling", "expanding"):
        raise ValueError("config.window_type must be either 'rolling' or 'expanding'.")

    working = data.copy()
    if pair_col != "pair":
        working = working.rename(columns={pair_col: "pair"})
    if horizon_col != "horizon_m":
        working = working.rename(columns={horizon_col: "horizon_m"})

    horizons = set(config.selected_horizons)
    working = working[working["horizon_m"].isin(horizons)]

    if working.empty:
        empty_pred_cols = [
            "date",
            "pair",
            "horizon_m",
            "y_pred",
            "y_true",
            "train_start",
            "train_end",
        ]
        return pd.DataFrame(columns=empty_pred_cols), pd.DataFrame(columns=["scope"])

    model_config = model_config or XGBModelConfig()

    oos_all: list[dict] = []
    is_all: list[dict] = []

    for pair in sorted(working["pair"].dropna().unique()):
        pair_slice = working[working["pair"] == pair]
        for horizon in sorted(horizons):
            if pair_slice[pair_slice["horizon_m"] == horizon].empty:
                continue

            oos_records, is_records = _predict_pair_horizon(
                working,
                feature_cols=feature_cols,
                target_col=target_col,
                date_col=date_col,
                pair=pair,
                horizon=horizon,
                config=config,
                model_config=model_config,
            )
            oos_all.extend(oos_records)
            is_all.extend(is_records)

    oos_predictions = pd.DataFrame(oos_all)
    if not oos_predictions.empty:
        oos_predictions = oos_predictions.sort_values(["date", "pair", "horizon_m"]).reset_index(drop=True)
    else:
        oos_predictions = pd.DataFrame(
            columns=["date", "pair", "horizon_m", "y_pred", "y_true", "train_start", "train_end"]
        )

    in_sample_diagnostics = pd.DataFrame(is_all)
    if not in_sample_diagnostics.empty:
        in_sample_diagnostics = in_sample_diagnostics.sort_values(
            ["pair", "horizon_m", "split_id"]
        ).reset_index(drop=True)

    return oos_predictions, in_sample_diagnostics
