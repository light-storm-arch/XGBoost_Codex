"""Evaluation metrics for out-of-sample forecast performance."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "date",
    "pair",
    "horizon_m",
    "y_pred",
    "y_true",
    "train_start",
    "train_end",
}


def validate_prediction_table(predictions: pd.DataFrame) -> None:
    """Validate that the OOS prediction table has the expected schema."""
    missing = REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"Missing required prediction columns: {missing_fields}")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - np.mean(y_true)
    sst = float(np.sum(centered**2))
    if sst == 0.0:
        return float("nan")
    sse = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (sse / sst)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _spread_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    positive_mask = y_pred > 0
    negative_mask = y_pred < 0

    spread_if_long = float(np.mean(y_true[positive_mask])) if np.any(positive_mask) else float("nan")
    spread_if_short = float(np.mean(y_true[negative_mask])) if np.any(negative_mask) else float("nan")

    signed_direction = np.sign(y_pred)
    realized_strategy_spread = float(np.mean(signed_direction * y_true))

    return spread_if_long, spread_if_short, realized_strategy_spread


def compute_group_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Compute OOS metrics for one pair/horizon slice."""
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)

    spread_if_long, spread_if_short, realized_strategy_spread = _spread_metrics(y_true, y_pred)

    return {
        "n_obs": int(len(group)),
        "rmse": _rmse(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
        "directional_accuracy": _directional_accuracy(y_true, y_pred),
        "avg_realized_spread_if_pred_pos": spread_if_long,
        "avg_realized_spread_if_pred_neg": spread_if_short,
        "avg_realized_spread_signed": realized_strategy_spread,
    }


def summarize_oos_metrics(
    predictions: pd.DataFrame,
    *,
    groupby_cols: Iterable[str] = ("pair", "horizon_m"),
) -> pd.DataFrame:
    """Aggregate OOS metrics by horizon (and optionally pair)."""
    validate_prediction_table(predictions)

    records: list[dict[str, float | int | str]] = []
    for keys, group in predictions.groupby(list(groupby_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, float | int | str] = {col: key for col, key in zip(groupby_cols, keys)}
        row.update(compute_group_metrics(group))
        records.append(row)

    return pd.DataFrame(records).sort_values(list(groupby_cols)).reset_index(drop=True)
