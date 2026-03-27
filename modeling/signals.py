"""Signal construction helpers for ETF pair allocation.

This module maps model-predicted spread values into transparent, interpretable
allocation signals.

Caveat:
    The outputs here are model signals, not probability-calibrated forecasts.
    Treat `confidence_proxy` only as a heuristic unless explicit calibration has
    been implemented and validated.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable


DEFAULT_SIGNAL_THRESHOLDS_BPS: dict[str, float] = {
    "slight": 5.0,
    "moderate": 15.0,
    "strong": 30.0,
}


@dataclass(frozen=True)
class SignalResult:
    """Structured signal output for an ETF pair trade.

    Attributes:
        preferred_etf: ETF to prefer given the sign of the predicted spread.
        predicted_excess_return: Raw model-predicted spread (decimal return).
        signal_bucket: Bucketed signal strength based on absolute spread.
        confidence_proxy: Optional normalized percentile of absolute spread.
    """

    preferred_etf: str
    predicted_excess_return: float
    signal_bucket: str
    confidence_proxy: float | None = None


def generate_etf_signal(
    pred_spread: float,
    etf_a: str,
    etf_b: str,
    thresholds_bps: dict[str, float] | None = None,
    abs_spread_reference: Iterable[float] | None = None,
) -> SignalResult:
    """Create a structured ETF preference signal from a predicted spread.

    Args:
        pred_spread: Predicted excess return for ETF A minus ETF B, in decimal
            return units (e.g., 0.0012 == 12 bps).
        etf_a: ETF preferred when `pred_spread > 0`.
        etf_b: ETF preferred when `pred_spread <= 0`.
        thresholds_bps: Optional user-configurable bucket thresholds in bps with
            keys: ``slight``, ``moderate``, and ``strong``.
        abs_spread_reference: Optional historical absolute predicted spreads used
            to compute a normalized percentile confidence proxy.

    Returns:
        A ``SignalResult`` containing preferred ETF, raw predicted excess return,
        bucketed signal strength, and optional confidence proxy.
    """

    active_thresholds = thresholds_bps or DEFAULT_SIGNAL_THRESHOLDS_BPS
    _validate_thresholds(active_thresholds)

    preferred_etf = etf_a if pred_spread > 0 else etf_b
    abs_spread = abs(pred_spread)
    signal_bucket = _bucket_signal(abs_spread, active_thresholds)

    confidence_proxy = None
    if abs_spread_reference is not None:
        confidence_proxy = _percentile_proxy(abs_spread, abs_spread_reference)

    return SignalResult(
        preferred_etf=preferred_etf,
        predicted_excess_return=pred_spread,
        signal_bucket=signal_bucket,
        confidence_proxy=confidence_proxy,
    )


def _validate_thresholds(thresholds_bps: dict[str, float]) -> None:
    required = ("slight", "moderate", "strong")
    missing = [key for key in required if key not in thresholds_bps]
    if missing:
        raise ValueError(f"Missing threshold keys: {missing}")

    slight = thresholds_bps["slight"]
    moderate = thresholds_bps["moderate"]
    strong = thresholds_bps["strong"]
    if not (0 <= slight <= moderate <= strong):
        raise ValueError(
            "Thresholds must satisfy 0 <= slight <= moderate <= strong (in bps)."
        )


def _bucket_signal(abs_spread: float, thresholds_bps: dict[str, float]) -> str:
    slight = thresholds_bps["slight"] / 10_000.0
    moderate = thresholds_bps["moderate"] / 10_000.0
    strong = thresholds_bps["strong"] / 10_000.0

    if abs_spread < slight:
        return "flat"
    if abs_spread < moderate:
        return "slight_overweight"
    if abs_spread < strong:
        return "moderate_overweight"
    return "strong_overweight"


def _percentile_proxy(abs_spread: float, reference_values: Iterable[float]) -> float:
    values = sorted(abs(v) for v in reference_values)
    if not values:
        raise ValueError("abs_spread_reference must contain at least one value.")

    rank = bisect_right(values, abs_spread)
    return rank / len(values)
