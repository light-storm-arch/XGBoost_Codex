"""Shared configuration dataclasses for the ETF allocation app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ModelSettings:
    """Model-level settings for training and inference."""

    horizons: Tuple[int, ...] = (1, 3, 6, 12)
    xgb_objective: str = "binary:logistic"
    xgb_eval_metric: str = "logloss"
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 4
    xgb_n_estimators: int = 250
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_random_state: int = 42


@dataclass(frozen=True)
class BacktestSettings:
    """Walk-forward and portfolio simulation settings."""

    train_months: int = 60
    test_months: int = 3
    step_months: int = 1
    rebalance_frequency: str = "monthly"
    transaction_cost_bps: float = 2.0


@dataclass(frozen=True)
class SignalBucketThresholds:
    """Probability thresholds that map model outputs to allocation buckets."""

    risk_off: float = 0.40
    neutral: float = 0.55
    risk_on: float = 0.70

    def as_ordered(self) -> Tuple[float, float, float]:
        """Return ordered thresholds for downstream validation."""
        return (self.risk_off, self.neutral, self.risk_on)


@dataclass(frozen=True)
class AppSettings:
    """Top-level settings bundle used by pipeline orchestration."""

    etf_pairs: Tuple[Tuple[str, str], ...] = (("VTV", "VUG"), ("VTI", "VXUS"))
    model: ModelSettings = field(default_factory=ModelSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    buckets: SignalBucketThresholds = field(default_factory=SignalBucketThresholds)
