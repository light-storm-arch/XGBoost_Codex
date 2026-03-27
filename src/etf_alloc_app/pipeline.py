"""Pipeline orchestration for ETF allocation workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config.defaults import DEFAULT_SETTINGS
from .config.types import AppSettings


def run_pipeline(settings: Optional[AppSettings] = None) -> Dict[str, Any]:
    """Run a baseline orchestration flow and return summary metadata."""
    effective_settings = settings or DEFAULT_SETTINGS
    return {
        "status": "initialized",
        "etf_pairs": effective_settings.etf_pairs,
        "horizons": effective_settings.model.horizons,
        "walk_forward": {
            "train_months": effective_settings.backtest.train_months,
            "test_months": effective_settings.backtest.test_months,
            "step_months": effective_settings.backtest.step_months,
        },
    }
