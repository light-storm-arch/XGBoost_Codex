"""Default configuration values for ETF allocation workflows."""

from __future__ import annotations

from .types import AppSettings

DEFAULT_SETTINGS = AppSettings()

DEFAULT_ETF_PAIRS = DEFAULT_SETTINGS.etf_pairs
DEFAULT_HORIZONS = DEFAULT_SETTINGS.model.horizons
DEFAULT_WALK_FORWARD = DEFAULT_SETTINGS.backtest
DEFAULT_XGBOOST_PARAMS = {
    "objective": DEFAULT_SETTINGS.model.xgb_objective,
    "eval_metric": DEFAULT_SETTINGS.model.xgb_eval_metric,
    "learning_rate": DEFAULT_SETTINGS.model.xgb_learning_rate,
    "max_depth": DEFAULT_SETTINGS.model.xgb_max_depth,
    "n_estimators": DEFAULT_SETTINGS.model.xgb_n_estimators,
    "subsample": DEFAULT_SETTINGS.model.xgb_subsample,
    "colsample_bytree": DEFAULT_SETTINGS.model.xgb_colsample_bytree,
    "reg_alpha": DEFAULT_SETTINGS.model.xgb_reg_alpha,
    "reg_lambda": DEFAULT_SETTINGS.model.xgb_reg_lambda,
    "random_state": DEFAULT_SETTINGS.model.xgb_random_state,
}
