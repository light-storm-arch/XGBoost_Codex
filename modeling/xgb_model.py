"""XGBoost model wrappers used by the backtesting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import xgboost as xgb


@dataclass
class XGBModelConfig:
    """Configuration for :class:`xgboost.XGBRegressor`."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42


class XGBRegressorWrapper:
    """Thin wrapper around :class:`xgboost.XGBRegressor`.

    The wrapper centralizes model hyperparameters and keeps construction
    and fitting logic consistent across pairs and forecast horizons.
    """

    def __init__(self, config: XGBModelConfig | None = None, **overrides: Any) -> None:
        self.config = config or XGBModelConfig()
        if overrides:
            for key, value in overrides.items():
                if not hasattr(self.config, key):
                    raise ValueError(f"Unknown XGBModelConfig field: {key}")
                setattr(self.config, key, value)
        self._model: xgb.XGBRegressor | None = None

    @property
    def model(self) -> xgb.XGBRegressor:
        """Return the fitted internal model instance."""
        if self._model is None:
            raise RuntimeError("Model has not been initialized. Call fit first.")
        return self._model

    def as_params(self) -> Mapping[str, Any]:
        """Serialize the current configuration into XGBRegressor kwargs."""
        return {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "min_child_weight": self.config.min_child_weight,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "random_state": self.config.random_state,
            "objective": "reg:squarederror",
        }

    def fit(self, X, y) -> "XGBRegressorWrapper":
        """Fit an internal ``XGBRegressor`` model and return ``self``."""
        self._model = xgb.XGBRegressor(**self.as_params())
        self._model.fit(X, y)
        return self

    def predict(self, X):
        """Generate predictions using the fitted internal model."""
        return self.model.predict(X)
