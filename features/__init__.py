"""Feature engineering package."""

from .diagnostics import (
    build_lag_audit_columns,
    feature_coverage_by_date,
    first_valid_timestamp_per_feature,
    na_counts,
)
from .feature_builder import (
    PairConfig,
    build_feature_matrix,
    build_macro_transformed_features,
    build_pair_relative_features,
)
from .targets import align_features_and_targets, build_pair_targets, build_targets, forward_return

__all__ = [
    "PairConfig",
    "build_pair_relative_features",
    "build_macro_transformed_features",
    "build_feature_matrix",
    "forward_return",
    "build_pair_targets",
    "build_targets",
    "align_features_and_targets",
    "feature_coverage_by_date",
    "na_counts",
    "first_valid_timestamp_per_feature",
    "build_lag_audit_columns",
]
