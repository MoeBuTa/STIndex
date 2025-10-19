"""
STIndex Evaluation Framework

Comprehensive evaluation metrics and tools for spatiotemporal extraction.
"""

from eval.metrics import (
    TemporalMetrics,
    SpatialMetrics,
    OverallMetrics,
    calculate_temporal_match,
    calculate_spatial_match,
)
from eval.evaluation import STIndexEvaluator, run_evaluation

__all__ = [
    "TemporalMetrics",
    "SpatialMetrics",
    "OverallMetrics",
    "STIndexEvaluator",
    "run_evaluation",
    "calculate_temporal_match",
    "calculate_spatial_match",
]
