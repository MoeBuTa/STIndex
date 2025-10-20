"""
STIndex Evaluation Framework

Comprehensive evaluation metrics and tools for spatiotemporal extraction.
"""

from stindex.eval.metrics import (
    TemporalMetrics,
    SpatialMetrics,
    OverallMetrics,
    calculate_temporal_match,
    calculate_spatial_match,
)

__all__ = [
    "TemporalMetrics",
    "SpatialMetrics",
    "OverallMetrics",
    "calculate_temporal_match",
    "calculate_spatial_match",
]
