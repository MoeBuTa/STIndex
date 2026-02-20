"""
STIndex Evaluation Framework

Comprehensive evaluation metrics and tools for spatiotemporal extraction.

Three evaluation dimensions:
1. Efficiency: Timing overhead, throughput, latency
2. Retrieval: Recall@k, Precision@k, MRR, NDCG
3. QA: Exact match, Token F1, Answer coverage
"""

# Extraction quality metrics (temporal, spatial)
from stindex.eval.metrics import (
    TemporalMetrics,
    SpatialMetrics,
    OverallMetrics,
    calculate_temporal_match,
    calculate_spatial_match,
)

# Efficiency metrics
from stindex.eval.efficiency_metrics import EfficiencyMetrics
from stindex.eval.efficiency_evaluation import run_efficiency_evaluation

# Retrieval metrics
from stindex.eval.retrieval_metrics import RetrievalMetrics, compute_retrieval_metrics

# QA metrics
from stindex.eval.qa_metrics import QAMetrics, compute_qa_metrics

__all__ = [
    # Extraction metrics
    "TemporalMetrics",
    "SpatialMetrics",
    "OverallMetrics",
    "calculate_temporal_match",
    "calculate_spatial_match",
    # Efficiency
    "EfficiencyMetrics",
    "run_efficiency_evaluation",
    # Retrieval
    "RetrievalMetrics",
    "compute_retrieval_metrics",
    # QA
    "QAMetrics",
    "compute_qa_metrics",
]
