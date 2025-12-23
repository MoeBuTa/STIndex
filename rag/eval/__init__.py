"""
RAG Evaluation Module

QA evaluation using RRF-4 retrieval and LLM-based answer generation.
"""

from rag.eval.qa_metrics import (
    QAMetrics,
    normalize_answer,
    tokenize,
    compute_qa_metrics,
)

from rag.eval.qa_evaluation import (
    QAEvaluator,
    run_qa_evaluation,
    QuestionType,
)

__all__ = [
    # Metrics
    "QAMetrics",
    "normalize_answer",
    "tokenize",
    "compute_qa_metrics",
    # Evaluation
    "QAEvaluator",
    "run_qa_evaluation",
    "QuestionType",
]
