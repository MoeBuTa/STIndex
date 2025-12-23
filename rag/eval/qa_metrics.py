"""
QA evaluation metrics for question answering systems.

Implements standard QA metrics:
- Exact Match (EM)
- Token F1 Score
- Answer Coverage (% answer tokens in context)
- Retrieval-QA Correlation

Based on SQuAD, NaturalQuestions, and biomedical QA benchmarks.
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import re
from collections import Counter
import string


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace

    Args:
        text: Answer text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for F1 computation.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    return normalize_answer(text).split()


@dataclass
class QAMetrics:
    """Metrics for QA evaluation"""

    # Per-question results
    question_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Aggregate counts
    total_questions: int = 0
    exact_match_count: int = 0

    def exact_match(self, predicted: str, gold: str) -> bool:
        """
        Check if predicted answer exactly matches gold answer.

        Comparison is done after normalization (lowercase, remove punctuation, etc.).

        Args:
            predicted: Predicted answer
            gold: Gold standard answer

        Returns:
            True if exact match, False otherwise
        """
        return normalize_answer(predicted) == normalize_answer(gold)

    def token_f1(self, predicted: str, gold: str) -> Dict[str, float]:
        """
        Compute token-level F1 score.

        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            predicted: Predicted answer
            gold: Gold standard answer

        Returns:
            Dictionary with precision, recall, f1
        """
        pred_tokens = tokenize(predicted)
        gold_tokens = tokenize(gold)

        # Handle empty predictions
        if len(pred_tokens) == 0:
            if len(gold_tokens) == 0:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            else:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if len(gold_tokens) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Count token overlaps
        pred_counter = Counter(pred_tokens)
        gold_counter = Counter(gold_tokens)

        # True positives
        tp = sum((pred_counter & gold_counter).values())

        # Precision and recall
        precision = tp / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = tp / len(gold_tokens) if len(gold_tokens) > 0 else 0.0

        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def answer_coverage(self, answer: str, context: str) -> float:
        """
        Compute answer coverage: percentage of answer tokens in context.

        Coverage = |answer_tokens ∩ context_tokens| / |answer_tokens|

        Args:
            answer: Answer text
            context: Retrieved context text

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        answer_tokens = set(tokenize(answer))
        context_tokens = set(tokenize(context))

        if len(answer_tokens) == 0:
            return 1.0  # Empty answer is trivially covered

        overlap = len(answer_tokens & context_tokens)
        coverage = overlap / len(answer_tokens)

        return coverage

    def add_question_result(
        self,
        question_id: str,
        predicted_answer: str,
        gold_answer: str,
        retrieved_context: Optional[str] = None,
        retrieval_rank: Optional[int] = None,
    ):
        """
        Add question result for evaluation.

        Args:
            question_id: Question identifier
            predicted_answer: Predicted answer
            gold_answer: Gold standard answer
            retrieved_context: Retrieved context (optional, for coverage)
            retrieval_rank: Rank of relevant document (optional, for correlation)
        """
        # Compute metrics
        em = self.exact_match(predicted_answer, gold_answer)
        f1_scores = self.token_f1(predicted_answer, gold_answer)

        coverage = None
        if retrieved_context:
            coverage = self.answer_coverage(gold_answer, retrieved_context)

        # Store result
        self.question_results[question_id] = {
            "predicted": predicted_answer,
            "gold": gold_answer,
            "exact_match": em,
            "token_f1": f1_scores["f1"],
            "token_precision": f1_scores["precision"],
            "token_recall": f1_scores["recall"],
            "answer_coverage": coverage,
            "retrieval_rank": retrieval_rank,
        }

        # Update counts
        self.total_questions += 1
        if em:
            self.exact_match_count += 1

    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all questions.

        Returns:
            Dictionary with aggregate metrics
        """
        if self.total_questions == 0:
            return {
                "exact_match": 0.0,
                "token_f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "answer_coverage": 0.0,
            }

        # Exact match
        em = self.exact_match_count / self.total_questions

        # Token F1 (average over questions)
        f1_scores = [r["token_f1"] for r in self.question_results.values()]
        precision_scores = [r["token_precision"] for r in self.question_results.values()]
        recall_scores = [r["token_recall"] for r in self.question_results.values()]

        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)

        # Answer coverage (only for questions with context)
        coverage_scores = [
            r["answer_coverage"]
            for r in self.question_results.values()
            if r["answer_coverage"] is not None
        ]
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0

        return {
            "exact_match": round(em, 4),
            "token_f1": round(avg_f1, 4),
            "token_precision": round(avg_precision, 4),
            "token_recall": round(avg_recall, 4),
            "answer_coverage": round(avg_coverage, 4),
        }

    def retrieval_qa_correlation(self) -> Optional[float]:
        """
        Compute Pearson correlation between retrieval quality and QA performance.

        Correlation between:
        - X: Retrieval rank (lower is better)
        - Y: QA accuracy (F1 score)

        A negative correlation indicates that better retrieval (lower rank)
        leads to better QA performance (higher F1).

        Returns:
            Pearson correlation coefficient, or None if insufficient data
        """
        # Extract data
        data = [
            (r["retrieval_rank"], r["token_f1"])
            for r in self.question_results.values()
            if r["retrieval_rank"] is not None
        ]

        if len(data) < 2:
            return None

        ranks = np.array([d[0] for d in data])
        f1_scores = np.array([d[1] for d in data])

        # Compute Pearson correlation
        correlation = np.corrcoef(ranks, f1_scores)[0, 1]

        return float(correlation) if not np.isnan(correlation) else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        metrics = self.compute_aggregate_metrics()

        correlation = self.retrieval_qa_correlation()
        if correlation is not None:
            metrics["retrieval_qa_correlation"] = round(correlation, 4)

        return {
            "metrics": metrics,
            "total_questions": self.total_questions,
        }

    def meets_targets(
        self,
        target_exact_match: float = 0.40,
        target_token_f1: float = 0.55,
        target_answer_coverage: float = 0.80,
    ) -> Dict[str, bool]:
        """
        Check if QA metrics meet target thresholds.

        Targets:
        - Exact Match: ≥0.40
        - Token F1: ≥0.55
        - Answer Coverage: ≥0.80

        Args:
            target_exact_match: Target for exact match
            target_token_f1: Target for token F1
            target_answer_coverage: Target for answer coverage

        Returns:
            Dictionary indicating which targets are met
        """
        metrics = self.compute_aggregate_metrics()

        return {
            "exact_match": bool(metrics["exact_match"] >= target_exact_match),
            "token_f1": bool(metrics["token_f1"] >= target_token_f1),
            "answer_coverage": bool(metrics["answer_coverage"] >= target_answer_coverage),
        }


def compute_qa_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> QAMetrics:
    """
    Compute QA metrics from predictions and ground truth.

    Args:
        predictions: List of prediction dicts with 'question_id', 'predicted_answer', 'context', 'rank'
        ground_truth: List of ground truth dicts with 'question_id', 'gold_answer'

    Returns:
        QAMetrics object with computed metrics
    """
    metrics = QAMetrics()

    # Create lookup for ground truth
    gt_lookup = {item["question_id"]: item["gold_answer"] for item in ground_truth}

    # Process predictions
    for pred in predictions:
        question_id = pred["question_id"]
        gold_answer = gt_lookup.get(question_id)

        if gold_answer is None:
            continue  # Skip if no ground truth

        metrics.add_question_result(
            question_id=question_id,
            predicted_answer=pred.get("predicted_answer", ""),
            gold_answer=gold_answer,
            retrieved_context=pred.get("context"),
            retrieval_rank=pred.get("rank"),
        )

    return metrics
