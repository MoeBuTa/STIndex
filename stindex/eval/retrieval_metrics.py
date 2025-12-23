"""
Retrieval metrics for information retrieval evaluation.

Implements standard IR metrics:
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)

Based on TREC and CLEF evaluation standards.
"""

from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""

    # Per-query metrics
    query_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Aggregate metrics
    total_queries: int = 0
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    def recall_at_k(self, k: int) -> float:
        """
        Calculate Recall@k: proportion of relevant docs retrieved in top-k.

        Recall@k = |relevant ∩ retrieved@k| / |relevant|

        Args:
            k: Cutoff rank

        Returns:
            Average recall@k across all queries
        """
        if self.total_queries == 0:
            return 0.0

        recalls = []
        for query_id, result in self.query_results.items():
            relevant = set(result.get("relevant_docs", []))
            retrieved = set(result.get(f"retrieved@{k}", []))

            if len(relevant) == 0:
                continue  # Skip queries with no relevant docs

            recall = len(relevant & retrieved) / len(relevant)
            recalls.append(recall)

        return np.mean(recalls) if recalls else 0.0

    def precision_at_k(self, k: int) -> float:
        """
        Calculate Precision@k: proportion of retrieved docs that are relevant in top-k.

        Precision@k = |relevant ∩ retrieved@k| / k

        Args:
            k: Cutoff rank

        Returns:
            Average precision@k across all queries
        """
        if self.total_queries == 0:
            return 0.0

        precisions = []
        for query_id, result in self.query_results.items():
            relevant = set(result.get("relevant_docs", []))
            retrieved = set(result.get(f"retrieved@{k}", []))

            if len(retrieved) == 0:
                precision = 0.0
            else:
                precision = len(relevant & retrieved) / len(retrieved)

            precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0

    def mean_reciprocal_rank(self) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = (1/|Q|) * sum(1 / rank_first_relevant)

        Returns:
            Mean reciprocal rank across all queries
        """
        if self.total_queries == 0:
            return 0.0

        reciprocal_ranks = []
        for query_id, result in self.query_results.items():
            relevant = set(result.get("relevant_docs", []))
            ranked_list = result.get("ranked_list", [])

            # Find rank of first relevant document
            first_relevant_rank = None
            for rank, doc_id in enumerate(ranked_list, start=1):
                if doc_id in relevant:
                    first_relevant_rank = rank
                    break

            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def ndcg_at_k(self, k: int, graded: bool = False) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).

        DCG@k = sum_{i=1}^k (rel_i / log2(i+1))
        NDCG@k = DCG@k / IDCG@k

        Args:
            k: Cutoff rank
            graded: If True, use graded relevance (0-3); if False, binary (0-1)

        Returns:
            Average NDCG@k across all queries
        """
        if self.total_queries == 0:
            return 0.0

        ndcgs = []
        for query_id, result in self.query_results.items():
            relevant = set(result.get("relevant_docs", []))
            ranked_list = result.get(f"retrieved@{k}", [])

            if not relevant:
                continue  # Skip queries with no relevant docs

            # Compute DCG
            dcg = 0.0
            for rank, doc_id in enumerate(ranked_list, start=1):
                if doc_id in relevant:
                    rel = 1.0  # Binary relevance
                    dcg += rel / np.log2(rank + 1)

            # Compute IDCG (ideal DCG)
            num_relevant = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

            # NDCG
            if idcg > 0:
                ndcg = dcg / idcg
                ndcgs.append(ndcg)

        return np.mean(ndcgs) if ndcgs else 0.0

    def add_query_result(
        self,
        query_id: str,
        relevant_docs: List[str],
        retrieved_docs: List[str],
    ):
        """
        Add query result for evaluation.

        Args:
            query_id: Query identifier
            relevant_docs: List of relevant document IDs (ground truth)
            retrieved_docs: List of retrieved document IDs (ranked list)
        """
        # Store full ranked list
        self.query_results[query_id] = {
            "relevant_docs": relevant_docs,
            "ranked_list": retrieved_docs,
        }

        # Store retrieved@k for each k
        for k in self.k_values:
            self.query_results[query_id][f"retrieved@{k}"] = retrieved_docs[:k]

        self.total_queries += 1

    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all retrieval metrics.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "mrr": round(self.mean_reciprocal_rank(), 4),
        }

        # Compute recall@k, precision@k, ndcg@k for each k
        for k in self.k_values:
            metrics[f"recall@{k}"] = round(self.recall_at_k(k), 4)
            metrics[f"precision@{k}"] = round(self.precision_at_k(k), 4)
            metrics[f"ndcg@{k}"] = round(self.ndcg_at_k(k), 4)

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            "metrics": self.compute_all_metrics(),
            "total_queries": self.total_queries,
            "k_values": self.k_values,
        }

    def meets_targets(
        self,
        target_recall_at_10: float = 0.70,
        target_precision_at_10: float = 0.60,
        target_mrr: float = 0.60,
    ) -> Dict[str, bool]:
        """
        Check if retrieval metrics meet target thresholds.

        Targets:
        - Recall@10: ≥0.70
        - Precision@10: ≥0.60
        - MRR: ≥0.60

        Args:
            target_recall_at_10: Target for recall@10
            target_precision_at_10: Target for precision@10
            target_mrr: Target for MRR

        Returns:
            Dictionary indicating which targets are met
        """
        return {
            "recall@10": self.recall_at_k(10) >= target_recall_at_10,
            "precision@10": self.precision_at_k(10) >= target_precision_at_10,
            "mrr": self.mean_reciprocal_rank() >= target_mrr,
        }


def compute_retrieval_metrics(
    queries: List[Dict[str, Any]],
    retrieval_results: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10, 20],
) -> RetrievalMetrics:
    """
    Compute retrieval metrics from queries and results.

    Args:
        queries: List of query dictionaries with 'query_id', 'query_text', 'relevant_docs'
        retrieval_results: Dict mapping query_id to list of retrieved doc_ids (ranked)
        k_values: List of k values to evaluate

    Returns:
        RetrievalMetrics object with computed metrics
    """
    metrics = RetrievalMetrics(k_values=k_values)

    for query in queries:
        query_id = query["query_id"]
        relevant_docs = query.get("relevant_docs", [])
        retrieved_docs = retrieval_results.get(query_id, [])

        metrics.add_query_result(query_id, relevant_docs, retrieved_docs)

    return metrics
