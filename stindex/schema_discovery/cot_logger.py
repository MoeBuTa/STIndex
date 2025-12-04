"""
Chain-of-Thought (CoT) reasoning logger for schema discovery.

Saves LLM reasoning and raw responses to files for debugging and analysis.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger


class CoTLogger:
    """
    Centralized Chain-of-Thought reasoning storage.

    Saves reasoning traces to files for:
    - Global schema discovery
    - Per-cluster entity extraction batches
    - Final summary statistics
    """

    def __init__(self, output_dir: str):
        """
        Initialize CoT logger.

        Args:
            output_dir: Base directory for schema discovery outputs
        """
        self.output_dir = Path(output_dir)
        self.cot_dir = self.output_dir / "cot"

        # Create CoT directories
        self.cot_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.stats = {
            "global_discovery": {
                "has_reasoning": False,
                "reasoning_length": 0
            },
            "clusters": {}
        }

        logger.debug(f"CoT logger initialized: {self.cot_dir}")

    def log_global_discovery(
        self,
        reasoning: str,
        raw_response: str
    ):
        """
        Log global schema discovery reasoning.

        Args:
            reasoning: Extracted CoT reasoning
            raw_response: Raw LLM response
        """
        # Create global discovery directory
        global_dir = self.cot_dir / "global_discovery"
        global_dir.mkdir(parents=True, exist_ok=True)

        # Save reasoning
        reasoning_file = global_dir / "reasoning_global.txt"
        reasoning_file.write_text(reasoning if reasoning else "(no reasoning extracted)")

        # Save raw response
        raw_file = global_dir / "raw_response_global.txt"
        raw_file.write_text(raw_response)

        # Update stats
        self.stats["global_discovery"]["has_reasoning"] = bool(reasoning)
        self.stats["global_discovery"]["reasoning_length"] = len(reasoning) if reasoning else 0

        logger.debug(f"✓ Saved global discovery CoT (reasoning: {len(reasoning)} chars)")

    def log_cluster_batch(
        self,
        cluster_id: int,
        batch_idx: int,
        reasoning: str,
        raw_response: str,
        n_entities: int = 0
    ):
        """
        Log cluster batch extraction reasoning.

        Args:
            cluster_id: Cluster identifier
            batch_idx: Batch index within cluster
            reasoning: Extracted CoT reasoning
            raw_response: Raw LLM response
            n_entities: Number of entities extracted
        """
        # Create cluster directory
        cluster_dir = self.cot_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Save reasoning
        reasoning_file = cluster_dir / f"batch_{batch_idx:03d}_reasoning.txt"
        reasoning_file.write_text(reasoning if reasoning else "(no reasoning extracted)")

        # Save raw response
        raw_file = cluster_dir / f"batch_{batch_idx:03d}_raw.txt"
        raw_file.write_text(raw_response)

        # Update stats
        if cluster_id not in self.stats["clusters"]:
            self.stats["clusters"][cluster_id] = {
                "batches": 0,
                "batches_with_reasoning": 0,
                "total_reasoning_length": 0,
                "total_entities": 0
            }

        cluster_stats = self.stats["clusters"][cluster_id]
        cluster_stats["batches"] += 1
        if reasoning:
            cluster_stats["batches_with_reasoning"] += 1
            cluster_stats["total_reasoning_length"] += len(reasoning)
        cluster_stats["total_entities"] += n_entities

        logger.debug(
            f"✓ Saved Cluster {cluster_id} Batch {batch_idx} CoT "
            f"(reasoning: {len(reasoning)} chars, entities: {n_entities})"
        )

    def save_final_summary(self):
        """
        Save final summary statistics to JSON.
        """
        summary_file = self.cot_dir / "reasoning_summary.json"

        # Calculate aggregate statistics
        summary = {
            "global_discovery": self.stats["global_discovery"],
            "clusters": {}
        }

        total_batches = 0
        total_batches_with_reasoning = 0
        total_reasoning_chars = 0

        for cluster_id, cluster_stats in self.stats["clusters"].items():
            batches = cluster_stats["batches"]
            batches_with_reasoning = cluster_stats["batches_with_reasoning"]
            total_reasoning = cluster_stats["total_reasoning_length"]

            total_batches += batches
            total_batches_with_reasoning += batches_with_reasoning
            total_reasoning_chars += total_reasoning

            # Calculate cluster-level metrics
            avg_reasoning_length = (
                total_reasoning / batches_with_reasoning
                if batches_with_reasoning > 0 else 0
            )

            summary["clusters"][str(cluster_id)] = {
                "batches": batches,
                "batches_with_reasoning": batches_with_reasoning,
                "reasoning_percentage": (
                    (batches_with_reasoning / batches * 100)
                    if batches > 0 else 0
                ),
                "avg_reasoning_length": int(avg_reasoning_length),
                "total_entities": cluster_stats["total_entities"]
            }

        # Overall statistics
        summary["overall"] = {
            "total_batches": total_batches,
            "batches_with_reasoning": total_batches_with_reasoning,
            "reasoning_percentage": (
                (total_batches_with_reasoning / total_batches * 100)
                if total_batches > 0 else 0
            ),
            "avg_reasoning_length": (
                int(total_reasoning_chars / total_batches_with_reasoning)
                if total_batches_with_reasoning > 0 else 0
            )
        }

        # Save to JSON
        summary_file.write_text(json.dumps(summary, indent=2))

        logger.info(f"✓ Saved CoT reasoning summary: {summary_file}")
        logger.info(
            f"  • {total_batches_with_reasoning}/{total_batches} batches "
            f"({summary['overall']['reasoning_percentage']:.1f}%) have reasoning"
        )
        logger.info(
            f"  • Average reasoning length: {summary['overall']['avg_reasoning_length']} chars"
        )

        return summary
