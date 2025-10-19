"""
Batch evaluation framework for STIndex using config-based initialization.

Supports:
- Config-driven LLM initialization (uses cfg/extract.yml and provider configs)
- Batch processing for efficiency
- Resume capability with JSON checkpointing
- Compatible with eval/evaluation.py metrics

Usage:
    python eval/batch_evaluation.py <dataset_path> [--config extract] [--batch-size 8]
"""

import json
import csv
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# Add parent directory to path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from stindex import STIndexExtractor
from stindex.agents.response.models import ExtractionResult
from stindex.agents.prompts.extraction import ExtractionPrompt
from eval.metrics import OverallMetrics

console = Console()


class BatchEvaluator:
    """Batch evaluator with resume capability"""

    def __init__(
        self,
        extractor: STIndexExtractor,
        output_dir: str = "data/output/eval_results",
        batch_size: int = 8,
    ):
        """
        Initialize batch evaluator.

        Args:
            extractor: STIndexExtractor instance (pre-configured)
            output_dir: Directory to save results
            batch_size: Number of samples per batch
        """
        self.extractor = extractor
        self.llm = extractor.llm
        self.geocoder = extractor.geocoder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        self.prompt_builder = ExtractionPrompt()
        self.metrics = OverallMetrics()
        self.results: List[Dict[str, Any]] = []

    def evaluate_dataset(
        self,
        dataset_path: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on dataset with batch processing.

        Args:
            dataset_path: Path to JSON dataset file
            resume: Whether to resume from existing results

        Returns:
            Dictionary with overall metrics
        """
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Load existing results if resuming
        existing_completions, evaluated_ids = self._load_existing_results()
        if resume and existing_completions:
            logger.info(f"Resuming evaluation: {len(evaluated_ids)} entries already processed")

        # Process in batches
        logger.info(f"Evaluating {len(dataset)} entries with batch_size={self.batch_size}")

        all_results = []
        for i in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch = dataset[i:i + self.batch_size]

            # Separate into cached and new samples
            cached_batch = []
            generate_batch = []
            batch_indices = []

            for idx, sample in enumerate(batch):
                sample_id = sample.get("id", f"sample_{i + idx}")
                if resume and sample_id in evaluated_ids:
                    cached_batch.append((idx, existing_completions[sample_id]))
                else:
                    generate_batch.append(sample)
                    batch_indices.append(idx)

            # Generate for new samples
            batch_results = []
            if generate_batch:
                batch_results = self._process_batch(generate_batch)

            # Merge cached and new results in original order
            merged_results = [None] * len(batch)
            for idx, result in cached_batch:
                merged_results[idx] = result
            for idx, result in zip(batch_indices, batch_results):
                merged_results[idx] = result

            all_results.extend(merged_results)

            # Save checkpoint after each batch
            if generate_batch:
                self._save_checkpoint(all_results)

        # Calculate final metrics
        self.results = all_results
        final_metrics = self._calculate_metrics()

        # Save final results
        self._save_detailed_results()
        self._save_metrics_summary(final_metrics)

        return final_metrics

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples.

        Args:
            batch: List of dataset entries

        Returns:
            List of evaluation results
        """
        # Build messages for batch
        messages_batch = []
        for sample in batch:
            messages = self.prompt_builder.build_messages(sample["text"])
            messages_batch.append(messages)

        # Batch extraction with LLM
        start_time = time.time()
        try:
            extraction_results = self.llm.generate_batch(
                messages_batch=messages_batch,
                response_model=ExtractionResult,
                max_tokens=2048,
            )
            processing_time = time.time() - start_time
            success = True
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            processing_time = time.time() - start_time
            extraction_results = [None] * len(batch)
            success = False

        # Post-process and geocode
        results = []
        for sample, extraction_result in zip(batch, extraction_results):
            if extraction_result:
                # Add character positions
                text = sample["text"]
                for entity in extraction_result.temporal_mentions:
                    if entity.text in text:
                        start_char = text.find(entity.text)
                        entity.start_char = start_char
                        entity.end_char = start_char + len(entity.text)

                for entity in extraction_result.spatial_mentions:
                    if entity.text in text:
                        start_char = text.find(entity.text)
                        entity.start_char = start_char
                        entity.end_char = start_char + len(entity.text)

                # Geocode spatial mentions
                for entity in extraction_result.spatial_mentions:
                    parent_region = getattr(entity, 'parent_region', None)
                    coords = self.geocoder.get_coordinates(
                        entity.text,
                        context=parent_region
                    )
                    if coords:
                        entity.latitude = coords[0]
                        entity.longitude = coords[1]

            # Create result entry
            result = {
                "id": sample.get("id", "unknown"),
                "text": sample["text"],
                "ground_truth": sample["ground_truth"],
                "extraction_result": extraction_result,
                "success": success and extraction_result is not None,
                "processing_time": processing_time / len(batch),
            }
            results.append(result)

        return results

    def _load_existing_results(self) -> tuple[Dict[int, Any], Set[int]]:
        """
        Load existing results from checkpoint.

        Returns:
            Tuple of (completions dict, set of evaluated IDs)
        """
        checkpoint_file = self.output_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return {}, set()

        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            completions = {r["id"]: r for r in results}
            evaluated_ids = set(completions.keys())

            return completions, evaluated_ids
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}, set()

    def _save_checkpoint(self, results: List[Dict[str, Any]]):
        """Save checkpoint of current results"""
        checkpoint_file = self.output_dir / "checkpoint.json"

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        logger.debug(f"Checkpoint saved: {len(results)} entries")

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics from results"""
        # Import evaluation logic from evaluation.py
        from eval.evaluation import STIndexEvaluator

        # Create temporary evaluator to reuse metric calculation
        temp_evaluator = STIndexEvaluator()

        for result in self.results:
            if not result["extraction_result"]:
                continue

            # Evaluate temporal
            temporal_eval = temp_evaluator._evaluate_temporal(
                result["extraction_result"].temporal_mentions,
                result["ground_truth"].get("temporal", [])
            )

            # Evaluate spatial
            spatial_eval = temp_evaluator._evaluate_spatial(
                result["extraction_result"].spatial_mentions,
                result["ground_truth"].get("spatial", [])
            )

            # Update metrics
            temp_evaluator.metrics.total_documents += 1
            temp_evaluator.metrics.total_processing_time += result["processing_time"]
            if result["success"]:
                temp_evaluator.metrics.successful_extractions += 1

        return temp_evaluator.metrics.to_dict()

    def _save_detailed_results(self):
        """Save detailed per-entry results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"batch_detailed_results_{timestamp}.json"

        # Convert to serializable format
        serializable_results = []
        for r in self.results:
            entry = {
                "id": r["id"],
                "text": r["text"],
                "ground_truth": r["ground_truth"],
                "success": r["success"],
                "processing_time": r["processing_time"],
            }

            if r["extraction_result"]:
                entry["temporal_entities"] = [
                    {
                        "text": e.text,
                        "normalized": e.normalized,
                        "temporal_type": str(e.temporal_type) if hasattr(e, 'temporal_type') else None,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None),
                    }
                    for e in r["extraction_result"].temporal_mentions
                ]
                entry["spatial_entities"] = [
                    {
                        "text": e.text,
                        "location_type": str(e.location_type) if hasattr(e, 'location_type') else None,
                        "latitude": e.latitude,
                        "longitude": e.longitude,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None),
                    }
                    for e in r["extraction_result"].spatial_mentions
                ]

            serializable_results.append(entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {output_file}")

    def _save_metrics_summary(self, metrics: Dict[str, Any]):
        """Save metrics summary with configuration details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"batch_metrics_summary_{timestamp}.json"

        # Add configuration details to metrics
        metrics_with_config = {
            "config": self.extractor.config,
            "metrics": metrics
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_config, f, indent=2)

        logger.info(f"Metrics summary saved to {output_file}")
        console.print(f"\n[bold green]Evaluation completed![/bold green]")
        console.print(f"Combined F1: {metrics['overall']['combined_f1']:.4f}")
        console.print(f"Success Rate: {metrics['overall']['success_rate']:.4f}")


def run_batch_evaluation(
    dataset_path: str,
    batch_size: int = 8,
    output_dir: str = "data/output/eval_results",
    config_path: str = "extract",
):
    """
    Run batch evaluation using config-based LLM initialization.

    Args:
        dataset_path: Path to evaluation dataset JSON
        batch_size: Number of samples per batch
        output_dir: Directory to save results
        config_path: Config name/path to use (default: "extract" uses cfg/extract.yml)
    """
    logger.info(f"Initializing STIndexExtractor with config: {config_path}")
    extractor = STIndexExtractor(config_path=config_path)

    evaluator = BatchEvaluator(
        extractor=extractor,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    metrics = evaluator.evaluate_dataset(dataset_path, resume=True)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run batch evaluation using config-based initialization")
    parser.add_argument("dataset_path", type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="data/output/eval_results", help="Output directory")
    parser.add_argument("--config", type=str, default="extract", help="Config name to use (default: extract)")

    args = parser.parse_args()

    run_batch_evaluation(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        config_path=args.config,
    )
