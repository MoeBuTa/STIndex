"""
Evaluation framework for STIndex spatiotemporal extraction.

Evaluates extraction performance against ground truth annotations using:
- Entity-level precision, recall, F1
- Normalization accuracy for temporal expressions
- Geocoding accuracy and distance errors for spatial expressions
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path to allow imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track

from stindex import STIndexExtractor
from stindex.agents.response.models import SpatioTemporalResult
from eval.metrics import (
    TemporalMetrics,
    SpatialMetrics,
    OverallMetrics,
    calculate_temporal_match,
    calculate_spatial_match,
    normalize_temporal_value,
)


console = Console()


class STIndexEvaluator:
    """Evaluator for STIndex extraction system"""

    def __init__(
        self,
        extractor: Optional[STIndexExtractor] = None,
        output_dir: str = "data/output/eval_results"
    ):
        """
        Initialize evaluator.

        Args:
            extractor: STIndexExtractor instance (creates default if None)
            output_dir: Directory to save evaluation results
        """
        self.extractor = extractor or STIndexExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = OverallMetrics()
        self.results: List[Dict[str, Any]] = []

    def evaluate_dataset(
        self,
        dataset_path: str,
        save_detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate on a complete dataset.

        Args:
            dataset_path: Path to JSON dataset file
            save_detailed: Whether to save detailed per-entry results

        Returns:
            Dictionary with overall metrics
        """
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        logger.info(f"Evaluating on {len(dataset)} entries...")

        for entry in track(dataset, description="Evaluating..."):
            result = self.evaluate_entry(entry)
            self.results.append(result)

        # Calculate final metrics
        final_metrics = self.metrics.to_dict()

        # Save results
        if save_detailed:
            self._save_detailed_results()

        self._save_metrics_summary(final_metrics)

        return final_metrics

    def evaluate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single dataset entry.

        Args:
            entry: Dataset entry with 'text' and 'ground_truth'

        Returns:
            Dictionary with evaluation results for this entry
        """
        text = entry["text"]
        ground_truth = entry["ground_truth"]
        entry_id = entry.get("id", "unknown")

        # Extract with STIndex and capture raw LLM output
        start_time = time.time()
        raw_llm_output = None
        try:
            # Get raw LLM extraction first
            from stindex.agents.prompts.extraction import ExtractionPrompt
            from stindex.agents.response.models import ExtractionResult

            messages = ExtractionPrompt.build_messages(text.strip(), use_few_shot=False)
            raw_llm_output = self.extractor.llm.extract(
                messages=messages,
                response_model=ExtractionResult,
            )

            # Now get the full extraction result
            extraction_result = self.extractor.extract(text)
            processing_time = time.time() - start_time
            success = extraction_result.success
        except Exception as e:
            logger.error(f"Extraction failed for entry {entry_id}: {e}")
            processing_time = time.time() - start_time
            extraction_result = None
            success = False

        # Update overall metrics
        self.metrics.total_documents += 1
        self.metrics.total_processing_time += processing_time
        if success:
            self.metrics.successful_extractions += 1

        # Evaluate temporal entities
        temporal_eval = self._evaluate_temporal(
            extraction_result.temporal_entities if extraction_result else [],
            ground_truth.get("temporal", [])
        )

        # Evaluate spatial entities
        spatial_eval = self._evaluate_spatial(
            extraction_result.spatial_entities if extraction_result else [],
            ground_truth.get("spatial", [])
        )

        # Prepare detailed result with raw LLM output
        result = {
            "id": entry_id,
            "text": text,
            "input_prompt": entry.get("prompt", "Extract spatiotemporal information from this text."),
            "ground_truth": ground_truth,
            "llm_raw_output": {
                "temporal_mentions": [
                    {
                        "text": m.text,
                        "normalized": m.normalized,
                        "temporal_type": str(m.temporal_type)
                    }
                    for m in (raw_llm_output.temporal_mentions if raw_llm_output else [])
                ],
                "spatial_mentions": [
                    {
                        "text": m.text,
                        "location_type": str(m.location_type),
                        "parent_region": m.parent_region
                    }
                    for m in (raw_llm_output.spatial_mentions if raw_llm_output else [])
                ]
            },
            "final_output": {
                "temporal_entities": [
                    {
                        "text": e.text,
                        "normalized": e.normalized,
                        "temporal_type": str(e.temporal_type) if hasattr(e, 'temporal_type') else None,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None)
                    }
                    for e in (extraction_result.temporal_entities if extraction_result else [])
                ],
                "spatial_entities": [
                    {
                        "text": e.text,
                        "location_type": str(e.location_type) if hasattr(e, 'location_type') else None,
                        "latitude": e.latitude,
                        "longitude": e.longitude,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None)
                    }
                    for e in (extraction_result.spatial_entities if extraction_result else [])
                ]
            },
            "evaluation": {
                "success": success,
                "processing_time": round(processing_time, 3),
                "temporal_metrics": temporal_eval,
                "spatial_metrics": spatial_eval
            }
        }

        return result

    def _evaluate_temporal(
        self,
        predicted: List[Any],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate temporal extraction"""
        # Convert predicted entities to dicts
        pred_dicts = [
            {
                "text": e.text,
                "normalized": e.normalized if e.normalized else "",
                "temporal_type": str(e.temporal_type) if hasattr(e, 'temporal_type') else ""
            }
            for e in predicted
        ]

        # Match predictions to ground truth
        matched_pred = set()
        matched_gt = set()

        normalization_correct = 0
        type_correct = 0

        for gt_idx, gt in enumerate(ground_truth):
            best_match = None
            best_match_idx = None

            # Find best matching prediction
            for pred_idx, pred in enumerate(pred_dicts):
                if pred_idx in matched_pred:
                    continue

                # Try exact match first, then fuzzy
                if calculate_temporal_match(pred, gt, "exact"):
                    best_match = pred
                    best_match_idx = pred_idx
                    break
                elif calculate_temporal_match(pred, gt, "overlap"):
                    if best_match is None:
                        best_match = pred
                        best_match_idx = pred_idx

            if best_match:
                matched_pred.add(best_match_idx)
                matched_gt.add(gt_idx)
                self.metrics.temporal.true_positives += 1

                # Check normalization accuracy
                if best_match.get("normalized") and gt.get("normalized"):
                    pred_norm = normalize_temporal_value(best_match["normalized"])
                    gt_norm = normalize_temporal_value(gt["normalized"])
                    if pred_norm == gt_norm:
                        normalization_correct += 1
                        self.metrics.temporal.normalization_correct += 1
                    self.metrics.temporal.normalization_total += 1

                # Check type accuracy
                if best_match.get("temporal_type") and gt.get("temporal_type"):
                    if best_match["temporal_type"] == gt["temporal_type"]:
                        type_correct += 1
                        self.metrics.temporal.type_correct += 1
                    self.metrics.temporal.type_total += 1

        # Count false positives and false negatives
        false_positives = len(pred_dicts) - len(matched_pred)
        false_negatives = len(ground_truth) - len(matched_gt)

        self.metrics.temporal.false_positives += false_positives
        self.metrics.temporal.false_negatives += false_negatives

        return {
            "true_positives": len(matched_pred),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "normalization_correct": normalization_correct,
            "type_correct": type_correct
        }

    def _evaluate_spatial(
        self,
        predicted: List[Any],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate spatial extraction"""
        # Convert predicted entities to dicts
        pred_dicts = [
            {
                "text": e.text,
                "location_type": str(e.location_type) if hasattr(e, 'location_type') else "",
                "latitude": e.latitude,
                "longitude": e.longitude
            }
            for e in predicted
        ]

        # Match predictions to ground truth
        matched_pred = set()
        matched_gt = set()

        distance_errors = []
        type_correct = 0
        geocoding_attempted = 0
        geocoding_successful = 0

        for gt_idx, gt in enumerate(ground_truth):
            best_match = None
            best_match_idx = None

            # Find best matching prediction
            for pred_idx, pred in enumerate(pred_dicts):
                if pred_idx in matched_pred:
                    continue

                # Try exact match first, then fuzzy
                is_match, distance = calculate_spatial_match(pred, gt, "exact")
                if not is_match:
                    is_match, distance = calculate_spatial_match(pred, gt, "fuzzy")

                if is_match:
                    best_match = pred
                    best_match_idx = pred_idx
                    if distance is not None:
                        distance_errors.append(distance)
                    break

            if best_match:
                matched_pred.add(best_match_idx)
                matched_gt.add(gt_idx)
                self.metrics.spatial.true_positives += 1

                # Check geocoding
                if gt.get("latitude") is not None:
                    geocoding_attempted += 1
                    self.metrics.spatial.geocoding_attempted += 1

                    if best_match.get("latitude") is not None:
                        geocoding_successful += 1
                        self.metrics.spatial.geocoding_successful += 1

                # Check type accuracy
                if best_match.get("location_type") and gt.get("location_type"):
                    if best_match["location_type"] == gt["location_type"]:
                        type_correct += 1
                        self.metrics.spatial.type_correct += 1
                    self.metrics.spatial.type_total += 1

        # Count false positives and false negatives
        false_positives = len(pred_dicts) - len(matched_pred)
        false_negatives = len(ground_truth) - len(matched_gt)

        self.metrics.spatial.false_positives += false_positives
        self.metrics.spatial.false_negatives += false_negatives
        self.metrics.spatial.distance_errors.extend(distance_errors)

        return {
            "true_positives": len(matched_pred),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "geocoding_attempted": geocoding_attempted,
            "geocoding_successful": geocoding_successful,
            "distance_errors_km": [round(d, 2) for d in distance_errors],
            "type_correct": type_correct
        }

    def _save_detailed_results(self):
        """Save detailed per-entry results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"detailed_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {output_file}")

    def _save_metrics_summary(self, metrics: Dict[str, Any]):
        """Save metrics summary with configuration details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"metrics_summary_{timestamp}.json"

        # Get config from extractor (already loaded during initialization)
        config = self.extractor.config

        # Add configuration details to metrics
        metrics_with_config = {
            "config": config,
            "metrics": metrics
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_config, f, indent=2)

        logger.info(f"Metrics summary saved to {output_file}")

        # Also print to console
        self._print_metrics_table(metrics)

    def _print_metrics_table(self, metrics: Dict[str, Any]):
        """Print formatted metrics table"""
        console.print("\n[bold cyan]Evaluation Results[/bold cyan]\n")

        # Overall metrics
        overall_table = Table(title="Overall Metrics", show_header=True)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="green")

        overall = metrics["overall"]
        overall_table.add_row("Combined F1 Score", f"{overall['combined_f1']:.4f}")
        overall_table.add_row("Success Rate", f"{overall['success_rate']:.4f}")
        overall_table.add_row("Avg Processing Time", f"{overall['average_processing_time_seconds']:.2f}s")
        overall_table.add_row("Total Documents", str(overall['total_documents']))

        console.print(overall_table)
        console.print()

        # Temporal metrics
        temporal_table = Table(title="Temporal Extraction Metrics", show_header=True)
        temporal_table.add_column("Metric", style="cyan")
        temporal_table.add_column("Value", style="yellow")

        temporal = metrics["temporal"]
        temporal_table.add_row("Precision", f"{temporal['precision']:.4f}")
        temporal_table.add_row("Recall", f"{temporal['recall']:.4f}")
        temporal_table.add_row("F1 Score", f"{temporal['f1_score']:.4f}")
        temporal_table.add_row("Normalization Accuracy", f"{temporal['normalization_accuracy']:.4f}")
        temporal_table.add_row("Type Accuracy", f"{temporal['type_accuracy']:.4f}")

        console.print(temporal_table)
        console.print()

        # Spatial metrics
        spatial_table = Table(title="Spatial Extraction Metrics", show_header=True)
        spatial_table.add_column("Metric", style="cyan")
        spatial_table.add_column("Value", style="magenta")

        spatial = metrics["spatial"]
        spatial_table.add_row("Precision", f"{spatial['precision']:.4f}")
        spatial_table.add_row("Recall", f"{spatial['recall']:.4f}")
        spatial_table.add_row("F1 Score", f"{spatial['f1_score']:.4f}")
        spatial_table.add_row("Geocoding Success Rate", f"{spatial['geocoding_success_rate']:.4f}")
        spatial_table.add_row("Mean Distance Error", f"{spatial['mean_distance_error_km']:.2f} km")
        spatial_table.add_row("Median Distance Error", f"{spatial['median_distance_error_km']:.2f} km")
        spatial_table.add_row("% within 25km", f"{spatial['percentage_within_25km']:.4f}")
        spatial_table.add_row("% 25km to 200km", f"{spatial['percentage_25km_to_200km']:.4f}")
        spatial_table.add_row("% above 200km", f"{spatial['percentage_above_200km']:.4f}")
        spatial_table.add_row("Type Accuracy", f"{spatial['type_accuracy']:.4f}")

        console.print(spatial_table)


def run_evaluation(
    dataset_path: str,
    output_dir: str = "data/output/eval_results",
    config_path: str = "extract"
):
    """
    Run evaluation on a dataset.

    Args:
        dataset_path: Path to evaluation dataset JSON
        output_dir: Directory to save results
        config_path: Config name/path to use (default: "extract" uses cfg/extract.yml)
    """
    extractor = STIndexExtractor(config_path=config_path)
    evaluator = STIndexEvaluator(extractor=extractor, output_dir=output_dir)
    metrics = evaluator.evaluate_dataset(dataset_path, save_detailed=True)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation on STIndex dataset")
    parser.add_argument("dataset_path", type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--output-dir", type=str, default="data/output/eval_results", help="Output directory for results")
    parser.add_argument("--config", type=str, default="extract", help="Config name to use (default: extract)")

    args = parser.parse_args()
    run_evaluation(args.dataset_path, args.output_dir, args.config)
