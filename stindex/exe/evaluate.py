"""
Evaluate command execution - runs evaluation on datasets.
"""

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

from stindex import STIndexExtractor
from stindex.eval.metrics import (
    TemporalMetrics,
    SpatialMetrics,
    OverallMetrics,
    calculate_temporal_match,
    calculate_spatial_match,
)
from stindex.utils.config import load_config_from_file
from stindex.utils.constants import PROJECT_DIR


console = Console()


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_checkpoint(output_dir: Path, config_hash: str) -> Optional[Path]:
    """
    Find the latest checkpoint CSV file in the output directory.

    Args:
        output_dir: Output directory to search
        config_hash: Hash of the current config to match checkpoint files

    Returns:
        Path to checkpoint CSV or None if not found
    """
    if not output_dir.exists():
        return None

    # Look for CSV files matching pattern: eval_YYYYMMDD_HHMMSS_{config_hash}.csv
    csv_files = list(output_dir.glob(f"eval_*_{config_hash}.csv"))
    if not csv_files:
        # Also try without config hash for backward compatibility
        csv_files = list(output_dir.glob("eval_*.csv"))

    if csv_files:
        # Return the most recent one
        return max(csv_files, key=lambda p: p.stat().st_mtime)

    return None


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load checkpoint data from CSV file.

    Returns:
        Dict with processed_ids and last_row_data
    """
    processed_ids = set()
    last_metrics = None

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        for row in rows:
            if row.get("id"):
                processed_ids.add(row["id"])

        if rows:
            last_metrics = rows[-1]

    return {
        "processed_ids": processed_ids,
        "last_metrics": last_metrics,
        "checkpoint_path": checkpoint_path
    }


def get_config_hash(config: Dict[str, Any]) -> str:
    """Generate a simple hash from config for checkpoint matching."""
    # Use provider and model as simple identifier
    llm_config = config.get("llm", {})
    provider = llm_config.get("llm_provider", "unknown")
    model = config.get("model_name", "default")
    return f"{provider}_{model}".replace("/", "_").replace(":", "_")


def _evaluate_extraction_result(
    result: Any,
    ground_truth_temporal: List[Dict[str, Any]],
    ground_truth_spatial: List[Dict[str, Any]],
    spatial_match_mode: str,
) -> tuple[TemporalMetrics, SpatialMetrics, List[Dict], List[Dict], str]:
    """
    Common evaluation logic for extraction results.

    Args:
        result: Extraction result from STIndexExtractor
        ground_truth_temporal: Ground truth temporal entities
        ground_truth_spatial: Ground truth spatial entities
        spatial_match_mode: Spatial matching mode (temporal always uses "value_exact")

    Returns:
        Tuple of (temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output)
    """
    temporal_metrics = TemporalMetrics()
    spatial_metrics = SpatialMetrics()
    llm_raw_output = ""
    predicted_temporal = []
    predicted_spatial = []

    # Get raw LLM output if available (even on failure)
    if result.extraction_config and result.extraction_config.raw_llm_output:
        llm_raw_output = result.extraction_config.raw_llm_output

    if not result.success:
        return temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output

    # Evaluate temporal entities (only if ground truth exists)
    predicted_temporal = [e.dict() for e in result.temporal_entities]

    if ground_truth_temporal:
        matched_gt = set()
        for pred in predicted_temporal:
            match_found = False
            for i, gt in enumerate(ground_truth_temporal):
                if i in matched_gt:
                    continue
                # Always use value_exact matching for temporal (compare ISO 8601 values)
                if calculate_temporal_match(pred, gt, "value_exact"):
                    temporal_metrics.true_positives += 1
                    matched_gt.add(i)
                    match_found = True

                    # Check normalization accuracy
                    temporal_metrics.normalization_total += 1
                    if pred.get("normalized") == gt.get("normalized"):
                        temporal_metrics.normalization_correct += 1

                    # Check type accuracy (case-insensitive)
                    temporal_metrics.type_total += 1
                    pred_type = str(pred.get("temporal_type", "")).lower()
                    gt_type = str(gt.get("temporal_type", "")).lower()
                    if pred_type and gt_type and pred_type == gt_type:
                        temporal_metrics.type_correct += 1
                    break

            if not match_found:
                temporal_metrics.false_positives += 1

        temporal_metrics.false_negatives = len(ground_truth_temporal) - len(matched_gt)

    # Evaluate spatial entities (only if ground truth exists)
    predicted_spatial = [e.dict() for e in result.spatial_entities]

    if ground_truth_spatial:
        matched_gt_spatial = set()
        for pred in predicted_spatial:
            match_found = False
            for i, gt in enumerate(ground_truth_spatial):
                if i in matched_gt_spatial:
                    continue
                is_match, distance_error = calculate_spatial_match(pred, gt, spatial_match_mode)
                if is_match:
                    spatial_metrics.true_positives += 1
                    matched_gt_spatial.add(i)
                    match_found = True

                    # Track geocoding
                    if "latitude" in pred:
                        spatial_metrics.geocoding_attempted += 1
                        if pred.get("latitude") is not None:
                            spatial_metrics.geocoding_successful += 1

                            # Track distance error
                            if distance_error is not None:
                                spatial_metrics.distance_errors.append(distance_error)

                    # Check type accuracy (case-insensitive)
                    spatial_metrics.type_total += 1
                    pred_type = str(pred.get("location_type", "")).lower()
                    gt_type = str(gt.get("location_type", "")).lower()
                    if pred_type and gt_type and pred_type == gt_type:
                        spatial_metrics.type_correct += 1
                    break

            if not match_found:
                spatial_metrics.false_positives += 1

        spatial_metrics.false_negatives = len(ground_truth_spatial) - len(matched_gt_spatial)

    return temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output


def _build_result_dict(
    item_id: str,
    input_text: str,
    llm_raw_output: str,
    predicted_temporal: List[Dict],
    predicted_spatial: List[Dict],
    ground_truth_temporal: List[Dict],
    ground_truth_spatial: List[Dict],
    temporal_metrics: TemporalMetrics,
    spatial_metrics: SpatialMetrics,
    processing_time: float,
    error_msg: str = "",
) -> Dict[str, Any]:
    """
    Build result dictionary with metrics.

    Returns dict with empty string for metrics where ground truth doesn't exist.
    """
    return {
        "id": item_id,
        "input_text": input_text,
        "llm_raw_output": llm_raw_output,
        "temporal_predicted": json.dumps(predicted_temporal),
        "temporal_ground_truth": json.dumps(ground_truth_temporal),
        "temporal_precision": temporal_metrics.precision() if ground_truth_temporal else "",
        "temporal_recall": temporal_metrics.recall() if ground_truth_temporal else "",
        "temporal_f1": temporal_metrics.f1_score() if ground_truth_temporal else "",
        "temporal_normalization_accuracy": temporal_metrics.normalization_accuracy() if ground_truth_temporal else "",
        "spatial_predicted": json.dumps(predicted_spatial),
        "spatial_ground_truth": json.dumps(ground_truth_spatial),
        "spatial_precision": spatial_metrics.precision() if ground_truth_spatial else "",
        "spatial_recall": spatial_metrics.recall() if ground_truth_spatial else "",
        "spatial_f1": spatial_metrics.f1_score() if ground_truth_spatial else "",
        "spatial_geocoding_success_rate": spatial_metrics.geocoding_success_rate() if ground_truth_spatial else "",
        "spatial_mean_distance_error_km": spatial_metrics.mean_distance_error() if ground_truth_spatial else "",
        "spatial_accuracy_within_25km": spatial_metrics.accuracy_at_threshold(25) if ground_truth_spatial else "",
        "processing_time_seconds": processing_time,
        "error": error_msg,
        "_metrics": {
            "temporal": temporal_metrics,
            "spatial": spatial_metrics,
            "has_temporal_gt": bool(ground_truth_temporal),
            "has_spatial_gt": bool(ground_truth_spatial)
        }
    }


def evaluate_single_item(
    extractor: STIndexExtractor,
    item: Dict[str, Any],
    spatial_match_mode: str,
) -> Dict[str, Any]:
    """
    Evaluate a single dataset item.

    Args:
        extractor: STIndexExtractor instance
        item: Dataset item with text and ground_truth
        spatial_match_mode: Spatial matching mode (temporal always uses "value_exact")

    Returns:
        Dict with evaluation results for this item
    """
    # Extract
    start_time = time.time()
    result = extractor.extract(item["text"])
    processing_time = time.time() - start_time

    # Get ground truth
    ground_truth_temporal = item.get("ground_truth", {}).get("temporal", [])
    ground_truth_spatial = item.get("ground_truth", {}).get("spatial", [])

    # Evaluate using common logic
    temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output = _evaluate_extraction_result(
        result,
        ground_truth_temporal,
        ground_truth_spatial,
        spatial_match_mode
    )

    # Build and return result dict
    error_msg = result.error if not result.success else ""
    return _build_result_dict(
        item.get("id", "unknown"),
        item["text"],
        llm_raw_output,
        predicted_temporal,
        predicted_spatial,
        ground_truth_temporal,
        ground_truth_spatial,
        temporal_metrics,
        spatial_metrics,
        processing_time,
        error_msg
    )


def evaluate_batch_items(
    extractor: STIndexExtractor,
    items: List[Dict[str, Any]],
    spatial_match_mode: str,
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of dataset items using LLM batch API.

    Args:
        extractor: STIndexExtractor instance
        items: List of dataset items with text and ground_truth
        spatial_match_mode: Spatial matching mode (temporal always uses "value_exact")

    Returns:
        List of evaluation result dicts
    """
    # Extract texts
    texts = [item["text"] for item in items]

    # Batch extraction using LLM batch API
    results = extractor.extract_batch(texts, use_batch_api=True)

    # Evaluate each result
    evaluated_results = []
    for item, result in zip(items, results):
        # Get ground truth
        ground_truth_temporal = item.get("ground_truth", {}).get("temporal", [])
        ground_truth_spatial = item.get("ground_truth", {}).get("spatial", [])

        # Evaluate using common logic
        temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output = _evaluate_extraction_result(
            result,
            ground_truth_temporal,
            ground_truth_spatial,
            spatial_match_mode
        )

        # Build result dict
        error_msg = result.error if not result.success else ""
        result_dict = _build_result_dict(
            item.get("id", "unknown"),
            item["text"],
            llm_raw_output,
            predicted_temporal,
            predicted_spatial,
            ground_truth_temporal,
            ground_truth_spatial,
            temporal_metrics,
            spatial_metrics,
            result.processing_time,
            error_msg
        )
        evaluated_results.append(result_dict)

    return evaluated_results


def execute_evaluate(
    config: str = "evaluate",
    dataset: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sample_limit: Optional[int] = None,
    resume: Optional[bool] = None,
):
    """
    Execute evaluation on a dataset.

    Args:
        config: Config file name (default: evaluate.yml)
        dataset: Override dataset path
        output_dir: Override output directory
        sample_limit: Override sample limit
        resume: Override resume setting
    """
    try:
        # Load configuration
        eval_config = load_config_from_file(config)

        # Get evaluation settings
        eval_settings = eval_config.get("evaluation", {})
        output_settings = eval_config.get("output", {})

        # Resolve paths
        dataset_path = Path(dataset) if dataset else Path(PROJECT_DIR) / eval_settings.get("dataset_path", "data/input/eval_dataset_100.json")

        # Get dataset name (without extension) for subdirectory
        dataset_name = dataset_path.stem  # e.g., "eval_dataset_100" from "eval_dataset_100.json"

        # Get model name from config and normalize it (remove slashes, colons, etc.)
        model_name = eval_config.get("llm", {}).get("model_name", "default")
        # Normalize model name: replace slashes, colons, and other path-unsafe characters
        normalized_model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")

        # If output_dir is provided, use it directly; otherwise create subdirectory based on dataset name and model
        if output_dir:
            output_directory = Path(output_dir)
        else:
            base_output_dir = Path(PROJECT_DIR) / eval_settings.get("output_dir", "data/output/evaluations")
            # Create subdirectory: {dataset}-{normalized_model_name}
            output_directory = base_output_dir / f"{dataset_name}-{normalized_model_name}"

        # Override sample limit if provided
        if sample_limit is not None:
            eval_settings["sample_limit"] = sample_limit

        # Override resume if provided
        if resume is not None:
            eval_settings["resume"] = resume

        # Create output directory
        output_directory.mkdir(parents=True, exist_ok=True)

        # Load dataset
        console.print(f"[bold blue]Loading dataset:[/bold blue] {dataset_path}")
        dataset_items = load_dataset(dataset_path)

        # Apply sample limit
        limit = eval_settings.get("sample_limit")
        if limit:
            dataset_items = dataset_items[:limit]

        console.print(f"[green]✓ Loaded {len(dataset_items)} items[/green]")

        # Check for checkpoint and resume
        config_hash = get_config_hash(eval_config)
        checkpoint_data = None
        processed_ids = set()

        if eval_settings.get("resume", True):
            checkpoint_path = find_latest_checkpoint(output_directory, config_hash)
            if checkpoint_path:
                console.print(f"[yellow]Found checkpoint:[/yellow] {checkpoint_path}")
                checkpoint_data = load_checkpoint(checkpoint_path)
                processed_ids = checkpoint_data["processed_ids"]
                console.print(f"[green]✓ Resuming from {len(processed_ids)} completed items[/green]")

        # Filter out already processed items
        remaining_items = [item for item in dataset_items if item.get("id") not in processed_ids]

        if not remaining_items:
            console.print("[green]All items already processed![/green]")
            return

        console.print(f"[blue]Processing {len(remaining_items)} remaining items...[/blue]")

        # Create extractor
        extractor = STIndexExtractor(config_path=config)

        # Prepare CSV output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"eval_{timestamp}_{config_hash}.csv"
        csv_path = output_directory / csv_filename

        # Determine if we need to write headers (new file vs resume)
        write_headers = not checkpoint_data
        file_mode = "w" if not checkpoint_data else "a"

        # If resuming, use the same file
        if checkpoint_data:
            csv_path = checkpoint_data["checkpoint_path"]
            file_mode = "a"

        # CSV columns - organized with predictions next to ground truth for easy comparison
        csv_columns = output_settings.get("csv_columns", [
            "id",
            "input_text",
            # Temporal results
            "temporal_predicted",
            "temporal_ground_truth",
            "temporal_precision",
            "temporal_recall",
            "temporal_f1",
            "temporal_normalization_accuracy",
            # Spatial results
            "spatial_predicted",
            "spatial_ground_truth",
            "spatial_precision",
            "spatial_recall",
            "spatial_f1",
            "spatial_geocoding_success_rate",
            "spatial_mean_distance_error_km",
            "spatial_accuracy_within_25km",
            # Processing metadata
            "processing_time_seconds",
            "error",
            "llm_raw_output"
        ])

        # Overall metrics aggregation
        overall_metrics = OverallMetrics()

        # Get batch mode settings
        batch_mode = eval_settings.get("batch_mode", False)
        batch_size = eval_settings.get("batch_size", 10)

        if batch_mode:
            console.print(f"[yellow]Batch mode enabled:[/yellow] batch_size={batch_size}")
        else:
            console.print(f"[yellow]Single-item mode:[/yellow] processing one-by-one")

        # Suppress INFO-level logs during evaluation to avoid interfering with progress bar
        # Store original log level
        logger.info("Starting evaluation...")
        original_level = logger._core.min_level
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="WARNING")  # Only show WARNING and above during progress

        # Process items with progress bar
        with open(csv_path, file_mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

            if write_headers:
                writer.writeheader()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Evaluating...", total=len(remaining_items))

                if batch_mode:
                    # Batch processing mode
                    for batch_start in range(0, len(remaining_items), batch_size):
                        batch_end = min(batch_start + batch_size, len(remaining_items))
                        batch_items = remaining_items[batch_start:batch_end]
                        current_batch_size = len(batch_items)

                        # Evaluate batch
                        batch_results = evaluate_batch_items(
                            extractor,
                            batch_items,
                            eval_settings.get("spatial_match_mode", "fuzzy")
                        )

                        # Process and write results
                        for i, item_result in enumerate(batch_results):
                            # Aggregate metrics (only for entries with ground truth)
                            item_metrics = item_result.pop("_metrics")

                            # Only aggregate temporal metrics if ground truth exists
                            if item_metrics.get("has_temporal_gt"):
                                overall_metrics.temporal.true_positives += item_metrics["temporal"].true_positives
                                overall_metrics.temporal.false_positives += item_metrics["temporal"].false_positives
                                overall_metrics.temporal.false_negatives += item_metrics["temporal"].false_negatives
                                overall_metrics.temporal.normalization_correct += item_metrics["temporal"].normalization_correct
                                overall_metrics.temporal.normalization_total += item_metrics["temporal"].normalization_total
                                overall_metrics.temporal.type_correct += item_metrics["temporal"].type_correct
                                overall_metrics.temporal.type_total += item_metrics["temporal"].type_total

                            # Only aggregate spatial metrics if ground truth exists
                            if item_metrics.get("has_spatial_gt"):
                                overall_metrics.spatial.true_positives += item_metrics["spatial"].true_positives
                                overall_metrics.spatial.false_positives += item_metrics["spatial"].false_positives
                                overall_metrics.spatial.false_negatives += item_metrics["spatial"].false_negatives
                                overall_metrics.spatial.geocoding_attempted += item_metrics["spatial"].geocoding_attempted
                                overall_metrics.spatial.geocoding_successful += item_metrics["spatial"].geocoding_successful
                                overall_metrics.spatial.distance_errors.extend(item_metrics["spatial"].distance_errors)
                                overall_metrics.spatial.type_correct += item_metrics["spatial"].type_correct
                                overall_metrics.spatial.type_total += item_metrics["spatial"].type_total

                            overall_metrics.total_documents += 1
                            overall_metrics.successful_extractions += 1 if not item_result["error"] else 0
                            overall_metrics.total_processing_time += item_result["processing_time_seconds"]

                            # Write to CSV
                            writer.writerow(item_result)

                        # Flush at checkpoint intervals
                        if (batch_start + current_batch_size) % eval_settings.get("checkpoint_interval", 10) == 0:
                            csvfile.flush()

                        # Update progress bar once per batch
                        progress.update(task, advance=current_batch_size)

                else:
                    # Single-item processing mode (original behavior)
                    for i, item in enumerate(remaining_items):
                        # Evaluate single item
                        item_result = evaluate_single_item(
                            extractor,
                            item,
                            eval_settings.get("spatial_match_mode", "fuzzy")
                        )

                        # Aggregate metrics (only for entries with ground truth)
                        item_metrics = item_result.pop("_metrics")

                        # Only aggregate temporal metrics if ground truth exists
                        if item_metrics.get("has_temporal_gt"):
                            overall_metrics.temporal.true_positives += item_metrics["temporal"].true_positives
                            overall_metrics.temporal.false_positives += item_metrics["temporal"].false_positives
                            overall_metrics.temporal.false_negatives += item_metrics["temporal"].false_negatives
                            overall_metrics.temporal.normalization_correct += item_metrics["temporal"].normalization_correct
                            overall_metrics.temporal.normalization_total += item_metrics["temporal"].normalization_total
                            overall_metrics.temporal.type_correct += item_metrics["temporal"].type_correct
                            overall_metrics.temporal.type_total += item_metrics["temporal"].type_total

                        # Only aggregate spatial metrics if ground truth exists
                        if item_metrics.get("has_spatial_gt"):
                            overall_metrics.spatial.true_positives += item_metrics["spatial"].true_positives
                            overall_metrics.spatial.false_positives += item_metrics["spatial"].false_positives
                            overall_metrics.spatial.false_negatives += item_metrics["spatial"].false_negatives
                            overall_metrics.spatial.geocoding_attempted += item_metrics["spatial"].geocoding_attempted
                            overall_metrics.spatial.geocoding_successful += item_metrics["spatial"].geocoding_successful
                            overall_metrics.spatial.distance_errors.extend(item_metrics["spatial"].distance_errors)
                            overall_metrics.spatial.type_correct += item_metrics["spatial"].type_correct
                            overall_metrics.spatial.type_total += item_metrics["spatial"].type_total

                        overall_metrics.total_documents += 1
                        overall_metrics.successful_extractions += 1 if not item_result["error"] else 0
                        overall_metrics.total_processing_time += item_result["processing_time_seconds"]

                        # Write to CSV
                        writer.writerow(item_result)

                        # Flush every checkpoint_interval items
                        if (i + 1) % eval_settings.get("checkpoint_interval", 10) == 0:
                            csvfile.flush()

                        progress.update(task, advance=1)

        # Restore logger to INFO level after progress bar
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        console.print(f"\n[green]✓ Evaluation complete![/green]")
        console.print(f"[blue]Results saved to:[/blue] {csv_path}")

        # Calculate cumulative metrics from ALL rows in the CSV (including previous runs)
        cumulative_metrics = calculate_cumulative_metrics(
            csv_path,
            temporal_match_mode=eval_settings.get("temporal_match_mode", "exact"),
            spatial_match_mode=eval_settings.get("spatial_match_mode", "exact")
        )

        # Display cumulative summary metrics
        console.print("\n[bold cyan]Cumulative Metrics (All Results):[/bold cyan]")
        display_summary_metrics(cumulative_metrics)

        # Save cumulative summary JSON if configured
        if output_settings.get("save_summary", True):
            summary_path = csv_path.with_suffix(".summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(cumulative_metrics.to_dict(), f, indent=2)
            console.print(f"[blue]Cumulative summary saved to:[/blue] {summary_path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def calculate_cumulative_metrics(csv_path: Path, temporal_match_mode: str = "exact", spatial_match_mode: str = "exact") -> OverallMetrics:
    """
    Calculate cumulative metrics from all rows in the CSV file.

    Args:
        csv_path: Path to the CSV file
        temporal_match_mode: Temporal matching mode (exact, overlap, normalized)
        spatial_match_mode: Spatial matching mode (exact, fuzzy)

    Returns:
        OverallMetrics aggregated from all rows
    """
    cumulative_metrics = OverallMetrics()

    if not csv_path.exists():
        return cumulative_metrics

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Parse temporal metrics from the row
            temporal_pred = json.loads(row.get("temporal_predicted", "[]"))
            temporal_gt = json.loads(row.get("temporal_ground_truth", "[]"))

            # Reconstruct temporal metrics for this row (only if ground truth exists)
            temporal_tp = 0
            temporal_fp = 0
            normalization_correct = 0
            normalization_total = 0

            if temporal_gt:  # Only calculate if ground truth has temporal entities
                matched_gt = set()
                for pred in temporal_pred:
                    match_found = False
                    for i, gt in enumerate(temporal_gt):
                        if i in matched_gt:
                            continue
                        # Use proper matching function (case-insensitive by default in "exact" mode)
                        if calculate_temporal_match(pred, gt, temporal_match_mode):
                            temporal_tp += 1
                            matched_gt.add(i)
                            match_found = True

                            # Check normalization
                            normalization_total += 1
                            if pred.get("normalized") == gt.get("normalized"):
                                normalization_correct += 1

                            # Check type (case-insensitive)
                            cumulative_metrics.temporal.type_total += 1
                            pred_type = str(pred.get("temporal_type", "")).lower()
                            gt_type = str(gt.get("temporal_type", "")).lower()
                            if pred_type and gt_type and pred_type == gt_type:
                                cumulative_metrics.temporal.type_correct += 1
                            break

                    if not match_found:
                        temporal_fp += 1

                temporal_fn = len(temporal_gt) - len(matched_gt)
            else:
                temporal_fn = 0

            # Parse spatial metrics (only if ground truth exists)
            spatial_pred = json.loads(row.get("spatial_predicted", "[]"))
            spatial_gt = json.loads(row.get("spatial_ground_truth", "[]"))

            spatial_tp = 0
            spatial_fp = 0
            geocoding_attempted = 0
            geocoding_successful = 0

            if spatial_gt:  # Only calculate if ground truth has spatial entities
                matched_gt_spatial = set()
                for pred in spatial_pred:
                    match_found = False
                    for i, gt in enumerate(spatial_gt):
                        if i in matched_gt_spatial:
                            continue
                        # Use proper matching function (case-insensitive by default in "exact" mode)
                        is_match, distance_error = calculate_spatial_match(pred, gt, spatial_match_mode)
                        if is_match:
                            spatial_tp += 1
                            matched_gt_spatial.add(i)
                            match_found = True

                            # Track geocoding
                            if "latitude" in pred:
                                geocoding_attempted += 1
                                if pred.get("latitude") is not None:
                                    geocoding_successful += 1

                                    # Track distance error (calculated by matching function)
                                    if distance_error is not None:
                                        cumulative_metrics.spatial.distance_errors.append(distance_error)

                            # Check type (case-insensitive)
                            cumulative_metrics.spatial.type_total += 1
                            pred_type = str(pred.get("location_type", "")).lower()
                            gt_type = str(gt.get("location_type", "")).lower()
                            if pred_type and gt_type and pred_type == gt_type:
                                cumulative_metrics.spatial.type_correct += 1
                            break

                    if not match_found:
                        spatial_fp += 1

                spatial_fn = len(spatial_gt) - len(matched_gt_spatial)
            else:
                spatial_fn = 0

            # Aggregate into cumulative metrics
            cumulative_metrics.temporal.true_positives += temporal_tp
            cumulative_metrics.temporal.false_positives += temporal_fp
            cumulative_metrics.temporal.false_negatives += temporal_fn
            cumulative_metrics.temporal.normalization_correct += normalization_correct
            cumulative_metrics.temporal.normalization_total += normalization_total

            cumulative_metrics.spatial.true_positives += spatial_tp
            cumulative_metrics.spatial.false_positives += spatial_fp
            cumulative_metrics.spatial.false_negatives += spatial_fn
            cumulative_metrics.spatial.geocoding_attempted += geocoding_attempted
            cumulative_metrics.spatial.geocoding_successful += geocoding_successful

            cumulative_metrics.total_documents += 1
            cumulative_metrics.successful_extractions += 1 if not row.get("error") else 0

            try:
                processing_time = float(row.get("processing_time_seconds", 0))
                cumulative_metrics.total_processing_time += processing_time
            except (ValueError, TypeError):
                pass

    return cumulative_metrics


def display_summary_metrics(metrics: OverallMetrics):
    """Display summary metrics in a nice table format."""
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    # Overall metrics
    table.add_row("Total Documents", str(metrics.total_documents))
    table.add_row("Successful Extractions", str(metrics.successful_extractions))
    table.add_row("Success Rate", f"{metrics.success_rate():.2%}")
    table.add_row("Avg Processing Time", f"{metrics.average_processing_time():.2f}s")
    table.add_row("", "")

    # Temporal metrics
    table.add_row("[bold]Temporal Metrics[/bold]", "")
    table.add_row("Precision", f"{metrics.temporal.precision():.4f}")
    table.add_row("Recall", f"{metrics.temporal.recall():.4f}")
    table.add_row("F1 Score", f"{metrics.temporal.f1_score():.4f}")
    table.add_row("Normalization Accuracy", f"{metrics.temporal.normalization_accuracy():.4f}")
    table.add_row("", "")

    # Spatial metrics
    table.add_row("[bold]Spatial Metrics[/bold]", "")
    table.add_row("Precision", f"{metrics.spatial.precision():.4f}")
    table.add_row("Recall", f"{metrics.spatial.recall():.4f}")
    table.add_row("F1 Score", f"{metrics.spatial.f1_score():.4f}")
    table.add_row("Geocoding Success Rate", f"{metrics.spatial.geocoding_success_rate():.4f}")
    table.add_row("Mean Distance Error", f"{metrics.spatial.mean_distance_error():.2f} km")
    table.add_row("Accuracy within 25km", f"{metrics.spatial.accuracy_at_threshold(25):.2%}")
    table.add_row("", "")

    # Combined
    table.add_row("[bold]Combined F1[/bold]", f"{metrics.combined_f1():.4f}")

    console.print(table)
