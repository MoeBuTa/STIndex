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


def evaluate_single_item(
    extractor: STIndexExtractor,
    item: Dict[str, Any],
    temporal_match_mode: str,
    spatial_match_mode: str,
) -> Dict[str, Any]:
    """
    Evaluate a single dataset item.

    Args:
        extractor: STIndexExtractor instance
        item: Dataset item with text and ground_truth
        temporal_match_mode: Temporal matching mode
        spatial_match_mode: Spatial matching mode

    Returns:
        Dict with evaluation results for this item
    """
    # Extract
    start_time = time.time()
    result = extractor.extract(item["text"])
    processing_time = time.time() - start_time

    # Initialize metrics for this item
    temporal_metrics = TemporalMetrics()
    spatial_metrics = SpatialMetrics()

    error_msg = None
    llm_raw_output = ""

    # Initialize result variables
    predicted_temporal = []
    predicted_spatial = []
    ground_truth_temporal = item.get("ground_truth", {}).get("temporal", [])
    ground_truth_spatial = item.get("ground_truth", {}).get("spatial", [])

    if not result.success:
        error_msg = result.error
    else:
        # Get raw LLM output if available
        if result.extraction_config and result.extraction_config.raw_llm_output:
            llm_raw_output = result.extraction_config.raw_llm_output

        # Evaluate temporal entities
        predicted_temporal = [e.dict() for e in result.temporal_entities]

        # Calculate temporal metrics
        matched_gt = set()
        for pred in predicted_temporal:
            match_found = False
            for i, gt in enumerate(ground_truth_temporal):
                if i in matched_gt:
                    continue
                if calculate_temporal_match(pred, gt, temporal_match_mode):
                    temporal_metrics.true_positives += 1
                    matched_gt.add(i)
                    match_found = True

                    # Check normalization accuracy
                    temporal_metrics.normalization_total += 1
                    if pred.get("normalized") == gt.get("normalized"):
                        temporal_metrics.normalization_correct += 1
                    break

            if not match_found:
                temporal_metrics.false_positives += 1

        temporal_metrics.false_negatives = len(ground_truth_temporal) - len(matched_gt)

        # Evaluate spatial entities
        predicted_spatial = [e.dict() for e in result.spatial_entities]

        # Calculate spatial metrics
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
                    break

            if not match_found:
                spatial_metrics.false_positives += 1

        spatial_metrics.false_negatives = len(ground_truth_spatial) - len(matched_gt_spatial)

    # Return detailed results
    return {
        "id": item.get("id", "unknown"),
        "input_text": item["text"],
        "llm_raw_output": llm_raw_output,
        "temporal_predicted": json.dumps(predicted_temporal),
        "temporal_ground_truth": json.dumps(ground_truth_temporal),
        "temporal_precision": temporal_metrics.precision(),
        "temporal_recall": temporal_metrics.recall(),
        "temporal_f1": temporal_metrics.f1_score(),
        "temporal_normalization_accuracy": temporal_metrics.normalization_accuracy(),
        "spatial_predicted": json.dumps(predicted_spatial),
        "spatial_ground_truth": json.dumps(ground_truth_spatial),
        "spatial_precision": spatial_metrics.precision(),
        "spatial_recall": spatial_metrics.recall(),
        "spatial_f1": spatial_metrics.f1_score(),
        "spatial_geocoding_success_rate": spatial_metrics.geocoding_success_rate(),
        "spatial_mean_distance_error_km": spatial_metrics.mean_distance_error(),
        "spatial_accuracy_within_25km": spatial_metrics.accuracy_at_threshold(25),
        "processing_time_seconds": processing_time,
        "error": error_msg or "",
        "_metrics": {
            "temporal": temporal_metrics,
            "spatial": spatial_metrics
        }
    }


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
        output_directory = Path(output_dir) if output_dir else Path(PROJECT_DIR) / eval_settings.get("output_dir", "data/output/evaluations")

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

        # CSV columns
        csv_columns = output_settings.get("csv_columns", [
            "id", "input_text", "llm_raw_output",
            "temporal_predicted", "temporal_ground_truth",
            "temporal_precision", "temporal_recall", "temporal_f1", "temporal_normalization_accuracy",
            "spatial_predicted", "spatial_ground_truth",
            "spatial_precision", "spatial_recall", "spatial_f1",
            "spatial_geocoding_success_rate", "spatial_mean_distance_error_km", "spatial_accuracy_within_25km",
            "processing_time_seconds", "error"
        ])

        # Overall metrics aggregation
        overall_metrics = OverallMetrics()

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

                for i, item in enumerate(remaining_items):
                    # Evaluate single item
                    item_result = evaluate_single_item(
                        extractor,
                        item,
                        eval_settings.get("temporal_match_mode", "exact"),
                        eval_settings.get("spatial_match_mode", "exact")
                    )

                    # Aggregate metrics
                    item_metrics = item_result.pop("_metrics")
                    overall_metrics.temporal.true_positives += item_metrics["temporal"].true_positives
                    overall_metrics.temporal.false_positives += item_metrics["temporal"].false_positives
                    overall_metrics.temporal.false_negatives += item_metrics["temporal"].false_negatives
                    overall_metrics.temporal.normalization_correct += item_metrics["temporal"].normalization_correct
                    overall_metrics.temporal.normalization_total += item_metrics["temporal"].normalization_total

                    overall_metrics.spatial.true_positives += item_metrics["spatial"].true_positives
                    overall_metrics.spatial.false_positives += item_metrics["spatial"].false_positives
                    overall_metrics.spatial.false_negatives += item_metrics["spatial"].false_negatives
                    overall_metrics.spatial.geocoding_attempted += item_metrics["spatial"].geocoding_attempted
                    overall_metrics.spatial.geocoding_successful += item_metrics["spatial"].geocoding_successful
                    overall_metrics.spatial.distance_errors.extend(item_metrics["spatial"].distance_errors)

                    overall_metrics.total_documents += 1
                    overall_metrics.successful_extractions += 1 if not item_result["error"] else 0
                    overall_metrics.total_processing_time += item_result["processing_time_seconds"]

                    # Write to CSV
                    writer.writerow(item_result)

                    # Flush every checkpoint_interval items
                    if (i + 1) % eval_settings.get("checkpoint_interval", 10) == 0:
                        csvfile.flush()

                    progress.update(task, advance=1)

        console.print(f"\n[green]✓ Evaluation complete![/green]")
        console.print(f"[blue]Results saved to:[/blue] {csv_path}")

        # Display summary metrics
        display_summary_metrics(overall_metrics)

        # Save summary JSON if configured
        if output_settings.get("save_summary", True):
            summary_path = csv_path.with_suffix(".summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(overall_metrics.to_dict(), f, indent=2)
            console.print(f"[blue]Summary saved to:[/blue] {summary_path}")

        # Save aggregated metrics if configured
        if output_settings.get("save_aggregated_metrics", True):
            metrics_path = output_directory / f"metrics_{timestamp}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(overall_metrics.to_dict(), f, indent=2)
            console.print(f"[blue]Aggregated metrics saved to:[/blue] {metrics_path}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
