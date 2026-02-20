"""
Context-Aware Evaluation - Compare baseline vs context-aware extraction.

Evaluates on the context-aware dataset with ground truth annotations.
Compares two modes:
1. Baseline: No context (each chunk extracted independently)
2. Context-aware: With ExtractionContext (maintains state across document chunks)
"""

import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

from stindex import DimensionalExtractor
from stindex.extraction.context_manager import ExtractionContext
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

# Multi-threading configuration
MAX_WORKERS = 10  # Maximum concurrent threads for baseline evaluation


def load_context_aware_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Load context-aware evaluation dataset.

    Returns chunks sorted by document_id and chunk_index for sequential processing.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Sort by document_id and chunk_index to ensure correct order
    chunks.sort(key=lambda x: (x["document_id"], x["chunk_index"]))

    return chunks


def group_chunks_by_document(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by document_id."""
    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk["document_id"]].append(chunk)

    # Sort chunks within each document by chunk_index
    for doc_id in documents:
        documents[doc_id].sort(key=lambda x: x["chunk_index"])

    return dict(documents)


def extract_chunk_baseline(
    extractor: DimensionalExtractor,
    chunk: Dict[str, Any]
) -> Tuple[Any, float]:
    """
    Extract single chunk without context (baseline).

    Returns:
        Tuple of (extraction_result, processing_time)
    """
    start_time = time.time()
    result = extractor.extract(
        text=chunk["text"],
        document_metadata=chunk.get("document_metadata", {})
    )
    processing_time = time.time() - start_time

    return result, processing_time


def extract_document_with_context(
    extractor_with_context: DimensionalExtractor,
    chunks: List[Dict[str, Any]]
) -> List[Tuple[Any, float]]:
    """
    Extract all chunks of a document with shared context.

    Args:
        extractor_with_context: DimensionalExtractor with ExtractionContext initialized
        chunks: List of chunks from the same document (sorted by chunk_index)

    Returns:
        List of (extraction_result, processing_time) tuples, one per chunk
    """
    results = []

    for i, chunk in enumerate(chunks):
        # Update chunk position in context
        if extractor_with_context.extraction_context:
            extractor_with_context.extraction_context.set_chunk_position(
                chunk_index=i,
                total_chunks=len(chunks)
            )

        # Extract with context
        start_time = time.time()
        result = extractor_with_context.extract(
            text=chunk["text"],
            document_metadata=chunk.get("document_metadata", {}),
            update_context=True
        )
        processing_time = time.time() - start_time

        results.append((result, processing_time))

    return results


def evaluate_extraction_result(
    result: Any,
    ground_truth_temporal: List[Dict],
    ground_truth_spatial: List[Dict],
    spatial_match_mode: str = "fuzzy"
) -> Tuple[TemporalMetrics, SpatialMetrics, List[Dict], List[Dict], str]:
    """
    Evaluate extraction result against ground truth.

    Returns:
        Tuple of (temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output)
    """
    temporal_metrics = TemporalMetrics()
    spatial_metrics = SpatialMetrics()
    llm_raw_output = ""
    predicted_temporal = []
    predicted_spatial = []

    # Get raw LLM output
    if result.extraction_config:
        if isinstance(result.extraction_config, dict):
            llm_raw_output = result.extraction_config.get("raw_llm_output", "")
        else:
            llm_raw_output = result.extraction_config.raw_llm_output if hasattr(result.extraction_config, "raw_llm_output") else ""

    if not result.success:
        return temporal_metrics, spatial_metrics, predicted_temporal, predicted_spatial, llm_raw_output

    # Extract predicted entities (handle both dicts and Pydantic models)
    for e in result.temporal_entities:
        if isinstance(e, dict):
            predicted_temporal.append(e)
        else:
            predicted_temporal.append(e.dict() if hasattr(e, 'dict') else e.model_dump())

    for e in result.spatial_entities:
        if isinstance(e, dict):
            predicted_spatial.append(e)
        else:
            predicted_spatial.append(e.dict() if hasattr(e, 'dict') else e.model_dump())

    # Evaluate temporal entities
    if ground_truth_temporal:
        matched_gt = set()
        for pred in predicted_temporal:
            match_found = False
            for i, gt in enumerate(ground_truth_temporal):
                if i in matched_gt:
                    continue

                # Always use value_exact for temporal (compare ISO 8601 values)
                if calculate_temporal_match(pred, gt, "value_exact"):
                    temporal_metrics.true_positives += 1
                    matched_gt.add(i)
                    match_found = True

                    # Check normalization accuracy
                    temporal_metrics.normalization_total += 1
                    if pred.get("normalized") == gt.get("normalized"):
                        temporal_metrics.normalization_correct += 1

                    # Check type accuracy
                    temporal_metrics.type_total += 1
                    pred_type = str(pred.get("normalization_type", "")).lower()
                    gt_type = str(gt.get("normalization_type", "")).lower()
                    if pred_type and gt_type and pred_type == gt_type:
                        temporal_metrics.type_correct += 1

                    break

            if not match_found:
                temporal_metrics.false_positives += 1

        temporal_metrics.false_negatives = len(ground_truth_temporal) - len(matched_gt)

    # Evaluate spatial entities
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

                    # Check type accuracy
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


def get_category_for_chunk(doc_id: str) -> str:
    """
    Determine test case category based on doc_id/document_id.

    Categories:
    - simple: Documents 6-8 (doc_ids with mining_accident, school_opening, product_launch)
    - normal: Documents 9-11 (festival, transport, climate_research)
    - ambiguous: Documents 12-16 (hospital, university_conference, beach_closure, shopping_center, community_meeting)
    - relative: Documents 17-20 (political_rally, art_exhibition, construction_timeline, emergency_response)
    """
    doc_id_lower = doc_id.lower()

    # Simple cases
    if any(x in doc_id_lower for x in ["mining_accident", "school_opening", "product_launch"]):
        return "simple"

    # Normal cases
    elif any(x in doc_id_lower for x in ["festival", "transport", "climate_research"]):
        return "normal"

    # Ambiguous cases
    elif any(x in doc_id_lower for x in ["hospital_emergency", "university_conference", "beach_closure",
                                             "shopping_center", "community_meeting"]):
        return "ambiguous"

    # Relative cases
    elif any(x in doc_id_lower for x in ["political_rally", "art_exhibition", "construction_timeline",
                                             "emergency_response"]):
        return "relative"

    # First 5 documents (existing in dataset)
    elif any(x in doc_id_lower for x in ["cyclone_wa", "uwa_research", "health_outbreak",
                                             "afl_finals", "bushfire_nsw"]):
        return "mixed"

    return "unknown"


def evaluate_single_chunk_baseline(
    extractor: DimensionalExtractor,
    chunk: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single chunk in baseline mode (thread-safe).

    Returns dict with all metrics and results for this chunk.
    """
    # Extract
    result, proc_time = extract_chunk_baseline(extractor, chunk)

    # Evaluate
    temporal_metrics, spatial_metrics, pred_temporal, pred_spatial, raw_output = evaluate_extraction_result(
        result,
        chunk.get("ground_truth", {}).get("temporal", []),
        chunk.get("ground_truth", {}).get("spatial", []),
        spatial_match_mode="fuzzy"
    )

    # Get category
    category = get_category_for_chunk(chunk.get("doc_id", chunk.get("chunk_id", "")))

    # Build result dict
    return {
        "chunk": chunk,
        "result": result,
        "proc_time": proc_time,
        "temporal_metrics": temporal_metrics,
        "spatial_metrics": spatial_metrics,
        "pred_temporal": pred_temporal,
        "pred_spatial": pred_spatial,
        "category": category,
        "csv_row": {
            "doc_id": chunk.get("doc_id", chunk.get("chunk_id", "")),
            "document_id": chunk["document_id"],
            "category": category,
            "text": chunk["text"],
            "temporal_predicted": json.dumps(pred_temporal),
            "temporal_ground_truth": json.dumps(chunk.get("ground_truth", {}).get("temporal", [])),
            "spatial_predicted": json.dumps(pred_spatial),
            "spatial_ground_truth": json.dumps(chunk.get("ground_truth", {}).get("spatial", [])),
            "temporal_precision": temporal_metrics.precision(),
            "temporal_recall": temporal_metrics.recall(),
            "temporal_f1": temporal_metrics.f1_score(),
            "spatial_precision": spatial_metrics.precision(),
            "spatial_recall": spatial_metrics.recall(),
            "spatial_f1": spatial_metrics.f1_score(),
            "processing_time_seconds": proc_time,
            "error": result.error if not result.success else "",
        }
    }


def evaluate_baseline_chunks_parallel(
    extractor: DimensionalExtractor,
    chunks: List[Dict[str, Any]],
    progress,
    task
) -> List[Dict[str, Any]]:
    """
    Evaluate baseline chunks in parallel using ThreadPoolExecutor.

    Returns list of evaluation results.
    """
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(evaluate_single_chunk_baseline, extractor, chunk): chunk
            for chunk in chunks
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            try:
                result = future.result()
                results.append(result)
                progress.update(task, advance=1)
            except Exception as e:
                logger.error(f"Error evaluating chunk: {e}")
                progress.update(task, advance=1)

    return results


def evaluate_single_document_context_aware(
    config: str,
    doc_id: str,
    chunks: List[Dict[str, Any]],
    llm_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single document with context-aware extraction (thread-safe).

    Each document gets its own extractor with its own ExtractionContext.
    Returns list of evaluation results for all chunks in this document.
    """
    llm_overrides = llm_overrides or {}

    # Load config to get context_aware settings
    from stindex.utils.config import load_config_from_file
    eval_config = load_config_from_file(config)
    context_config = eval_config.get("context_aware", {})

    # Create context for this document with config settings
    context = ExtractionContext(
        document_metadata=chunks[0].get("document_metadata", {}),
        enable_nearby_locations=context_config.get("enable_nearby_locations", False),
        max_memory_refs=context_config.get("max_memory_refs", 10)
    )

    # Create context-aware extractor for this document
    extractor = DimensionalExtractor(
        config_path=config,
        extraction_context=context,
        model_name=llm_overrides.get("model_name"),
        temperature=llm_overrides.get("temperature"),
        max_tokens=llm_overrides.get("max_tokens"),
        base_url=llm_overrides.get("base_url"),
    )

    # Extract all chunks with shared context
    doc_results = []
    context_chunk_results = extract_document_with_context(extractor, chunks)

    for chunk, (result, proc_time) in zip(chunks, context_chunk_results):
        # Evaluate
        temporal_metrics, spatial_metrics, pred_temporal, pred_spatial, raw_output = evaluate_extraction_result(
            result,
            chunk.get("ground_truth", {}).get("temporal", []),
            chunk.get("ground_truth", {}).get("spatial", []),
            spatial_match_mode="fuzzy"
        )

        # Get category
        category = get_category_for_chunk(chunk.get("doc_id", chunk.get("chunk_id", "")))

        # Build result dict
        doc_results.append({
            "chunk": chunk,
            "result": result,
            "proc_time": proc_time,
            "temporal_metrics": temporal_metrics,
            "spatial_metrics": spatial_metrics,
            "pred_temporal": pred_temporal,
            "pred_spatial": pred_spatial,
            "category": category,
            "csv_row": {
                "doc_id": chunk.get("doc_id", chunk.get("chunk_id", "")),
                "document_id": doc_id,
                "category": category,
                "text": chunk["text"],
                "temporal_predicted": json.dumps(pred_temporal),
                "temporal_ground_truth": json.dumps(chunk.get("ground_truth", {}).get("temporal", [])),
                "spatial_predicted": json.dumps(pred_spatial),
                "spatial_ground_truth": json.dumps(chunk.get("ground_truth", {}).get("spatial", [])),
                "temporal_precision": temporal_metrics.precision(),
                "temporal_recall": temporal_metrics.recall(),
                "temporal_f1": temporal_metrics.f1_score(),
                "spatial_precision": spatial_metrics.precision(),
                "spatial_recall": spatial_metrics.recall(),
                "spatial_f1": spatial_metrics.f1_score(),
                "processing_time_seconds": proc_time,
                "error": result.error if not result.success else "",
            }
        })

    return doc_results


def evaluate_context_aware_documents_parallel(
    config: str,
    documents: Dict[str, List[Dict[str, Any]]],
    progress,
    task,
    llm_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate documents in parallel, each with its own context.

    Returns list of evaluation results for all chunks.
    """
    llm_overrides = llm_overrides or {}
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all documents
        future_to_doc = {
            executor.submit(evaluate_single_document_context_aware, config, doc_id, chunks, llm_overrides): (doc_id, len(chunks))
            for doc_id, chunks in documents.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_doc):
            try:
                doc_results = future.result()
                all_results.extend(doc_results)
                doc_id, num_chunks = future_to_doc[future]
                progress.update(task, advance=num_chunks)
            except Exception as e:
                logger.error(f"Error evaluating document: {e}")
                doc_id, num_chunks = future_to_doc[future]
                progress.update(task, advance=num_chunks)

    return all_results


def execute_context_aware_evaluation(
    config: str = "extract",
    dataset: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sample_limit: Optional[int] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
):
    """
    Execute context-aware evaluation comparing baseline vs context-aware extraction.

    Args:
        config: Config file name (default: extract.yml)
        dataset: Path to context-aware evaluation dataset
        output_dir: Output directory for results
        sample_limit: Limit number of chunks to process (for testing)
        model: Model name override
        temperature: Sampling temperature override
        max_tokens: Max tokens override
        base_url: LLM server base URL override (for hf provider)
    """
    # Store LLM overrides for use by extractors created in this evaluation run
    _llm_overrides = {
        k: v for k, v in {
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": base_url,
        }.items() if v is not None
    }

    try:
        # Load configuration
        eval_config = load_config_from_file(config)

        # Resolve dataset path
        # Priority: CLI argument > config file > default
        if dataset:
            dataset_path = Path(dataset)
        elif "dataset" in eval_config and "path" in eval_config["dataset"]:
            # Load from config
            config_dataset_path = eval_config["dataset"]["path"]
            if Path(config_dataset_path).is_absolute():
                dataset_path = Path(config_dataset_path)
            else:
                dataset_path = Path(PROJECT_DIR) / config_dataset_path
        else:
            # Default fallback
            dataset_path = Path(PROJECT_DIR) / "data/evaluation/context_aware_eval.json"

        # Resolve sample limit
        # Priority: CLI argument > config file > None
        if sample_limit is None and "dataset" in eval_config and "sample_limit" in eval_config["dataset"]:
            sample_limit = eval_config["dataset"]["sample_limit"]

        if output_dir:
            output_directory = Path(output_dir)
        else:
            output_directory = Path(PROJECT_DIR) / "data/output/evaluations/context_aware"

        output_directory.mkdir(parents=True, exist_ok=True)

        # Load dataset
        console.print(f"\n[bold blue]Loading context-aware dataset:[/bold blue] {dataset_path}")
        all_chunks = load_context_aware_dataset(dataset_path)

        # Apply sample limit
        if sample_limit:
            all_chunks = all_chunks[:sample_limit]

        console.print(f"[green]✓ Loaded {len(all_chunks)} chunks[/green]")

        # Group chunks by document
        documents = group_chunks_by_document(all_chunks)
        console.print(f"[green]✓ Grouped into {len(documents)} documents[/green]")

        # Create extractors
        console.print(f"\n[bold cyan]Initializing extractors...[/bold cyan]")

        # Baseline extractor (no context)
        baseline_extractor = DimensionalExtractor(
            config_path=config,
            model_name=_llm_overrides.get("model_name"),
            temperature=_llm_overrides.get("temperature"),
            max_tokens=_llm_overrides.get("max_tokens"),
            base_url=_llm_overrides.get("base_url"),
        )
        console.print("[green]✓ Baseline extractor ready (no context)[/green]")

        # Metrics storage
        baseline_metrics = OverallMetrics()
        context_metrics = OverallMetrics()

        # Category-specific metrics
        category_baseline_metrics = defaultdict(OverallMetrics)
        category_context_metrics = defaultdict(OverallMetrics)

        # Results storage for CSV
        baseline_results = []
        context_results = []

        # Timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Suppress INFO logs during evaluation
        original_level = logger._core.min_level
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

        # Process documents with progress bar
        total_chunks = len(all_chunks)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Evaluating (Baseline + Context-aware)...", total=total_chunks * 2)

            # ===================================================================
            # BASELINE EXTRACTION (No Context) - Parallel Processing
            # ===================================================================
            console.print(f"\n[bold cyan]Running baseline evaluation (parallel, MAX_WORKERS={MAX_WORKERS})...[/bold cyan]")

            baseline_parallel_results = evaluate_baseline_chunks_parallel(
                baseline_extractor,
                all_chunks,
                progress,
                task
            )

            # Aggregate baseline metrics
            for eval_result in baseline_parallel_results:
                temporal_metrics = eval_result["temporal_metrics"]
                spatial_metrics = eval_result["spatial_metrics"]
                category = eval_result["category"]
                result = eval_result["result"]
                proc_time = eval_result["proc_time"]

                # Aggregate overall metrics
                baseline_metrics.temporal.true_positives += temporal_metrics.true_positives
                baseline_metrics.temporal.false_positives += temporal_metrics.false_positives
                baseline_metrics.temporal.false_negatives += temporal_metrics.false_negatives
                baseline_metrics.temporal.normalization_correct += temporal_metrics.normalization_correct
                baseline_metrics.temporal.normalization_total += temporal_metrics.normalization_total
                baseline_metrics.temporal.type_correct += temporal_metrics.type_correct
                baseline_metrics.temporal.type_total += temporal_metrics.type_total

                baseline_metrics.spatial.true_positives += spatial_metrics.true_positives
                baseline_metrics.spatial.false_positives += spatial_metrics.false_positives
                baseline_metrics.spatial.false_negatives += spatial_metrics.false_negatives
                baseline_metrics.spatial.geocoding_attempted += spatial_metrics.geocoding_attempted
                baseline_metrics.spatial.geocoding_successful += spatial_metrics.geocoding_successful
                baseline_metrics.spatial.distance_errors.extend(spatial_metrics.distance_errors)
                baseline_metrics.spatial.type_correct += spatial_metrics.type_correct
                baseline_metrics.spatial.type_total += spatial_metrics.type_total

                baseline_metrics.total_documents += 1
                baseline_metrics.successful_extractions += 1 if result.success else 0
                baseline_metrics.total_processing_time += proc_time

                # Category-specific metrics
                cat_metrics = category_baseline_metrics[category]
                cat_metrics.temporal.true_positives += temporal_metrics.true_positives
                cat_metrics.temporal.false_positives += temporal_metrics.false_positives
                cat_metrics.temporal.false_negatives += temporal_metrics.false_negatives
                cat_metrics.spatial.true_positives += spatial_metrics.true_positives
                cat_metrics.spatial.false_positives += spatial_metrics.false_positives
                cat_metrics.spatial.false_negatives += spatial_metrics.false_negatives
                cat_metrics.total_documents += 1

                # Store CSV row
                baseline_results.append(eval_result["csv_row"])

            # ===================================================================
            # CONTEXT-AWARE EXTRACTION (With ExtractionContext) - Parallel Processing
            # ===================================================================
            console.print(f"\n[bold cyan]Running context-aware evaluation (parallel, MAX_WORKERS={MAX_WORKERS})...[/bold cyan]")

            context_parallel_results = evaluate_context_aware_documents_parallel(
                config,
                documents,
                progress,
                task,
                llm_overrides=_llm_overrides,
            )

            # Aggregate context-aware metrics
            for eval_result in context_parallel_results:
                temporal_metrics = eval_result["temporal_metrics"]
                spatial_metrics = eval_result["spatial_metrics"]
                category = eval_result["category"]
                result = eval_result["result"]
                proc_time = eval_result["proc_time"]

                # Aggregate overall metrics
                context_metrics.temporal.true_positives += temporal_metrics.true_positives
                context_metrics.temporal.false_positives += temporal_metrics.false_positives
                context_metrics.temporal.false_negatives += temporal_metrics.false_negatives
                context_metrics.temporal.normalization_correct += temporal_metrics.normalization_correct
                context_metrics.temporal.normalization_total += temporal_metrics.normalization_total
                context_metrics.temporal.type_correct += temporal_metrics.type_correct
                context_metrics.temporal.type_total += temporal_metrics.type_total

                context_metrics.spatial.true_positives += spatial_metrics.true_positives
                context_metrics.spatial.false_positives += spatial_metrics.false_positives
                context_metrics.spatial.false_negatives += spatial_metrics.false_negatives
                context_metrics.spatial.geocoding_attempted += spatial_metrics.geocoding_attempted
                context_metrics.spatial.geocoding_successful += spatial_metrics.geocoding_successful
                context_metrics.spatial.distance_errors.extend(spatial_metrics.distance_errors)
                context_metrics.spatial.type_correct += spatial_metrics.type_correct
                context_metrics.spatial.type_total += spatial_metrics.type_total

                context_metrics.total_documents += 1
                context_metrics.successful_extractions += 1 if result.success else 0
                context_metrics.total_processing_time += proc_time

                # Category-specific metrics
                cat_metrics = category_context_metrics[category]
                cat_metrics.temporal.true_positives += temporal_metrics.true_positives
                cat_metrics.temporal.false_positives += temporal_metrics.false_positives
                cat_metrics.temporal.false_negatives += temporal_metrics.false_negatives
                cat_metrics.spatial.true_positives += spatial_metrics.true_positives
                cat_metrics.spatial.false_positives += spatial_metrics.false_positives
                cat_metrics.spatial.false_negatives += spatial_metrics.false_negatives
                cat_metrics.total_documents += 1

                # Store CSV row
                context_results.append(eval_result["csv_row"])
        # Restore logger
        logger.remove()
        logger.add(sys.stderr, level="INFO")

        # Save results to CSV
        console.print(f"\n[bold green]✓ Evaluation complete![/bold green]")

        # CSV columns
        csv_columns = [
            "doc_id", "document_id", "category", "text",
            "temporal_predicted", "temporal_ground_truth",
            "temporal_precision", "temporal_recall", "temporal_f1",
            "spatial_predicted", "spatial_ground_truth",
            "spatial_precision", "spatial_recall", "spatial_f1",
            "processing_time_seconds", "error"
        ]

        # Save baseline results
        baseline_csv = output_directory / f"baseline_{timestamp}.csv"
        with open(baseline_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(baseline_results)
        console.print(f"[blue]Baseline results:[/blue] {baseline_csv}")

        # Save context-aware results
        context_csv = output_directory / f"context_aware_{timestamp}.csv"
        with open(context_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(context_results)
        console.print(f"[blue]Context-aware results:[/blue] {context_csv}")

        # Display comparison summary
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]OVERALL COMPARISON: Baseline vs Context-Aware[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]")

        display_comparison_table(baseline_metrics, context_metrics, "Overall")

        # Display category-specific comparison
        for category in ["simple", "normal", "ambiguous", "relative", "mixed"]:
            if category in category_baseline_metrics:
                console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
                console.print(f"[bold magenta]CATEGORY: {category.upper()}[/bold magenta]")
                console.print(f"[bold magenta]{'='*80}[/bold magenta]")
                display_comparison_table(
                    category_baseline_metrics[category],
                    category_context_metrics[category],
                    category.capitalize()
                )

        # Save summary JSON
        summary = {
            "overall": {
                "baseline": baseline_metrics.to_dict(),
                "context_aware": context_metrics.to_dict(),
                "improvement": calculate_improvement(baseline_metrics, context_metrics)
            },
            "by_category": {}
        }

        for category in category_baseline_metrics:
            summary["by_category"][category] = {
                "baseline": category_baseline_metrics[category].to_dict(),
                "context_aware": category_context_metrics[category].to_dict(),
                "improvement": calculate_improvement(
                    category_baseline_metrics[category],
                    category_context_metrics[category]
                )
            }

        summary_path = output_directory / f"comparison_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        console.print(f"\n[blue]Comparison summary:[/blue] {summary_path}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def display_comparison_table(
    baseline_metrics: OverallMetrics,
    context_metrics: OverallMetrics,
    title: str = "Comparison"
):
    """Display comparison table between baseline and context-aware metrics."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="yellow", width=30)
    table.add_column("Baseline", justify="right", style="white")
    table.add_column("Context-Aware", justify="right", style="white")
    table.add_column("Δ", justify="right", style="green")

    # Temporal metrics
    table.add_row("[bold]Temporal Metrics[/bold]", "", "", "")

    baseline_t_p = baseline_metrics.temporal.precision()
    context_t_p = context_metrics.temporal.precision()
    table.add_row("Precision", f"{baseline_t_p:.4f}", f"{context_t_p:.4f}", f"+{context_t_p - baseline_t_p:.4f}")

    baseline_t_r = baseline_metrics.temporal.recall()
    context_t_r = context_metrics.temporal.recall()
    table.add_row("Recall", f"{baseline_t_r:.4f}", f"{context_t_r:.4f}", f"+{context_t_r - baseline_t_r:.4f}")

    baseline_t_f1 = baseline_metrics.temporal.f1_score()
    context_t_f1 = context_metrics.temporal.f1_score()
    table.add_row("F1 Score", f"{baseline_t_f1:.4f}", f"{context_t_f1:.4f}", f"+{context_t_f1 - baseline_t_f1:.4f}")

    baseline_t_norm = baseline_metrics.temporal.normalization_accuracy()
    context_t_norm = context_metrics.temporal.normalization_accuracy()
    table.add_row("Normalization Acc", f"{baseline_t_norm:.4f}", f"{context_t_norm:.4f}", f"+{context_t_norm - baseline_t_norm:.4f}")

    table.add_row("", "", "", "")

    # Spatial metrics
    table.add_row("[bold]Spatial Metrics[/bold]", "", "", "")

    baseline_s_p = baseline_metrics.spatial.precision()
    context_s_p = context_metrics.spatial.precision()
    table.add_row("Precision", f"{baseline_s_p:.4f}", f"{context_s_p:.4f}", f"+{context_s_p - baseline_s_p:.4f}")

    baseline_s_r = baseline_metrics.spatial.recall()
    context_s_r = context_metrics.spatial.recall()
    table.add_row("Recall", f"{baseline_s_r:.4f}", f"{context_s_r:.4f}", f"+{context_s_r - baseline_s_r:.4f}")

    baseline_s_f1 = baseline_metrics.spatial.f1_score()
    context_s_f1 = context_metrics.spatial.f1_score()
    table.add_row("F1 Score", f"{baseline_s_f1:.4f}", f"{context_s_f1:.4f}", f"+{context_s_f1 - baseline_s_f1:.4f}")

    baseline_s_geo = baseline_metrics.spatial.geocoding_success_rate()
    context_s_geo = context_metrics.spatial.geocoding_success_rate()
    table.add_row("Geocoding Success", f"{baseline_s_geo:.4f}", f"{context_s_geo:.4f}", f"+{context_s_geo - baseline_s_geo:.4f}")

    baseline_s_dist = baseline_metrics.spatial.mean_distance_error()
    context_s_dist = context_metrics.spatial.mean_distance_error()
    table.add_row("Mean Distance Error (km)", f"{baseline_s_dist:.2f}", f"{context_s_dist:.2f}", f"{context_s_dist - baseline_s_dist:.2f}")

    table.add_row("", "", "", "")

    # Combined
    baseline_combined = baseline_metrics.combined_f1()
    context_combined = context_metrics.combined_f1()
    table.add_row("[bold]Combined F1[/bold]", f"[bold]{baseline_combined:.4f}[/bold]",
                  f"[bold]{context_combined:.4f}[/bold]",
                  f"[bold green]+{context_combined - baseline_combined:.4f}[/bold green]")

    console.print(table)


def calculate_improvement(baseline: OverallMetrics, context: OverallMetrics) -> Dict[str, float]:
    """Calculate improvement percentages."""
    return {
        "temporal_f1_delta": context.temporal.f1_score() - baseline.temporal.f1_score(),
        "temporal_f1_pct_improvement": ((context.temporal.f1_score() - baseline.temporal.f1_score()) / baseline.temporal.f1_score() * 100) if baseline.temporal.f1_score() > 0 else 0,
        "spatial_f1_delta": context.spatial.f1_score() - baseline.spatial.f1_score(),
        "spatial_f1_pct_improvement": ((context.spatial.f1_score() - baseline.spatial.f1_score()) / baseline.spatial.f1_score() * 100) if baseline.spatial.f1_score() > 0 else 0,
        "combined_f1_delta": context.combined_f1() - baseline.combined_f1(),
        "combined_f1_pct_improvement": ((context.combined_f1() - baseline.combined_f1()) / baseline.combined_f1() * 100) if baseline.combined_f1() > 0 else 0,
    }
