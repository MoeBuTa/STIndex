#!/usr/bin/env python3
"""
Corpus extraction CLI - extracts dimensional metadata from corpus documents.

This module processes corpus documents through the dimensional extraction pipeline,
generating both raw extraction results and hierarchical dimensional labels for RAG.

Usage:
    python -m stindex.exe.extract_corpus \
        --config cfg/extraction/corpus_extraction_textbook.yml

    # Test mode (100 documents)
    python -m stindex.exe.extract_corpus \
        --config cfg/extraction/corpus_extraction_textbook_test.yml
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.extraction.dimension_loader import DimensionConfigLoader
from stindex.warehouse.chunk_labeler import DimensionalChunkLabeler
from stindex.llm.manager import LLMManager
from stindex.utils.config import load_config_from_file
from stindex.utils.timing import TimingStats


def load_corpus(corpus_path: str, sample_limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Load corpus from JSONL file.

    Args:
        corpus_path: Path to corpus JSONL file
        sample_limit: Optional limit on number of documents to load
        offset: Skip first N documents (for parallel processing)

    Returns:
        List of corpus documents
    """
    corpus = []
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            # Skip documents before offset
            if i < offset:
                continue
            # Stop if we've reached the limit
            if sample_limit and len(corpus) >= sample_limit:
                break
            data = json.loads(line)
            corpus.append(data)
    return corpus


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Save extraction results to JSONL file.

    Args:
        results: List of extraction result dicts
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "extraction_results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    return output_file


class TimingWriter:
    """Incremental JSONL writer for per-document timing data."""

    def __init__(self, output_path: Path, num_gpus: int = 1, append_mode: bool = False):
        self.output_path = output_path
        self.num_gpus = num_gpus
        mode = 'a' if append_mode else 'w'
        self.file = open(output_path, mode)
        self.timing_stats = TimingStats(name="corpus_extraction", num_gpus=num_gpus)

    def write_document_timing(self, doc_timing: Dict[str, Any]):
        """Write single document timing to JSONL."""
        self.file.write(json.dumps(doc_timing) + '\n')
        self.file.flush()

        # Aggregate for summary
        if "total_duration_seconds" in doc_timing:
            self.timing_stats.add_timing("total", doc_timing["total_duration_seconds"])

        for component, duration in doc_timing.get("components", {}).items():
            if isinstance(duration, dict) and "duration_seconds" in duration:
                self.timing_stats.add_timing(component, duration["duration_seconds"])

    def close(self):
        """Close file and save summary."""
        self.file.close()

        # Save aggregate summary
        summary_path = self.output_path.parent / f"{self.output_path.stem}_summary.json"
        summary = self.timing_stats.get_summary()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✓ Timing summary saved: {summary_path}")


async def process_documents_concurrent(
    corpus: List[Dict[str, Any]],
    extractor: 'DimensionalExtractor',
    labeler: 'DimensionalChunkLabeler',
    output_file: Path,
    timing_writer: 'TimingWriter',
    corpus_name: str,
    num_gpus: int,
    max_concurrent: int = 32,
    failed_chunks_file: Path = None
) -> tuple:
    """
    Process documents with controlled concurrency for better throughput.

    Uses asyncio + ThreadPoolExecutor to submit multiple extraction requests
    concurrently, maximizing GPU utilization via vLLM's continuous batching.

    Args:
        corpus: List of document dicts
        extractor: DimensionalExtractor instance
        labeler: DimensionalChunkLabeler instance
        output_file: Path to output JSONL file
        timing_writer: TimingWriter for per-document timing
        corpus_name: Name of corpus for metadata
        num_gpus: Number of GPUs (for timing stats)
        max_concurrent: Max concurrent extractions (default: 32)

    Returns:
        tuple of (successful_count, failed_count)
    """
    sem = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_event_loop()

    successful = 0
    failed = 0
    no_labels = 0
    lock = asyncio.Lock()  # For thread-safe file writing

    async def process_one(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document with rate limiting."""
        nonlocal successful, failed, no_labels

        async with sem:
            doc_start = time.time()
            doc_id = doc.get('doc_id', '')
            title = doc.get('title', '')
            text = doc.get('contents', '')

            try:
                # Run sync extract in thread pool (LLM call is I/O bound)
                result = await loop.run_in_executor(None, extractor.extract, text)

                # Labeling (CPU bound, fast)
                labels = labeler.label_chunk(
                    chunk_text=text,
                    extraction_result=result,
                    chunk_index=0
                )

                # Build result dict
                result_dict = {
                    'doc_id': doc_id,
                    'doc_metadata': {
                        'title': title,
                        'source': doc.get('source', 'unknown'),
                        'corpus': corpus_name,
                        'original_id': doc.get('id', doc_id)
                    },
                    'text': text,
                    'schema_metadata': labels.to_dict()
                }

                # Timing
                doc_duration = time.time() - doc_start
                component_timings = {"extraction": {"duration_seconds": round(doc_duration, 3)}}
                if hasattr(result, 'component_timings') and result.component_timings:
                    component_timings["extraction"]["components"] = result.component_timings

                timing_entry = {
                    "doc_id": doc_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_duration_seconds": round(doc_duration, 3),
                    "num_gpus": num_gpus,
                    "gpu_hours": round((doc_duration / 3600) * num_gpus, 6),
                    "components": component_timings
                }

                # Check if any dimensions were extracted
                schema_data = labels.to_dict()
                has_labels = any(v for v in schema_data.values() if v)

                return {"result": result_dict, "timing": timing_entry, "success": True, "has_labels": has_labels, "doc_id": doc_id}

            except Exception as e:
                logger.error(f"Failed to extract from document {doc_id}: {e}")
                return {"doc_id": doc_id, "error": str(e), "success": False}

    # Submit all tasks
    tasks = [asyncio.create_task(process_one(doc)) for doc in corpus]

    # Process results as they complete, writing to file immediately
    with open(output_file, 'a') as outfile:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting (concurrent)"):
            result = await coro

            if result.get("success"):
                outfile.write(json.dumps(result["result"]) + '\n')
                outfile.flush()
                timing_writer.write_document_timing(result["timing"])
                successful += 1

                # Track chunks with no labels
                if not result.get("has_labels", True):
                    no_labels += 1
                    if failed_chunks_file:
                        with open(failed_chunks_file, 'a') as f:
                            f.write(json.dumps({"doc_id": result.get("doc_id"), "reason": "no_dimensions_after_retry"}) + '\n')
            else:
                failed += 1

    return successful, failed, no_labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract dimensional metadata from corpus documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full extraction (all 125K textbooks)
  python -m stindex.exe.extract_corpus --config cfg/extraction/corpus_extraction_textbook.yml

  # Test extraction (100 documents)
  python -m stindex.exe.extract_corpus --config cfg/extraction/corpus_extraction_textbook_test.yml

Config file format:
  input:
    corpus_path: "data/original/medcorp/train_textbooks_only.jsonl"

  extraction:
    dimension_config: "data/schema_discovery_mirage_textbook/final_schema"
    batch_size: 100
    sample_limit: null

  llm:
    llm_provider: "hf"
    model_name: null

  output:
    output_dir: "data/extraction_results_textbook"
"""
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")

    # Parallel processing arguments (override config values)
    parser.add_argument("--worker-id", type=int, help="Worker ID for parallel processing")
    parser.add_argument("--offset", type=int, help="Skip first N documents")
    parser.add_argument("--limit", type=int, help="Process only N documents")
    parser.add_argument("--num-gpus", type=int, help="Total number of GPUs (for timing stats)")
    parser.add_argument("--base-url", type=str, help="LLM server base URL (for parallel servers)")
    parser.add_argument("--concurrent", type=int, default=32,
                        help="Max concurrent extractions (default: 32). "
                             "Controls how many documents are processed in parallel.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode - only show progress bar and summary")

    args = parser.parse_args()

    # Configure logging based on quiet mode
    if args.quiet:
        import logging
        import warnings
        # Disable all loguru logs except ERROR
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
        # Suppress other loggers
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("swift").setLevel(logging.ERROR)
        logging.getLogger("stindex").setLevel(logging.ERROR)
        # Suppress warnings
        warnings.filterwarnings("ignore")

    # Helper for conditional printing
    def qprint(*pargs, **kwargs):
        """Print only if not in quiet mode."""
        if not args.quiet:
            print(*pargs, **kwargs)

    # Load config
    qprint(f"Loading config from: {args.config}")
    config = load_config_from_file(args.config)

    corpus_path = config['input']['corpus_path']
    extraction_config = config.get('extraction', {})
    dimension_config_path = extraction_config['dimension_config']
    dimension_overrides = extraction_config.get('dimension_overrides')  # NEW: Read override path
    llm_config = config.get('llm', {})
    output_dir = config['output']['output_dir']
    sample_limit = extraction_config.get('sample_limit')
    batch_size = extraction_config.get('batch_size', 100)

    # Override config with CLI arguments for parallel processing
    if args.offset is not None:
        config.setdefault('input', {})['offset'] = args.offset
    if args.limit is not None:
        config.setdefault('extraction', {})['limit'] = args.limit
    if args.worker_id is not None:
        config.setdefault('indexing', {})['worker_id'] = args.worker_id
    if args.num_gpus is not None:
        config.setdefault('indexing', {})['num_gpus'] = args.num_gpus
    if args.base_url is not None:
        config.setdefault('llm', {})['base_url'] = args.base_url

    # Validate inputs
    if not Path(corpus_path).exists():
        qprint(f"✗ Corpus file not found: {corpus_path}")
        return 1

    # Check if dimension config exists (discovered schema)
    dimension_config_file = Path(dimension_config_path)
    if not dimension_config_file.suffix:
        dimension_config_file = dimension_config_file.with_suffix('.yml')

    if not dimension_config_file.exists():
        qprint(f"✗ Dimension config not found: {dimension_config_file}")
        qprint(f"  Please run schema discovery first:")
        qprint(f"  python -m stindex.exe.discover_schema --config cfg/discovery/textbook_schema.yml")
        return 1

    qprint(f"\nConfiguration:")
    qprint(f"  Corpus: {corpus_path}")
    qprint(f"  Dimension config: {dimension_config_path}")
    qprint(f"  Sample limit: {sample_limit or 'None (all documents)'}")
    qprint(f"  Batch size: {batch_size}")
    qprint(f"  LLM provider: {llm_config.get('llm_provider', 'openai')}")
    qprint(f"  Model: {llm_config.get('model_name', 'default')}")
    qprint(f"  Output directory: {output_dir}")

    # Load dimension config
    qprint(f"\n⚙️  Loading dimension configuration...")
    loader = DimensionConfigLoader()
    dimension_config = loader.load_dimension_config(dimension_config_path)
    enabled_dims = loader.get_enabled_dimensions(dimension_config)
    qprint(f"   Loaded {len(enabled_dims)} enabled dimensions:")
    for dim_name in sorted(enabled_dims.keys()):
        qprint(f"     • {dim_name}")

    # Initialize LLM and extractor
    qprint(f"\n⚙️  Initializing dimensional extractor...")

    # Get dimension discovery config options
    enable_discovery = extraction_config.get('enable_dimension_discovery', False)
    discovery_threshold = extraction_config.get('discovery_confidence_threshold', 0.9)
    schema_output_path = extraction_config.get('schema_output_path', dimension_config_path)

    if enable_discovery:
        qprint(f"   Dimension discovery: ENABLED (threshold: {discovery_threshold})")
        qprint(f"   Schema output: {schema_output_path}")
    else:
        qprint(f"   Dimension discovery: DISABLED")

    # Build extraction config path from llm_config
    # DimensionalExtractor will create LLMManager internally
    extractor = DimensionalExtractor(
        config_path="extract",  # Use main extraction config
        dimension_config_path=dimension_config_path,
        dimension_overrides=dimension_overrides,  # NEW: Pass override config
        prompt_mode="corpus",  # Use corpus-optimized prompts for textbook extraction
        enable_dimension_discovery=enable_discovery,
        discovery_confidence_threshold=discovery_threshold,
        schema_output_path=schema_output_path if enable_discovery else None
    )

    # Override LLM config if specified in corpus extraction config
    if llm_config:
        extractor.llm_manager = LLMManager(llm_config)

    # Initialize labeler with dimension config
    qprint(f"⚙️  Initializing dimensional labeler...")
    labeler = DimensionalChunkLabeler(
        dimension_config=dimension_config,
        enabled_dimensions=set(extractor.dimensions.keys())  # Only output enabled dimensions
    )

    # Prepare output directory and get worker_id early (needed for resume logic)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get worker_id and GPU count for parallel processing
    worker_id = config.get('indexing', {}).get('worker_id', None)
    num_gpus = config.get('indexing', {}).get('num_gpus', 1)

    # Load corpus with offset/limit support for parallel processing
    qprint(f"\n📖 Loading corpus...")
    # Read offset/limit from config (may have been overridden by CLI args)
    offset = config.get('input', {}).get('offset', 0)
    limit = config.get('extraction', {}).get('limit', sample_limit)

    # Resume functionality: Load ALL completed doc_ids from ALL output files
    resume_enabled = extraction_config.get('resume', False)
    completed_doc_ids = set()

    if resume_enabled:
        qprint(f"\n🔄 Resume mode enabled - scanning for completed documents...")

        # Scan ALL worker output files in this directory (no timestamp in names)
        all_output_files = sorted(output_path.glob("corpus_extraction*.jsonl"))
        # Exclude timing files
        all_output_files = [f for f in all_output_files if "_timing" not in f.name]

        for output_file in all_output_files:
            if output_file.exists() and output_file.stat().st_size > 0:
                try:
                    with open(output_file) as f:
                        for line in f:
                            data = json.loads(line)
                            completed_doc_ids.add(data['doc_id'])
                except Exception as e:
                    logger.warning(f"Failed to read {output_file.name}: {e}")
                    continue

        if completed_doc_ids:
            qprint(f"   Found {len(completed_doc_ids):,} completed documents across all workers")
            qprint(f"   Will skip these documents and process only missing ones")
        else:
            qprint(f"   No completed documents found - starting fresh")

    # Load full corpus (will filter out completed docs after loading)
    full_corpus = load_corpus(corpus_path, sample_limit=limit, offset=offset)

    # Filter out completed documents
    if completed_doc_ids:
        qprint(f"   Filtering out {len(completed_doc_ids):,} completed documents...")
        corpus = [doc for doc in full_corpus if doc.get('doc_id') not in completed_doc_ids]
        qprint(f"   After filtering: {len(corpus):,} documents remaining to process")
        qprint(f"   Skipped: {len(full_corpus) - len(corpus):,} already completed documents")
    else:
        corpus = full_corpus
        if offset > 0 or limit:
            qprint(f"   Loaded {len(corpus):,} documents (offset={offset}, limit={limit})")
        else:
            qprint(f"   Loaded {len(corpus):,} documents")

    # Always print startup summary
    print(f"\n📊 Starting extraction: {len(corpus):,} documents, {len(enabled_dims)} dimensions")

    # Extract with metadata construction
    qprint(f"\n🔍 Extracting dimensions from corpus...")

    # Get batch size and corpus name from config
    batch_size = config.get('extraction', {}).get('batch_size', 100)
    corpus_name = config.get('indexing', {}).get('corpus_name', 'unknown')

    # Determine output file name and mode (no datetime in filenames)
    if worker_id is not None:
        output_file = output_path / f"corpus_extraction_worker{worker_id}.jsonl"
        timing_file = output_path / f"corpus_extraction_worker{worker_id}_timing.jsonl"
    else:
        output_file = output_path / "corpus_extraction.jsonl"
        timing_file = output_path / "corpus_extraction_timing.jsonl"

    # Check if resuming existing file
    if completed_doc_ids and output_file.exists():
        file_mode = 'a'
        qprint(f"   Output file (append mode): {output_file}")
    else:
        file_mode = 'w'
        qprint(f"   Output file: {output_file}")

    # Initialize timing writer with GPU count (append mode if resuming)
    timing_writer = TimingWriter(timing_file, num_gpus=num_gpus, append_mode=(file_mode == 'a'))

    # Create failed chunks file path
    failed_chunks_file = output_path / "failed_chunks.jsonl"

    # Choose processing mode based on --concurrent flag
    no_labels = 0
    if args.concurrent > 1:
        # Concurrent processing for better throughput
        qprint(f"   Mode: CONCURRENT (max_concurrent={args.concurrent})")
        qprint(f"   This maximizes GPU utilization via vLLM continuous batching")

        # Create empty file if fresh start
        if file_mode == 'w':
            open(output_file, 'w').close()

        successful, failed, no_labels = asyncio.run(
            process_documents_concurrent(
                corpus=corpus,
                extractor=extractor,
                labeler=labeler,
                output_file=output_file,
                timing_writer=timing_writer,
                corpus_name=corpus_name,
                num_gpus=num_gpus,
                max_concurrent=args.concurrent,
                failed_chunks_file=failed_chunks_file
            )
        )
    else:
        # Sequential processing (original behavior)
        qprint(f"   Mode: SEQUENTIAL (use --concurrent N for faster processing)")

        batch_results = []
        successful = 0
        failed = 0

        with open(output_file, file_mode) as outfile:
            for idx, doc in enumerate(tqdm(corpus, desc="Extracting dimensions")):
                doc_start = time.time()  # Start document timing

                try:
                    text = doc.get('contents', '')
                    doc_id = doc.get('doc_id', '')
                    title = doc.get('title', '')

                    # Track component timings
                    component_timings = {}

                    # Extract dimensions
                    extraction_start = time.time()
                    result = extractor.extract(text)
                    component_timings["extraction"] = {
                        "duration_seconds": round(time.time() - extraction_start, 3)
                    }

                    # Get component breakdown from result if available
                    if hasattr(result, 'component_timings') and result.component_timings:
                        component_timings["extraction"]["components"] = result.component_timings

                    # Generate dimensional labels
                    labeling_start = time.time()
                    labels = labeler.label_chunk(
                        chunk_text=text,
                        extraction_result=result,
                        chunk_index=0
                    )
                    component_timings["labeling"] = {
                        "duration_seconds": round(time.time() - labeling_start, 3)
                    }

                    # Build output with separated doc_metadata and schema_metadata
                    schema_data = labels.to_dict()
                    result_dict = {
                        'doc_id': doc_id,
                        'doc_metadata': {
                            'title': title,
                            'source': doc.get('source', 'unknown'),
                            'corpus': corpus_name,
                            'original_id': doc.get('id', doc_id)
                        },
                        'text': text,  # Full document content (not chunked)
                        'schema_metadata': schema_data  # Extracted dimensional metadata
                    }
                    batch_results.append(result_dict)
                    successful += 1

                    # Check if any dimensions were extracted
                    has_labels = any(v for v in schema_data.values() if v)
                    if not has_labels:
                        no_labels += 1
                        with open(failed_chunks_file, 'a') as f:
                            f.write(json.dumps({"doc_id": doc_id, "reason": "no_dimensions_after_retry"}) + '\n')

                    # Write timing data
                    doc_duration = time.time() - doc_start
                    gpu_hours = (doc_duration / 3600) * num_gpus
                    timing_entry = {
                        "doc_id": doc_id,
                        "timestamp": datetime.now().isoformat(),
                        "total_duration_seconds": round(doc_duration, 3),
                        "num_gpus": num_gpus,
                        "gpu_hours": round(gpu_hours, 6),
                        "components": component_timings
                    }
                    timing_writer.write_document_timing(timing_entry)

                    # Save batch when it reaches batch_size
                    if len(batch_results) >= batch_size:
                        for item in batch_results:
                            outfile.write(json.dumps(item) + '\n')
                        outfile.flush()  # Ensure data is written to disk
                        batch_results = []

                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to extract from document {doc.get('doc_id', 'unknown')}: {e}")
                    # Continue processing other documents

            # Save remaining results
            if batch_results:
                for item in batch_results:
                    outfile.write(json.dumps(item) + '\n')
                outfile.flush()

    # Close timing writer
    timing_writer.close()

    qprint(f"\n💾 Results saved incrementally to: {output_file}")
    qprint(f"⏱️  Timing data saved:")
    qprint(f"  Per-document: {timing_file}")
    qprint(f"  Summary: {timing_file.parent / f'{timing_file.stem}_summary.json'}")

    # Print summary (always show)
    print(f"\n{'=' * 60}")
    print(f"✅ Extraction completed!")
    print(f"{'=' * 60}")
    print(f"  Documents: {len(corpus):,} | Success: {successful:,} | Failed: {failed:,} | No dims: {no_labels:,}")
    print(f"  Success rate: {100 * successful / len(corpus):.1f}%")
    print(f"  Output: {output_file}")
    if no_labels > 0:
        print(f"  ⚠️  Failed chunks: {failed_chunks_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
