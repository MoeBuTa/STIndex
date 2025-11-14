#!/usr/bin/env python3
"""
Ablation Study: Context-Aware Module Evaluation

Runs extraction on preprocessed public health chunks with:
- Baseline (context-aware disabled)
- Context-aware (context-aware enabled)

Uses existing preprocessed chunks from case_studies/public_health/data/chunks/
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.utils.config import load_config_from_file
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configuration
CHUNKS_FILE = Path("case_studies/public_health/data/chunks/preprocessed_chunks.json")
OUTPUT_BASE = Path("data/output/ablation_study")
DIMENSION_CONFIG = "case_studies/public_health/config/health_dimensions.yml"

CONDITIONS = [
    {
        "name": "OpenAI Baseline (No Context)",
        "config": "cfg/ablation/openai_baseline.yml",
        "output_dir": "openai_baseline",
        "short_name": "baseline"
    },
    {
        "name": "OpenAI Context-Aware",
        "config": "cfg/ablation/openai_context.yml",
        "output_dir": "openai_context",
        "short_name": "context"
    },
]


def load_preprocessed_chunks():
    """Load preprocessed chunks from case study."""
    logger.info(f"Loading chunks from: {CHUNKS_FILE}")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    logger.info(f"✓ Loaded {len(chunks)} chunks from {len(set(c['document_id'] for c in chunks))} documents")
    return chunks


def run_extraction_condition(condition, chunks):
    """Run extraction for one condition."""
    logger.info("=" * 80)
    logger.info(f"Running: {condition['name']}")
    logger.info("=" * 80)

    # Create output directory
    output_dir = OUTPUT_BASE / condition["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config to check context-aware settings
    logger.info(f"Config: {condition['config']}")
    logger.info(f"Dimension config: {DIMENSION_CONFIG}")

    from stindex.utils.config import load_config_from_file
    config = load_config_from_file(condition['config'])
    context_aware_config = config.get("context_aware", {})

    # Check if context-aware is enabled
    context_enabled = context_aware_config.get("enabled", False)
    logger.info(f"Context-aware extraction: {'ENABLED ✅' if context_enabled else 'DISABLED ❌'}")

    # Group chunks by document for context management
    from collections import defaultdict
    chunks_by_doc = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk["document_id"]
        chunks_by_doc[doc_id].append(chunk)

    logger.info(f"Grouped {len(chunks)} chunks into {len(chunks_by_doc)} documents")

    # Define function to process a single document (all its chunks sequentially)
    def process_document(doc_id, doc_chunks):
        """Process all chunks for a single document (maintains context within document)."""
        doc_results = []
        doc_successful = 0
        doc_failed = 0

        # Sort chunks by index for this document
        doc_chunks_sorted = sorted(doc_chunks, key=lambda c: c["chunk_index"])

        # Create extraction context for this document if context-aware is enabled
        extraction_context = None
        if context_enabled:
            from stindex.extraction.context_manager import ExtractionContext
            extraction_context = ExtractionContext(
                document_metadata=doc_chunks_sorted[0].get("document_metadata", {}),
                max_memory_refs=context_aware_config.get("max_memory_refs", 10),
                enable_nearby_locations=context_aware_config.get("enable_nearby_locations", False),
            )
            logger.debug(f"Created ExtractionContext for document: {doc_id}")

        # Create extractor for this document (with or without context)
        extractor = DimensionalExtractor(
            config_path=condition["config"],
            dimension_config_path=DIMENSION_CONFIG,
            extraction_context=extraction_context  # ✅ Pass context!
        )

        # Process all chunks for this document (sequentially to maintain context)
        for chunk_idx, chunk in enumerate(doc_chunks_sorted):
            chunk_id = chunk["chunk_id"]

            logger.debug(f"[Doc {doc_id}] Processing chunk {chunk_idx+1}/{len(doc_chunks_sorted)}: {chunk_id}")

            # Update context position
            if extraction_context:
                extraction_context.set_chunk_position(
                    chunk["chunk_index"],
                    chunk["total_chunks"]
                )

            try:
                # Extract with document metadata for context
                result = extractor.extract(
                    text=chunk["text"],
                    document_metadata=chunk.get("document_metadata", {})
                )

                # Store result with chunk info (match case study output structure)
                result_dict = {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "document_id": chunk["document_id"],
                    "document_title": chunk.get("document_title"),
                    "document_metadata": chunk.get("document_metadata"),
                    "text": chunk["text"],  # Add text field like case study
                    "extraction": {
                        "success": result.success,
                        "entities": {}  # Use entities dict like case study
                    }
                }

                # Add all dimensional entities (temporal, spatial, custom dimensions)
                if hasattr(result, 'entities') and result.entities:
                    for dim_name, entities in result.entities.items():
                        result_dict["extraction"]["entities"][dim_name] = entities

                # Add context usage if available
                if hasattr(result, 'context_usage'):
                    result_dict["extraction"]["context_usage"] = result.context_usage

                doc_results.append(result_dict)

                if result.success:
                    doc_successful += 1
                    # Count entities across all dimensions
                    entity_counts = []
                    if hasattr(result, 'entities') and result.entities:
                        for dim_name, entities in result.entities.items():
                            if entities:
                                entity_counts.append(f"{dim_name}: {len(entities)}")
                    logger.debug(f"  ✓ Success - {', '.join(entity_counts) if entity_counts else 'no entities'}")
                else:
                    doc_failed += 1
                    logger.warning(f"  ✗ Failed - {result.error}")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                doc_failed += 1
                doc_results.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk["document_id"],
                    "extraction": {
                        "success": False,
                        "error": str(e)
                    }
                })

        logger.info(f"✓ Document {doc_id} complete: {doc_successful}/{len(doc_chunks_sorted)} successful")
        return doc_results, doc_successful, doc_failed

    # Process all documents in parallel (but chunks within each document are sequential)
    logger.info(f"\nExtracting from {len(chunks)} chunks across {len(chunks_by_doc)} documents...")
    logger.info(f"Using parallel processing with max_workers=10 (documents in parallel, chunks sequential within doc)")

    results = []
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=min(10, len(chunks_by_doc))) as executor:
        # Submit all documents for parallel processing
        future_to_doc = {
            executor.submit(process_document, doc_id, doc_chunks): doc_id
            for doc_id, doc_chunks in chunks_by_doc.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_doc):
            doc_id = future_to_doc[future]
            try:
                doc_results, doc_successful, doc_failed = future.result()
                results.extend(doc_results)
                successful += doc_successful
                failed += doc_failed
            except Exception as e:
                logger.error(f"Document {doc_id} failed with error: {e}")
                failed += len(chunks_by_doc[doc_id])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"extraction_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"\n✓ Extraction complete!")
    logger.info(f"  - Successful: {successful}/{len(chunks)}")
    logger.info(f"  - Failed: {failed}/{len(chunks)}")
    logger.info(f"  - Results saved to: {results_file}")

    return results, results_file


def main():
    """Run ablation study."""
    print("\n" + "=" * 80)
    print("STIndex Ablation Study: Context-Aware Module Evaluation")
    print("=" * 80)

    # Load chunks once
    try:
        chunks = load_preprocessed_chunks()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("\nPlease run the public health case study preprocessing first:")
        logger.info("  cd case_studies/public_health/scripts")
        logger.info("  python run_case_study.py")
        sys.exit(1)

    # Run all conditions
    all_results = {}

    for i, condition in enumerate(CONDITIONS, 1):
        print(f"\n{'='*80}")
        print(f"Condition {i}/{len(CONDITIONS)}: {condition['name']}")
        print(f"{'='*80}")

        results, results_file = run_extraction_condition(condition, chunks)
        all_results[condition['short_name']] = {
            "results": results,
            "file": results_file
        }

    # Summary
    print("\n" + "=" * 80)
    print("Ablation Study Complete!")
    print("=" * 80)

    for condition in CONDITIONS:
        short_name = condition['short_name']
        results = all_results[short_name]["results"]
        file = all_results[short_name]["file"]

        successful = sum(1 for r in results if r["extraction"]["success"])

        print(f"\n{condition['name']}:")
        print(f"  - Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"  - Results: {file}")

    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Run comparison metrics: python scripts/compare_ablation.py")
    print("  2. Analyze results in: data/output/ablation_study/")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
