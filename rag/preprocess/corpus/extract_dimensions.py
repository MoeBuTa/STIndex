#!/usr/bin/env python3
"""
Extract dimensional entities from MedCorp corpus using discovered schema.

This module processes the full MedCorp corpus (Textbooks + StatPearls)
and extracts dimensional entities using the schema discovered from MIRAGE questions.

Usage:
    python -m rag.preprocess.corpus.extract_dimensions \
        --corpus data/original/medcorp/train.jsonl \
        --schema data/schema_discovery_mirage_v2/final_schema \
        --output data/extraction_results \
        --llm-provider hf \
        --batch-size 100 \
        --sample-limit 1000  # For testing
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger

from stindex.extraction.dimensional_extraction import DimensionalExtractor


def load_corpus(corpus_path: str, sample_limit: int = None) -> List[Dict[str, Any]]:
    """Load corpus from JSONL file."""
    corpus = []
    logger.info(f"Loading corpus from {corpus_path}...")
    
    with open(corpus_path) as f:
        for i, line in enumerate(f):
            if sample_limit and i >= sample_limit:
                break
            corpus.append(json.loads(line))
    
    logger.info(f"  Loaded {len(corpus):,} documents")
    return corpus


def extract_from_corpus(
    corpus: List[Dict[str, Any]],
    schema_path: str,
    config_path: str,
    model_name: str,
    output_dir: Path,
    batch_size: int = 100,
    resume_from: int = 0
):
    """
    Extract dimensional entities from corpus documents.

    Args:
        corpus: List of documents with 'contents' field
        schema_path: Path to discovered schema config (without .yml extension)
        config_path: Path to main LLM config (e.g., 'hf' for cfg/extraction/inference/hf.yml)
        model_name: Model name override (optional, None = use config default)
        output_dir: Output directory for results
        batch_size: Save checkpoint every N documents
        resume_from: Resume from document index (for fault tolerance)
    """
    # Initialize extractor with discovered schema
    # DimensionalExtractor will create LLM manager internally
    extractor = DimensionalExtractor(
        config_path=config_path,
        dimension_config_path=schema_path,
        model_name=model_name,
        auto_start=False  # Don't auto-start, server already running
    )
    logger.info(f"✓ Initialized extractor with config: {config_path}")
    logger.info(f"✓ Loaded schema from: {schema_path}")
    
    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "extraction_results.jsonl"
    stats_file = output_dir / "extraction_stats.json"
    
    # Resume or start fresh
    if resume_from > 0:
        logger.info(f"Resuming from document {resume_from}")
        corpus = corpus[resume_from:]
        mode = 'a'  # Append mode
    else:
        mode = 'w'  # Write mode
    
    # Process corpus
    logger.info(f"Processing {len(corpus):,} documents...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output: {results_file}")
    logger.info("")
    
    stats = {
        'total_docs': len(corpus),
        'processed': 0,
        'success': 0,
        'failed': 0,
        'total_entities_extracted': 0,
        'by_dimension': {}
    }
    
    with open(results_file, mode) as f:
        for idx, doc in enumerate(tqdm(corpus, desc="Extracting"), start=resume_from):
            try:
                # Extract from document
                text = doc.get('contents', '')
                if not text or len(text) < 10:
                    stats['failed'] += 1
                    continue
                
                metadata = {
                    'doc_id': doc.get('doc_id'),
                    'title': doc.get('title'),
                    'source_corpus': doc.get('metadata', {}).get('source_corpus'),
                    'original_id': doc.get('metadata', {}).get('original_id')
                }
                
                # Extract
                result = extractor.extract(text, document_metadata=metadata)
                
                # Build result dict
                result_dict = {
                    'doc_id': doc.get('doc_id'),
                    'title': doc.get('title'),
                    'metadata': metadata,
                    'extraction': {
                        'success': result.success,
                        'error': result.error,
                        'dimensions': {}
                    }
                }
                
                # Collect all dimensional entities
                entity_count = 0
                if hasattr(result, 'entities') and result.entities:
                    for dim_name, entities in result.entities.items():
                        entity_list = []
                        for e in entities:
                            if hasattr(e, 'dict'):
                                entity_list.append(e.dict())
                            elif isinstance(e, dict):
                                entity_list.append(e)
                            else:
                                entity_list.append(str(e))
                        
                        result_dict['extraction']['dimensions'][dim_name] = entity_list
                        entity_count += len(entity_list)
                        
                        # Update dimension stats
                        if dim_name not in stats['by_dimension']:
                            stats['by_dimension'][dim_name] = 0
                        stats['by_dimension'][dim_name] += len(entity_list)
                
                # Write result
                f.write(json.dumps(result_dict) + '\n')
                
                # Update stats
                stats['processed'] += 1
                if result.success:
                    stats['success'] += 1
                    stats['total_entities_extracted'] += entity_count
                else:
                    stats['failed'] += 1
                
                # Periodic checkpoint
                if (idx + 1) % batch_size == 0:
                    f.flush()  # Force write to disk
                    logger.info(f"  Checkpoint: {idx + 1}/{len(corpus)} docs, " +
                              f"{stats['success']} success, {stats['failed']} failures, " +
                              f"{stats['total_entities_extracted']} entities")
                    
                    # Save stats
                    with open(stats_file, 'w') as sf:
                        json.dump(stats, sf, indent=2)
                    
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"  Failed to process doc {doc.get('doc_id')}: {e}")
                continue
    
    # Final stats
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total documents: {stats['total_docs']:,}")
    logger.info(f"  Successfully processed: {stats['success']:,}")
    logger.info(f"  Failed: {stats['failed']:,}")
    logger.info(f"  Total entities extracted: {stats['total_entities_extracted']:,}")
    logger.info("")
    logger.info("Entities by dimension:")
    for dim, count in sorted(stats['by_dimension'].items(), key=lambda x: -x[1]):
        logger.info(f"  {dim:40s}: {count:6,} entities")
    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Stats saved to: {stats_file}")
    logger.info("")
    
    # Save final stats
    with open(stats_file, 'w') as sf:
        json.dump(stats, sf, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract dimensions from MedCorp corpus")
    parser.add_argument("--corpus", default="data/original/medcorp/train.jsonl",
                       help="Path to corpus JSONL file")
    parser.add_argument("--schema", default="data/schema_discovery_mirage_v2/final_schema",
                       help="Path to discovered schema (without .yml extension)")
    parser.add_argument("--config", default="hf",
                       help="LLM config name (e.g., 'hf' for cfg/extraction/inference/hf.yml)")
    parser.add_argument("--model", default=None,
                       help="Model name override (optional, default: use config)")
    parser.add_argument("--output-dir", default="data/extraction_results",
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Save checkpoint every N documents")
    parser.add_argument("--sample-limit", type=int,
                       help="Limit to first N documents (for testing)")
    parser.add_argument("--resume-from", type=int, default=0,
                       help="Resume from document index")
    args = parser.parse_args()

    # Load corpus
    corpus = load_corpus(args.corpus, sample_limit=args.sample_limit)

    # Run extraction
    output_dir = Path(args.output_dir)
    extract_from_corpus(
        corpus=corpus,
        schema_path=args.schema,
        config_path=args.config,
        model_name=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
