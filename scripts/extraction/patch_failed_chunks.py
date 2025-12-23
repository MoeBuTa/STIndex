#!/usr/bin/env python3
"""
Patch script to re-extract failed chunks and append to existing output.

Usage:
    python -m scripts.extraction.patch_failed_chunks \
        --config cfg/extraction/corpus_extraction_parallel.yml \
        --failed-chunks data/extraction_results_parallel/failed_chunks.jsonl \
        --output data/extraction_results_parallel/corpus_extraction_worker1.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
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


def load_failed_doc_ids(failed_chunks_path: str) -> set:
    """Load doc_ids from failed_chunks.jsonl."""
    doc_ids = set()
    with open(failed_chunks_path) as f:
        for line in f:
            data = json.loads(line)
            doc_ids.add(data['doc_id'])
    return doc_ids


def load_corpus_by_ids(corpus_path: str, doc_ids: set) -> list:
    """Load only documents matching the given doc_ids."""
    docs = []
    with open(corpus_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get('doc_id') in doc_ids:
                docs.append(data)
    return docs


def main():
    parser = argparse.ArgumentParser(description="Re-extract failed chunks and append to output")
    parser.add_argument("--config", required=True, help="Path to extraction config YAML")
    parser.add_argument("--failed-chunks", required=True, help="Path to failed_chunks.jsonl")
    parser.add_argument("--output", required=True, help="Path to output file (will append)")
    parser.add_argument("--base-url", default="http://localhost:8001", help="LLM server URL")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config_from_file(args.config)

    corpus_path = config['input']['corpus_path']
    extraction_config = config.get('extraction', {})
    dimension_config_path = extraction_config['dimension_config']
    dimension_overrides = extraction_config.get('dimension_overrides')
    llm_config = config.get('llm', {})

    # Override base_url
    llm_config['base_url'] = args.base_url

    # Load failed doc_ids
    print(f"\nLoading failed chunks from: {args.failed_chunks}")
    failed_doc_ids = load_failed_doc_ids(args.failed_chunks)
    print(f"  Found {len(failed_doc_ids):,} failed documents to re-extract")

    if not failed_doc_ids:
        print("No failed documents to process. Exiting.")
        return 0

    # Load documents
    print(f"\nLoading documents from corpus: {corpus_path}")
    docs = load_corpus_by_ids(corpus_path, failed_doc_ids)
    print(f"  Loaded {len(docs):,} documents")

    # Initialize extractor
    print(f"\nInitializing extractor...")
    loader = DimensionConfigLoader()
    dimension_config = loader.load_dimension_config(dimension_config_path)

    extractor = DimensionalExtractor(
        config_path="extract",
        dimension_config_path=dimension_config_path,
        dimension_overrides=dimension_overrides,
        prompt_mode="corpus"
    )

    if llm_config:
        extractor.llm_manager = LLMManager(llm_config)

    labeler = DimensionalChunkLabeler(
        dimension_config=dimension_config,
        enabled_dimensions=set(extractor.dimensions.keys())
    )

    corpus_name = config.get('indexing', {}).get('corpus_name', 'unknown')

    # Process and append
    print(f"\nRe-extracting {len(docs):,} documents...")
    successful = 0
    still_failed = 0
    new_failed_ids = []

    with open(args.output, 'a') as outfile:
        for doc in tqdm(docs, desc="Patching"):
            try:
                text = doc.get('contents', '')
                doc_id = doc.get('doc_id', '')
                title = doc.get('title', '')

                result = extractor.extract(text)
                labels = labeler.label_chunk(
                    chunk_text=text,
                    extraction_result=result,
                    chunk_index=0
                )

                schema_data = labels.to_dict()
                has_labels = any(v for v in schema_data.values() if v)

                if has_labels:
                    result_dict = {
                        'doc_id': doc_id,
                        'doc_metadata': {
                            'title': title,
                            'source': doc.get('source', 'unknown'),
                            'corpus': corpus_name,
                            'original_id': doc.get('id', doc_id)
                        },
                        'text': text,
                        'schema_metadata': schema_data
                    }
                    outfile.write(json.dumps(result_dict) + '\n')
                    outfile.flush()
                    successful += 1
                else:
                    still_failed += 1
                    new_failed_ids.append(doc_id)

            except Exception as e:
                logger.error(f"Failed: {doc.get('doc_id', 'unknown')}: {e}")
                still_failed += 1
                new_failed_ids.append(doc.get('doc_id', ''))

    # Update failed_chunks.jsonl with remaining failures
    if new_failed_ids:
        failed_path = Path(args.failed_chunks)
        with open(failed_path, 'w') as f:
            for doc_id in new_failed_ids:
                f.write(json.dumps({"doc_id": doc_id, "reason": "still_failed_after_patch"}) + '\n')

    print(f"\n{'=' * 60}")
    print(f"✅ Patch completed!")
    print(f"{'=' * 60}")
    print(f"  Re-extracted: {len(docs):,}")
    print(f"  Success:      {successful:,}")
    print(f"  Still failed: {still_failed:,}")
    print(f"  Appended to:  {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
