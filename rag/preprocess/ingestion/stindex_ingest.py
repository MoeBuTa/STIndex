#!/usr/bin/env python3
"""
STIndex Corpus Ingestion Module.

This module processes corpus chunks through STIndex for:
1. Temporal entity extraction (dates, durations, periods)
2. Spatial entity extraction (locations, coordinates via geocoding)
3. Building hierarchical indexes for efficient filtering
4. Saving to warehouse (parquet, jsonl, geojsonl)

Usage:
    python -m rag.preprocess.ingestion.stindex_ingest \
        --input data/corpus/merged/chunks.jsonl \
        --output data/warehouse/rag \
        --batch-size 100 \
        --config extract
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines
from loguru import logger
from tqdm import tqdm

# Import STIndex components
try:
    from stindex import DimensionalExtractor
    from stindex.extraction.context_manager import ExtractionContext
    from stindex.warehouse import FileBasedWarehouse
    from stindex.warehouse.chunk_labeler import DimensionalChunkLabeler
except ImportError as e:
    logger.error(f"Failed to import STIndex: {e}")
    logger.error("Please install STIndex: pip install -e .")
    raise


class STIndexCorpusIngester:
    """
    Ingest corpus chunks into STIndex warehouse with dimensional extraction.

    Supports:
    - Context-aware extraction for document-level consistency
    - Batch processing with checkpointing
    - Multiple output formats (jsonl, parquet, geojson)
    """

    def __init__(
        self,
        config_path: str = "extract",
        output_dir: str = "data/warehouse/rag",
        enable_context: bool = True,
        enable_geocoding: bool = False,  # Disable by default for speed
        batch_size: int = 100,
    ):
        """
        Initialize the ingester.

        Args:
            config_path: STIndex config path or name
            output_dir: Output directory for warehouse
            enable_context: Enable context-aware extraction
            enable_geocoding: Enable geocoding for spatial entities
            batch_size: Batch size for processing
        """
        self.output_dir = Path(output_dir)
        self.enable_context = enable_context
        self.enable_geocoding = enable_geocoding
        self.batch_size = batch_size

        # Initialize extractor
        logger.info(f"Initializing STIndex extractor with config: {config_path}")
        self.extractor = DimensionalExtractor(config_path=config_path)

        # Initialize warehouse
        self.warehouse = FileBasedWarehouse(str(self.output_dir))

        # Initialize labeler
        self.labeler = DimensionalChunkLabeler()

        # Statistics
        self.stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "temporal_entities": 0,
            "spatial_entities": 0,
            "failed_chunks": 0,
            "start_time": None,
            "end_time": None,
        }

    def load_chunks(self, input_path: str, limit: Optional[int] = None) -> List[Dict]:
        """Load chunks from JSONL file."""
        chunks = []
        with jsonlines.open(input_path, "r") as reader:
            for i, chunk in enumerate(reader):
                if limit and i >= limit:
                    break
                chunks.append(chunk)
        return chunks

    def extract_chunk(
        self,
        chunk: Dict[str, Any],
        context: Optional[ExtractionContext] = None,
    ) -> Dict[str, Any]:
        """
        Extract dimensional entities from a single chunk.

        Args:
            chunk: Chunk dict with 'text', 'doc_id', etc.
            context: Optional extraction context for consistency

        Returns:
            Enriched chunk dict with extraction results
        """
        text = chunk.get("text", "")
        if not text:
            return chunk

        try:
            # Run extraction
            result = self.extractor.extract(text)

            # Extract entities
            temporal_entities = result.temporal_entities or []
            spatial_entities = result.spatial_entities or []

            # Update stats
            self.stats["temporal_entities"] += len(temporal_entities)
            self.stats["spatial_entities"] += len(spatial_entities)

            # Enrich chunk
            enriched = {
                **chunk,
                "temporal_entities": temporal_entities,
                "spatial_entities": spatial_entities,
                "extraction_timestamp": datetime.now().isoformat(),
            }

            # Add derived fields for indexing
            if temporal_entities:
                primary_temporal = temporal_entities[0]
                enriched["temporal_normalized"] = primary_temporal.get("normalized")
                enriched["temporal_type"] = primary_temporal.get("normalization_type")
                enriched["temporal_text"] = primary_temporal.get("text")

                # Parse year/month/quarter
                normalized = primary_temporal.get("normalized", "")
                if normalized and len(normalized) >= 4:
                    try:
                        enriched["temporal_year"] = int(normalized[:4])
                        if len(normalized) >= 7:
                            month = int(normalized[5:7])
                            enriched["temporal_month"] = month
                            enriched["temporal_quarter"] = (month - 1) // 3 + 1
                    except (ValueError, IndexError):
                        pass

            if spatial_entities:
                primary_spatial = spatial_entities[0]
                enriched["spatial_text"] = primary_spatial.get("text")
                enriched["latitude"] = primary_spatial.get("latitude")
                enriched["longitude"] = primary_spatial.get("longitude")
                enriched["location_type"] = primary_spatial.get("location_type")
                enriched["parent_region"] = primary_spatial.get("parent_region")

            return enriched

        except Exception as e:
            logger.warning(f"Extraction failed for chunk {chunk.get('chunk_id')}: {e}")
            self.stats["failed_chunks"] += 1
            return {**chunk, "extraction_error": str(e)}

    def process_batch(
        self,
        chunks: List[Dict[str, Any]],
        context: Optional[ExtractionContext] = None,
    ) -> List[Dict[str, Any]]:
        """Process a batch of chunks."""
        results = []
        for chunk in chunks:
            enriched = self.extract_chunk(chunk, context)
            results.append(enriched)
            self.stats["processed_chunks"] += 1
        return results

    def process_batch_parallel(
        self,
        chunks: List[Dict[str, Any]],
        max_workers: int = 100,
    ) -> List[Dict[str, Any]]:
        """Process a batch of chunks in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.extract_chunk, chunk, None): i
                for i, chunk in enumerate(chunks)
            }

            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    self.stats["processed_chunks"] += 1
                except Exception as e:
                    logger.warning(f"Parallel extraction failed for chunk {idx}: {e}")
                    results[idx] = {**chunks[idx], "extraction_error": str(e)}
                    self.stats["failed_chunks"] += 1

        return results

    def ingest(
        self,
        input_path: str,
        limit: Optional[int] = None,
        checkpoint_interval: int = 1000,
        resume: bool = True,
        parallel: bool = False,
        max_workers: int = 100,
    ) -> Dict[str, Any]:
        """
        Ingest corpus into warehouse.

        Args:
            input_path: Path to input chunks JSONL
            limit: Limit number of chunks to process
            checkpoint_interval: Save checkpoint every N chunks
            resume: Resume from checkpoint if available
            parallel: Enable parallel processing
            max_workers: Number of parallel workers (threads)

        Returns:
            Ingestion statistics
        """
        self.stats["start_time"] = datetime.now().isoformat()

        # Load chunks
        logger.info(f"Loading chunks from {input_path}")
        chunks = self.load_chunks(input_path, limit)
        self.stats["total_chunks"] = len(chunks)
        logger.info(f"Loaded {len(chunks)} chunks")

        if parallel:
            logger.info(f"Parallel processing enabled with {max_workers} workers")

        # Check for checkpoint
        checkpoint_path = self.output_dir / "checkpoint.json"
        start_idx = 0

        if resume and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)
            start_idx = checkpoint.get("processed", 0)
            logger.info(f"Resuming from checkpoint: {start_idx} chunks already processed")

        # Output file
        output_path = self.output_dir / "chunks_enriched.jsonl"
        mode = "a" if start_idx > 0 else "w"

        # Process in batches
        with jsonlines.open(output_path, mode=mode) as writer:
            for i in tqdm(range(start_idx, len(chunks), self.batch_size), desc="Ingesting"):
                batch = chunks[i:i + self.batch_size]

                # Create context for batch (group by doc_id)
                context = None
                if self.enable_context and not parallel:
                    # Simple context - just track doc_id (only for sequential)
                    pass

                # Process batch (parallel or sequential)
                if parallel:
                    enriched_batch = self.process_batch_parallel(batch, max_workers)
                else:
                    enriched_batch = self.process_batch(batch, context)

                # Write results
                for chunk in enriched_batch:
                    writer.write(chunk)

                # Save checkpoint
                if (i + self.batch_size) % checkpoint_interval == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"processed": i + self.batch_size}, f)

        # Build indexes
        logger.info("Building indexes...")
        self._build_indexes(output_path)

        # Export to parquet
        logger.info("Exporting to Parquet...")
        self._export_parquet(output_path)

        # Export GeoJSON for spatial visualization
        logger.info("Exporting GeoJSON...")
        self._export_geojson(output_path)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        self.stats["end_time"] = datetime.now().isoformat()

        # Save stats
        stats_path = self.output_dir / "ingestion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.success(f"Ingestion complete! Stats: {self.stats}")
        return self.stats

    def _build_indexes(self, enriched_path: Path):
        """Build inverted indexes for temporal and spatial filtering."""
        temporal_index = {}  # year/month -> chunk_ids
        spatial_index = {}   # region -> chunk_ids

        with jsonlines.open(enriched_path, "r") as reader:
            for chunk in reader:
                chunk_id = chunk.get("chunk_id", "")

                # Temporal index
                if "temporal_year" in chunk and chunk["temporal_year"]:
                    year = str(chunk["temporal_year"])
                    if year not in temporal_index:
                        temporal_index[year] = []
                    temporal_index[year].append(chunk_id)

                # Spatial index
                if "spatial_text" in chunk and chunk["spatial_text"]:
                    location = chunk["spatial_text"]
                    if location not in spatial_index:
                        spatial_index[location] = []
                    spatial_index[location].append(chunk_id)

        # Save indexes
        indexes_dir = self.output_dir / "indexes"
        indexes_dir.mkdir(exist_ok=True)

        with open(indexes_dir / "temporal_index.json", "w") as f:
            json.dump(temporal_index, f)

        with open(indexes_dir / "spatial_index.json", "w") as f:
            json.dump(spatial_index, f)

        logger.info(f"Built temporal index: {len(temporal_index)} years")
        logger.info(f"Built spatial index: {len(spatial_index)} locations")

    def _export_parquet(self, enriched_path: Path):
        """Export enriched chunks to Parquet format."""
        try:
            import pandas as pd

            chunks = []
            with jsonlines.open(enriched_path, "r") as reader:
                for chunk in reader:
                    # Flatten for parquet
                    flat_chunk = {
                        k: v for k, v in chunk.items()
                        if not isinstance(v, (list, dict))
                    }
                    # Store complex fields as JSON strings
                    if "temporal_entities" in chunk:
                        flat_chunk["temporal_entities_json"] = json.dumps(chunk["temporal_entities"])
                    if "spatial_entities" in chunk:
                        flat_chunk["spatial_entities_json"] = json.dumps(chunk["spatial_entities"])
                    chunks.append(flat_chunk)

            df = pd.DataFrame(chunks)
            parquet_path = self.output_dir / "chunks_enriched.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Exported {len(df)} chunks to {parquet_path}")

        except ImportError:
            logger.warning("pandas not installed, skipping Parquet export")

    def _export_geojson(self, enriched_path: Path):
        """Export spatial entities to GeoJSON."""
        features = []

        with jsonlines.open(enriched_path, "r") as reader:
            for chunk in reader:
                lat = chunk.get("latitude")
                lon = chunk.get("longitude")

                if lat is not None and lon is not None:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                        "properties": {
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "location": chunk.get("spatial_text"),
                            "temporal": chunk.get("temporal_normalized"),
                            "text_preview": chunk.get("text", "")[:200],
                        },
                    }
                    features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        geojson_path = self.output_dir / "spatial_entities.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f)

        logger.info(f"Exported {len(features)} spatial features to {geojson_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest corpus into STIndex warehouse"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/corpus/merged/chunks.jsonl",
        help="Input chunks JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/warehouse/rag",
        help="Output warehouse directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="extract",
        help="STIndex config path or name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process",
    )
    parser.add_argument(
        "--enable-geocoding",
        action="store_true",
        help="Enable geocoding for spatial entities",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Checkpoint interval",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing with ThreadPoolExecutor",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Number of parallel workers (default: 100)",
    )

    args = parser.parse_args()

    # Create ingester
    ingester = STIndexCorpusIngester(
        config_path=args.config,
        output_dir=args.output,
        enable_geocoding=args.enable_geocoding,
        batch_size=args.batch_size,
    )

    # Run ingestion
    stats = ingester.ingest(
        input_path=args.input,
        limit=args.limit,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        parallel=args.parallel,
        max_workers=args.workers,
    )

    print(f"\n=== Ingestion Complete ===")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Processed: {stats['processed_chunks']}")
    print(f"Temporal entities: {stats['temporal_entities']}")
    print(f"Spatial entities: {stats['spatial_entities']}")
    print(f"Failed: {stats['failed_chunks']}")


if __name__ == "__main__":
    main()
