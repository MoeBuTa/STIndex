"""
End-to-end pipeline orchestrator for STIndex.

Supports multiple execution modes:
1. Full pipeline: preprocessing → extraction → visualization
2. Preprocessing only: scraping → parsing → chunking
3. Extraction only: dimensional extraction from chunks
4. Visualization only: generate visualizations from results
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from stindex.core.dimensional_extraction import DimensionalExtractor
from stindex.preprocessing import DocumentChunk, InputDocument, Preprocessor
from stindex.visualization import STIndexVisualizer


class STIndexPipeline:
    """
    End-to-end pipeline orchestrator.

    Usage:
        # Full pipeline
        pipeline = STIndexPipeline(
            extractor_config="extract",
            dimension_config="dimensions"
        )

        docs = [
            InputDocument.from_url("https://example.com/article"),
            InputDocument.from_file("/path/to/doc.pdf"),
            InputDocument.from_text("Your text here")
        ]

        results = pipeline.run_pipeline(docs)

        # Preprocessing only
        chunks = pipeline.run_preprocessing(docs)

        # Extraction only (from preprocessed chunks)
        results = pipeline.run_extraction(chunks)

        # Visualization only
        pipeline.run_visualization(results, output_dir="output/viz")
    """

    def __init__(
        self,
        # Extraction config
        extractor_config: str = "extract",
        dimension_config: Optional[str] = "dimensions",

        # Preprocessing config
        max_chunk_size: int = 2000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "sliding_window",
        parsing_method: str = "unstructured",

        # Output config
        output_dir: Optional[str] = None,
        save_intermediate: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            extractor_config: Config path for DimensionalExtractor
            dimension_config: Dimension config path
            max_chunk_size: Maximum chunk size for preprocessing
            chunk_overlap: Overlap between chunks
            chunking_strategy: Chunking strategy
            parsing_method: Parsing method
            output_dir: Output directory for results
            save_intermediate: Save intermediate results (chunks, etc.)
        """
        # Store config paths
        self.extractor_config = extractor_config
        self.dimension_config_path = dimension_config

        # Initialize extractor
        self.extractor = DimensionalExtractor(
            config_path=extractor_config,
            dimension_config_path=dimension_config
        )

        # Initialize preprocessor
        self.preprocessor = Preprocessor(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            parsing_method=parsing_method
        )

        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else Path("data/output")
        self.save_intermediate = save_intermediate

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.output_dir / "chunks"
        self.results_dir = self.output_dir / "results"
        # Use data/visualizations for viz output (separate from other data)
        self.viz_dir = Path("data/visualizations")

        if save_intermediate:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        input_docs: List[InputDocument],
        save_results: bool = True,
        visualize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run full pipeline: preprocessing → extraction → visualization.

        Visualization is enabled by default and creates a zip archive with HTML report and all assets.

        Args:
            input_docs: List of InputDocument objects
            save_results: Save extraction results to file
            visualize: Generate visualizations (default: True, creates zip archive)

        Returns:
            List of extraction results (one per chunk)
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Full Mode")
        logger.info("=" * 80)

        # Step 1: Preprocessing
        logger.info("\n[1/3] Preprocessing...")
        all_chunks = self.run_preprocessing(input_docs, save_chunks=save_results)

        # Flatten chunks
        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

        # Step 2: Extraction
        logger.info(f"\n[2/3] Extraction ({len(flat_chunks)} chunks)...")
        results = self.run_extraction(flat_chunks, save_results=save_results)

        # Step 3: Visualization (optional)
        if visualize:
            logger.info("\n[3/3] Visualization...")
            self.run_visualization(results)

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete")
        logger.info("=" * 80)

        return results

    def run_preprocessing(
        self,
        input_docs: List[InputDocument],
        save_chunks: bool = True
    ) -> List[List[DocumentChunk]]:
        """
        Run preprocessing only: scraping → parsing → chunking.

        Args:
            input_docs: List of InputDocument objects
            save_chunks: Save chunks to file

        Returns:
            List of lists of DocumentChunk objects (one list per document)
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Preprocessing Mode")
        logger.info("=" * 80)

        all_chunks = self.preprocessor.process_batch(input_docs)

        # Save chunks if requested
        if save_chunks and self.save_intermediate:
            self._save_chunks(all_chunks)

        logger.info(f"\n✓ Preprocessing complete: {len(all_chunks)} documents, "
                   f"{sum(len(chunks) for chunks in all_chunks)} total chunks")

        return all_chunks

    def run_extraction(
        self,
        chunks: List[DocumentChunk],
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run extraction only (requires preprocessed chunks).

        Args:
            chunks: List of DocumentChunk objects
            save_results: Save results to file

        Returns:
            List of extraction results (one per chunk)
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Extraction Mode")
        logger.info("=" * 80)

        results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"\n[{i+1}/{len(chunks)}] Extracting from chunk: {chunk.chunk_id}")

            try:
                # Build metadata for extraction
                extraction_metadata = {
                    **chunk.document_metadata,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "document_title": chunk.document_title,
                }

                # Extract
                result = self.extractor.extract(
                    text=chunk.text,
                    document_metadata=extraction_metadata
                )

                # Store result
                result_data = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "extraction": result.model_dump(),
                }

                results.append(result_data)

            except Exception as e:
                logger.error(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
                result_data = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                    "error": str(e)
                }
                results.append(result_data)

        # Save results if requested
        if save_results:
            self._save_results(results)

        # Summary
        success_count = sum(1 for r in results if r.get('extraction', {}).get('success'))
        logger.info(f"\n✓ Extraction complete: {success_count}/{len(results)} successful")

        return results

    def run_visualization(
        self,
        results: Union[List[Dict[str, Any]], str],
        output_dir: Optional[str] = None,
        animated_map: bool = True,
        temporal_dim: str = "temporal",
        spatial_dim: str = "spatial",
        category_dim: Optional[str] = None
    ) -> Optional[str]:
        """
        Run visualization only (requires extraction results).

        Automatically creates a zip archive containing the HTML report and all source files.

        Args:
            results: Extraction results or path to results file
            output_dir: Output directory for visualizations
            animated_map: Create animated timeline map if True
            temporal_dim: Name of temporal dimension for timeline
            spatial_dim: Name of spatial dimension for mapping
            category_dim: Name of categorical dimension for color coding

        Returns:
            Path to generated zip file
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Visualization Mode")
        logger.info("=" * 80)

        # Set output directory
        viz_dir = Path(output_dir) if output_dir else self.viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nGenerating visualizations in: {viz_dir}")

        # Initialize visualizer
        visualizer = STIndexVisualizer(
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            category_dim=category_dim
        )

        # Generate visualization
        try:
            if isinstance(results, str):
                # Results file path provided
                zip_path = visualizer.visualize(
                    results_file=results,
                    output_dir=str(viz_dir),
                    animated_map=animated_map
                )
            else:
                # Results list provided
                zip_path = visualizer.visualize(
                    results=results,
                    output_dir=str(viz_dir),
                    animated_map=animated_map
                )

            logger.info(f"\n✓ Visualization complete: {zip_path}")
            return zip_path

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Fallback to basic summary
            logger.info("Generating basic summary instead...")
            if isinstance(results, str):
                with open(results, 'r') as f:
                    results = json.load(f)
            self._generate_summary(results, viz_dir)
            return None

    def load_chunks_from_file(self, chunks_file: str) -> List[DocumentChunk]:
        """
        Load preprocessed chunks from file.

        Args:
            chunks_file: Path to chunks JSON file

        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Loading chunks from: {chunks_file}")

        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)

        chunks = [DocumentChunk.from_dict(chunk_dict) for chunk_dict in chunks_data]

        logger.info(f"✓ Loaded {len(chunks)} chunks")
        return chunks

    def _save_chunks(self, all_chunks: List[List[DocumentChunk]]):
        """Save chunks to file."""
        # Flatten chunks
        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

        # Convert to dicts
        chunks_data = [chunk.to_dict() for chunk in flat_chunks]

        # Save
        output_file = self.chunks_dir / "preprocessed_chunks.json"
        with open(output_file, 'w') as f:
            json.dump(chunks_data, f, indent=2)

        logger.info(f"💾 Saved {len(flat_chunks)} chunks to: {output_file}")

    def _save_results(self, results: List[Dict[str, Any]]):
        """Save extraction results to file."""
        output_file = self.results_dir / "extraction_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Saved {len(results)} results to: {output_file}")

    def _generate_summary(self, results: List[Dict[str, Any]], output_dir: Path):
        """Generate basic summary visualization."""
        # Count dimensions
        dimension_counts = {}
        total_success = 0

        for result in results:
            if result.get('extraction', {}).get('success'):
                total_success += 1
                entities = result['extraction'].get('entities', {})
                for dim_name, dim_entities in entities.items():
                    if dim_entities:
                        dimension_counts[dim_name] = dimension_counts.get(dim_name, 0) + len(dim_entities)

        # Create summary
        summary = {
            "total_chunks": len(results),
            "successful_extractions": total_success,
            "failed_extractions": len(results) - total_success,
            "dimensions_extracted": dimension_counts
        }

        # Save summary
        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Summary:")
        logger.info(f"  Total chunks: {summary['total_chunks']}")
        logger.info(f"  Successful: {summary['successful_extractions']}")
        logger.info(f"  Failed: {summary['failed_extractions']}")
        logger.info(f"  Dimensions: {list(dimension_counts.keys())}")
        logger.info(f"  Saved to: {summary_file}")
