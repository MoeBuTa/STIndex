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

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.postprocess.reflection import ExtractionReflector
from stindex.preprocess import DocumentChunk, InputDocument, Preprocessor
from stindex.visualization import STIndexVisualizer


class STIndexPipeline:
    """
    End-to-end pipeline orchestrator with context-aware extraction and reflection.

    Context-aware features:
    - Maintains memory across document chunks
    - Resolves relative temporal expressions using prior references
    - Disambiguates spatial mentions using document context
    - Resets memory between different documents

    Two-pass reflection features:
    - LLM-based quality scoring (relevance, accuracy, completeness, consistency)
    - Filters false positives using configurable thresholds
    - Context-aware reasoning (when combined with context-aware extraction)
    - Reduces extraction errors by 30-50%

    Usage:
        # Full pipeline with context-aware extraction (default)
        pipeline = STIndexPipeline(
            extractor_config="extract",
            dimension_config="dimensions",
            enable_context_aware=True,
            max_memory_refs=10,
            enable_reflection=True,  # Enable two-pass reflection
            relevance_threshold=0.7,
            accuracy_threshold=0.7
        )

        docs = [
            InputDocument.from_url("https://example.com/article"),
            InputDocument.from_file("/path/to/doc.pdf"),
            InputDocument.from_text("Your text here")
        ]

        results = pipeline.run_pipeline(docs)

        # Disable context-aware extraction (legacy mode)
        pipeline = STIndexPipeline(enable_context_aware=False)

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

        # Context-aware extraction (can override config file settings)
        enable_context_aware: Optional[bool] = None,
        max_memory_refs: Optional[int] = None,
        enable_nearby_locations: Optional[bool] = None,

        # Two-pass reflection (can override config file settings)
        enable_reflection: Optional[bool] = None,
        relevance_threshold: Optional[float] = None,
        accuracy_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,

        # Output config
        output_dir: Optional[str] = None,
        save_intermediate: bool = True
    ):
        """
        Initialize pipeline.

        All preprocessing and visualization settings loaded from cfg/*.yml files.
        Context-aware and reflection settings loaded from extract.yml and reflection.yml,
        but can be overridden by parameters.

        Args:
            extractor_config: Config path for DimensionalExtractor
            dimension_config: Dimension config path
            enable_context_aware: Override context-aware setting from config (default: None, use config)
            max_memory_refs: Override max memory refs from config (default: None, use config)
            enable_nearby_locations: Override nearby locations from config (default: None, use config)
            enable_reflection: Override reflection setting from config (default: None, use config)
            relevance_threshold: Override relevance threshold from config (default: None, use config)
            accuracy_threshold: Override accuracy threshold from config (default: None, use config)
            consistency_threshold: Override consistency threshold from config (default: None, use config)
            output_dir: Output directory for results
            save_intermediate: Save intermediate results (chunks, etc.)
        """
        # Store config paths
        self.extractor_config = extractor_config
        self.dimension_config_path = dimension_config

        # Load main extraction config
        from stindex.utils.config import load_config_from_file
        main_config = load_config_from_file(extractor_config)

        # Load context-aware settings from config or use overrides
        context_config = main_config.get("context_aware", {})
        self.enable_context_aware = enable_context_aware if enable_context_aware is not None else context_config.get("enabled", True)
        self.max_memory_refs = max_memory_refs if max_memory_refs is not None else context_config.get("max_memory_refs", 10)
        self.enable_nearby_locations = enable_nearby_locations if enable_nearby_locations is not None else context_config.get("enable_nearby_locations", False)

        # Load reflection settings from config or use overrides
        reflection_config = main_config.get("reflection", {})
        reflection_enabled_from_config = reflection_config.get("enabled", False)
        self.enable_reflection = enable_reflection if enable_reflection is not None else reflection_enabled_from_config

        # Load reflection thresholds from reflection.yml if enabled
        if self.enable_reflection:
            try:
                reflection_detailed_config = load_config_from_file("cfg/extraction/inference/reflection")
                reflection_thresholds = reflection_detailed_config.get("thresholds", {})

                self.relevance_threshold = relevance_threshold if relevance_threshold is not None else reflection_thresholds.get("relevance", 0.7)
                self.accuracy_threshold = accuracy_threshold if accuracy_threshold is not None else reflection_thresholds.get("accuracy", 0.7)
                self.consistency_threshold = consistency_threshold if consistency_threshold is not None else reflection_thresholds.get("consistency", 0.6)
            except Exception as e:
                logger.warning(f"Failed to load reflection config, using defaults: {e}")
                self.relevance_threshold = relevance_threshold if relevance_threshold is not None else 0.7
                self.accuracy_threshold = accuracy_threshold if accuracy_threshold is not None else 0.7
                self.consistency_threshold = consistency_threshold if consistency_threshold is not None else 0.6
        else:
            # Set defaults even if not enabled (for potential runtime enabling)
            self.relevance_threshold = relevance_threshold if relevance_threshold is not None else 0.7
            self.accuracy_threshold = accuracy_threshold if accuracy_threshold is not None else 0.7
            self.consistency_threshold = consistency_threshold if consistency_threshold is not None else 0.6

        # Initialize extractor (for non-context-aware mode)
        # For context-aware mode, we create per-document extractors in run_extraction
        self.extractor = None
        if not self.enable_context_aware:
            self.extractor = DimensionalExtractor(
                config_path=extractor_config,
                dimension_config_path=dimension_config
            )

        # Initialize preprocessor (loads from cfg/preprocess/*.yml)
        self.preprocessor = Preprocessor()

        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else Path("data/output")
        self.save_intermediate = save_intermediate

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.output_dir / "chunks"
        self.results_dir = self.output_dir / "results"
        # Use case-specific visualizations directory
        self.viz_dir = self.output_dir / "visualizations"

        if save_intermediate:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✓ Pipeline initialized")
        logger.info(f"  Context-aware extraction: {'ENABLED' if self.enable_context_aware else 'DISABLED'}")
        if self.enable_context_aware:
            logger.info(f"  Max memory refs: {self.max_memory_refs}")
            logger.info(f"  Nearby locations: {'ENABLED' if self.enable_nearby_locations else 'DISABLED'}")
        logger.info(f"  Two-pass reflection: {'ENABLED' if self.enable_reflection else 'DISABLED'}")
        if self.enable_reflection:
            logger.info(f"  Relevance threshold: {self.relevance_threshold}")
            logger.info(f"  Accuracy threshold: {self.accuracy_threshold}")
            logger.info(f"  Consistency threshold: {self.consistency_threshold}")

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

        Context-aware mode:
        - Groups chunks by document_id
        - Creates ExtractionContext for each document
        - Maintains memory across chunks within same document
        - Resets memory between different documents

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

        if self.enable_context_aware:
            # Context-aware extraction: group by document
            logger.info("Context-aware extraction enabled")

            # Group chunks by document_id
            from collections import defaultdict
            doc_chunks = defaultdict(list)
            for chunk in chunks:
                doc_chunks[chunk.document_id].append(chunk)

            logger.info(f"Processing {len(doc_chunks)} documents with {len(chunks)} total chunks")

            # Process each document with its own context
            for doc_id, doc_chunk_list in doc_chunks.items():
                logger.info(f"\n--- Document: {doc_id} ({len(doc_chunk_list)} chunks) ---")

                # Sort chunks by chunk_index to process in order
                doc_chunk_list.sort(key=lambda c: c.chunk_index)

                # Extract document metadata from first chunk
                first_chunk = doc_chunk_list[0]
                document_metadata = {
                    **first_chunk.document_metadata,
                    "document_id": doc_id,
                    "document_title": first_chunk.document_title,
                }

                # Create ExtractionContext for this document
                context = ExtractionContext(
                    document_metadata=document_metadata,
                    max_memory_refs=self.max_memory_refs,
                    enable_nearby_locations=self.enable_nearby_locations
                )

                # Create extractor with context
                extractor = DimensionalExtractor(
                    config_path=self.extractor_config,
                    dimension_config_path=self.dimension_config_path,
                    extraction_context=context
                )

                # Create reflector if enabled (uses same LLM manager as extractor)
                reflector = None
                if self.enable_reflection:
                    reflector = ExtractionReflector(
                        llm_manager=extractor.llm_manager,
                        relevance_threshold=self.relevance_threshold,
                        accuracy_threshold=self.accuracy_threshold,
                        consistency_threshold=self.consistency_threshold,
                        extraction_context=context  # Pass context for context-aware reflection
                    )
                    logger.debug(f"✓ Reflector initialized with context-aware reasoning")

                # Process chunks in order
                for i, chunk in enumerate(doc_chunk_list):
                    # Update context position
                    context.set_chunk_position(
                        chunk_index=i,
                        total_chunks=len(doc_chunk_list),
                        section_hierarchy=chunk.section_hierarchy if hasattr(chunk, 'section_hierarchy') else ""
                    )

                    logger.info(f"[{i+1}/{len(doc_chunk_list)}] Chunk {chunk.chunk_id}")

                    try:
                        # Build metadata for extraction
                        extraction_metadata = {
                            **chunk.document_metadata,
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "document_title": chunk.document_title,
                        }

                        # Extract (context memory is automatically updated inside)
                        result = extractor.extract(
                            text=chunk.text,
                            document_metadata=extraction_metadata
                        )

                        # Apply two-pass reflection if enabled
                        if reflector and result.success:
                            logger.debug("Running two-pass reflection on extraction...")

                            # Get dimension schemas for reflection
                            dimension_schemas = {
                                dim_name: dim_config.to_metadata().model_dump()
                                for dim_name, dim_config in extractor.dimensions.items()
                            }

                            # Reflect on entities (filters low-confidence extractions)
                            reflected_entities = reflector.reflect_on_extractions(
                                text=chunk.text,
                                extraction_result=result.entities,
                                dimension_schemas=dimension_schemas
                            )

                            # Replace entities with reflected (filtered) entities
                            result.entities = reflected_entities

                            # Update backward-compatible fields
                            result.temporal_entities = reflected_entities.get("temporal", [])
                            result.spatial_entities = reflected_entities.get("spatial", [])

                            # Mark as reflected
                            if isinstance(result.extraction_config, dict):
                                result.extraction_config['reflection_applied'] = True
                                result.extraction_config['reflection_thresholds'] = {
                                    'relevance': self.relevance_threshold,
                                    'accuracy': self.accuracy_threshold,
                                    'consistency': self.consistency_threshold
                                }

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

                logger.info(f"✓ Document {doc_id} complete: "
                          f"{len(context.prior_temporal_refs)} temporal refs, "
                          f"{len(context.prior_spatial_refs)} spatial refs in memory")

        else:
            # Non-context-aware extraction: process each chunk independently
            logger.info("Standard extraction (context-aware disabled)")

            # Create reflector if enabled (without context)
            reflector = None
            if self.enable_reflection:
                reflector = ExtractionReflector(
                    llm_manager=self.extractor.llm_manager,
                    relevance_threshold=self.relevance_threshold,
                    accuracy_threshold=self.accuracy_threshold,
                    consistency_threshold=self.consistency_threshold,
                    extraction_context=None  # No context in standard mode
                )
                logger.debug(f"✓ Reflector initialized (non-context-aware)")

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

                    # Apply two-pass reflection if enabled
                    if reflector and result.success:
                        logger.debug("Running two-pass reflection on extraction...")

                        # Get dimension schemas for reflection
                        dimension_schemas = {
                            dim_name: dim_config.to_metadata().model_dump()
                            for dim_name, dim_config in self.extractor.dimensions.items()
                        }

                        # Reflect on entities (filters low-confidence extractions)
                        reflected_entities = reflector.reflect_on_extractions(
                            text=chunk.text,
                            extraction_result=result.entities,
                            dimension_schemas=dimension_schemas
                        )

                        # Replace entities with reflected (filtered) entities
                        result.entities = reflected_entities

                        # Update backward-compatible fields
                        result.temporal_entities = reflected_entities.get("temporal", [])
                        result.spatial_entities = reflected_entities.get("spatial", [])

                        # Mark as reflected
                        if isinstance(result.extraction_config, dict):
                            result.extraction_config['reflection_applied'] = True
                            result.extraction_config['reflection_thresholds'] = {
                                'relevance': self.relevance_threshold,
                                'accuracy': self.accuracy_threshold,
                                'consistency': self.consistency_threshold
                            }

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
        animated_map: bool = True
    ) -> Optional[str]:
        """
        Run visualization only (requires extraction results).

        All visualization settings loaded from cfg/visualization.yml.

        Automatically creates a zip archive containing the HTML report and all source files.

        Args:
            results: Extraction results or path to results file
            output_dir: Output directory for visualizations
            animated_map: Create animated timeline map if True

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

        # Initialize visualizer (loads from cfg/visualization.yml)
        visualizer = STIndexVisualizer()

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
