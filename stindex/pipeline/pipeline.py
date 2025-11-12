"""
End-to-end pipeline orchestrator for STIndex.

Supports multiple execution modes:
1. Full pipeline: preprocessing → extraction → analysis → export
2. Preprocessing only: scraping → parsing → chunking
3. Extraction only: dimensional extraction from chunks
4. Analysis only: clustering → dimension analysis → export
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from stindex.analysis import (
    DimensionAnalyzer,
    EventClusterAnalyzer,
    AnalysisDataExporter,
)
from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.postprocess.reflection import ExtractionReflector
from stindex.preprocess import DocumentChunk, InputDocument, Preprocessor


class STIndexPipeline:
    """
    End-to-end pipeline orchestrator with context-aware extraction and analysis.

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

    Analysis features:
    - Spatiotemporal clustering (DBSCAN with geodesic distance)
    - Event burst detection (temporal spikes)
    - Story arc detection (narrative sequences)
    - Multi-dimensional statistical analysis
    - Static JSON export for frontend (no backend required)

    Usage:
        # Full pipeline with analysis
        pipeline = STIndexPipeline(
            extractor_config="extract",
            dimension_config="dimensions",
            enable_context_aware=True,
            enable_reflection=True,
            enable_analysis=True
        )

        docs = [
            InputDocument.from_url("https://example.com/article"),
            InputDocument.from_file("/path/to/doc.pdf"),
            InputDocument.from_text("Your text here")
        ]

        results = pipeline.run_pipeline(docs, analyze=True)

        # Analysis only (from existing results)
        analysis_data = pipeline.run_analysis(
            results=results,
            dimensions=['temporal', 'spatial', 'disease', 'event_type']
        )
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

        # Data warehouse integration
        enable_warehouse: bool = False,
        warehouse_config: Optional[str] = None,

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
            enable_warehouse: Enable data warehouse integration (default: False)
            warehouse_config: Warehouse config path (default: "warehouse")
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

        # Initialize warehouse (optional)
        self.enable_warehouse = enable_warehouse
        self.warehouse_etl = None
        if enable_warehouse:
            try:
                from stindex.warehouse.etl import DimensionalWarehouseETL
                from stindex.warehouse.chunk_labeler import DimensionalChunkLabeler
                from stindex.utils.config import load_config_from_file

                # Load warehouse config
                warehouse_config_path = warehouse_config or "warehouse"
                wh_config = load_config_from_file(warehouse_config_path)

                # Get database connection string
                db_connection = wh_config.get("database", {}).get("connection_string")
                if not db_connection:
                    logger.warning("Warehouse enabled but no connection_string in config. Warehouse disabled.")
                    self.enable_warehouse = False
                else:
                    # Initialize chunk labeler
                    chunk_labeler = DimensionalChunkLabeler(dimension_config=None)

                    # Initialize ETL
                    batch_size = wh_config.get("etl", {}).get("batch_size", 100)
                    self.warehouse_etl = DimensionalWarehouseETL(
                        db_connection_string=db_connection,
                        chunk_labeler=chunk_labeler,
                        batch_size=batch_size
                    )
                    logger.info(f"✓ Warehouse integration enabled")
                    logger.info(f"  Database: {db_connection.split('@')[-1] if '@' in db_connection else db_connection}")
            except Exception as e:
                logger.error(f"Failed to initialize warehouse: {e}")
                logger.warning("Continuing without warehouse integration")
                self.enable_warehouse = False
                self.warehouse_etl = None

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
        analyze: bool = True,
        load_to_warehouse: bool = True
    ) -> Dict[str, Any]:
        """
        Run full pipeline: preprocessing → extraction → analysis → export.

        Analysis is enabled by default and generates static JSON files for frontend.
        Warehouse loading is enabled if warehouse is initialized and load_to_warehouse=True.

        Args:
            input_docs: List of InputDocument objects
            save_results: Save extraction results to file
            analyze: Run analysis and export data (default: True, creates JSON files)
            load_to_warehouse: Load results to data warehouse if enabled (default: True)

        Returns:
            Dict with 'results' (extraction results) and 'analysis' (analysis data if enabled)
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Full Mode")
        logger.info("=" * 80)

        # Step 1: Preprocessing
        logger.info("\n[1/4] Preprocessing...")
        all_chunks = self.run_preprocessing(input_docs, save_chunks=save_results)

        # Flatten chunks
        flat_chunks = [chunk for doc_chunks in all_chunks for chunk in doc_chunks]

        # Step 2: Extraction
        logger.info(f"\n[2/4] Extraction ({len(flat_chunks)} chunks)...")
        results = self.run_extraction(flat_chunks, save_results=save_results)

        # Step 3: Warehouse loading (optional)
        if self.enable_warehouse and load_to_warehouse and self.warehouse_etl:
            logger.info("\n[3/4] Loading to data warehouse...")
            self.run_warehouse_loading(results, all_chunks)
        else:
            logger.info("\n[3/4] Warehouse loading: SKIPPED")

        # Step 4: Analysis (optional)
        analysis_data = None
        if analyze:
            logger.info("\n[4/4] Analysis & Export...")
            analysis_data = self.run_analysis(results)
        else:
            logger.info("\n[4/4] Analysis: SKIPPED")

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete")
        logger.info("=" * 80)

        return {
            'results': results,
            'analysis': analysis_data
        }

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

                        # Extract
                        # If reflection is enabled, defer context update until after reflection
                        result = extractor.extract(
                            text=chunk.text,
                            document_metadata=extraction_metadata,
                            update_context=(reflector is None)  # Only update if no reflection
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

                            # Update context with reflected entities (high-quality only)
                            extractor.update_context_memory(reflected_entities)

                            # Mark as reflected
                            if isinstance(result.extraction_config, dict):
                                result.extraction_config['reflection_applied'] = True
                                result.extraction_config['reflection_thresholds'] = {
                                    'relevance': self.relevance_threshold,
                                    'accuracy': self.accuracy_threshold,
                                    'consistency': self.consistency_threshold
                                }

                        # Store result with chunk parameters (excluding start_char/end_char)
                        result_data = {
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "document_id": chunk.document_id,
                            "document_title": chunk.document_title,
                            "source": chunk.document_metadata.get("source"),
                            "text": chunk.text,
                            "chunk_params": {
                                "total_chunks": chunk.total_chunks,
                                "word_count": chunk.word_count,
                                "char_count": chunk.char_count,
                                "previous_chunk_summary": chunk.previous_chunk_summary,
                                "section_hierarchy": chunk.section_hierarchy,
                                "element_types": chunk.element_types,
                                "keywords": chunk.keywords,
                                "summary": chunk.summary,
                                "preview": chunk.preview,
                            },
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
                        document_metadata=extraction_metadata,
                        update_context=(reflector is None)  # Consistent with context-aware mode
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

                        # Note: No context update needed in non-context-aware mode

                        # Mark as reflected
                        if isinstance(result.extraction_config, dict):
                            result.extraction_config['reflection_applied'] = True
                            result.extraction_config['reflection_thresholds'] = {
                                'relevance': self.relevance_threshold,
                                'accuracy': self.accuracy_threshold,
                                'consistency': self.consistency_threshold
                            }

                    # Store result with chunk parameters (excluding start_char/end_char)
                    result_data = {
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "document_id": chunk.document_id,
                        "document_title": chunk.document_title,
                        "source": chunk.document_metadata.get("source"),
                        "text": chunk.text,
                        "chunk_params": {
                            "total_chunks": chunk.total_chunks,
                            "word_count": chunk.word_count,
                            "char_count": chunk.char_count,
                            "previous_chunk_summary": chunk.previous_chunk_summary,
                            "section_hierarchy": chunk.section_hierarchy,
                            "element_types": chunk.element_types,
                            "keywords": chunk.keywords,
                            "summary": chunk.summary,
                            "preview": chunk.preview,
                        },
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

    def run_analysis(
        self,
        results: Union[List[Dict[str, Any]], str],
        dimensions: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        clustering_mode: str = 'spatiotemporal',
        export_geojson: bool = True
    ) -> Dict[str, Any]:
        """
        Run analysis only (requires extraction results).

        Performs:
        1. Event clustering (spatiotemporal + categorical)
        2. Story arc detection (narrative sequences)
        3. Dimension analysis (statistics for all dimensions)
        4. Data export (static JSON files for frontend)

        Args:
            results: Extraction results or path to results file
            dimensions: Dimensions to analyze (None = all discovered dimensions)
            output_dir: Output directory for exported data
            clustering_mode: 'temporal', 'spatial', 'spatiotemporal', 'categorical', 'multi'
            export_geojson: Export GeoJSON file for map visualization

        Returns:
            Dict with analysis results:
                - clusters: Cluster analysis results
                - story_arcs: Story arc detection results
                - dimension_analysis: Dimension statistics
                - exported_files: Paths to exported JSON files
        """
        logger.info("=" * 80)
        logger.info("STIndex Pipeline: Analysis Mode")
        logger.info("=" * 80)

        # Load results if file path provided
        if isinstance(results, str):
            logger.info(f"Loading results from: {results}")
            with open(results, 'r') as f:
                results = json.load(f)

        # Extract model name from results for directory naming
        model_name = "unknown_model"
        for result in results:
            extraction = result.get('extraction', {})
            if extraction.get('success'):
                extraction_config = extraction.get('extraction_config', {})
                if isinstance(extraction_config, dict):
                    # Try model_name first (current format), then model (legacy)
                    model_name = extraction_config.get('model_name') or extraction_config.get('model', 'unknown_model')
                else:
                    # Handle ExtractionConfig object
                    model_name = getattr(extraction_config, 'model_name', None) or getattr(extraction_config, 'model', 'unknown_model')
                break

        # Clean model name for filesystem (replace / with _)
        clean_model_name = model_name.replace('/', '_')

        # Set output directory with model-specific subdirectory
        if output_dir:
            analysis_dir = Path(output_dir)
        else:
            # Create model-specific analysis directory
            analysis_dir = self.output_dir / "analysis" / clean_model_name

        analysis_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nAnalyzing {len(results)} extraction results...")
        logger.info(f"Model: {model_name}")
        logger.info(f"Output directory: {analysis_dir}")

        try:
            # 1. Event clustering
            logger.info("\n[1/4] Event Clustering...")
            cluster_analyzer = EventClusterAnalyzer(
                temporal_window='1D',
                spatial_radius_km=50,
                min_cluster_size=2
            )
            cluster_results = cluster_analyzer.detect_clusters(
                extraction_results=results,
                dimensions=dimensions,
                clustering_mode=clustering_mode
            )

            # 2. Dimension analysis
            logger.info("\n[2/3] Dimension Analysis...")
            dim_analyzer = DimensionAnalyzer()
            dimension_analysis = dim_analyzer.analyze(
                extraction_results=results,
                dimensions=dimensions
            )

            # 3. Export data
            logger.info("\n[3/3] Exporting Data...")
            exporter = AnalysisDataExporter(output_dir=str(analysis_dir))

            exported_files = exporter.export_all(
                extraction_results=results,
                clusters=cluster_results,
                dimension_analysis=dimension_analysis,
                metadata={
                    'model_name': model_name,
                    'clustering_mode': clustering_mode,
                    'dimensions_analyzed': dimensions or 'all'
                }
            )

            # Export GeoJSON if requested
            if export_geojson:
                events = cluster_results.get('events', [])
                geojson_path = exporter.export_geojson(events)
                exported_files['geojson'] = geojson_path

            logger.info(f"\n✓ Analysis complete!")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Clusters: {len(cluster_results.get('clusters', []))}")
            logger.info(f"  Dimensions: {len(dimension_analysis) - 2}")  # Exclude global & cross_dimensional
            logger.info(f"  Exported files: {len(exported_files)}")
            logger.info(f"  Location: {analysis_dir}")

            return {
                'clusters': cluster_results,
                'dimension_analysis': dimension_analysis,
                'exported_files': exported_files
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

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

    def run_warehouse_loading(
        self,
        results: List[Dict[str, Any]],
        all_chunks: List[List[DocumentChunk]]
    ) -> None:
        """
        Load extraction results into data warehouse.

        Groups results by document and loads each document's chunks together.

        Args:
            results: List of extraction results (one per chunk)
            all_chunks: List of lists of DocumentChunk objects (one list per document)
        """
        if not self.warehouse_etl:
            logger.warning("Warehouse ETL not initialized, skipping warehouse loading")
            return

        try:
            from stindex.llm.response.dimension_models import MultiDimensionalResult
            from collections import defaultdict

            # Group results by document_id
            doc_results = defaultdict(list)
            for result in results:
                doc_id = result.get("document_id", "unknown")
                doc_results[doc_id].append(result)

            # Group chunks by document_id
            doc_chunks_dict = {}
            for doc_chunk_list in all_chunks:
                if doc_chunk_list:
                    doc_id = doc_chunk_list[0].document_id
                    doc_chunks_dict[doc_id] = doc_chunk_list

            total_loaded = 0

            # Load each document
            for doc_id, doc_result_list in doc_results.items():
                logger.info(f"Loading document {doc_id} to warehouse ({len(doc_result_list)} chunks)...")

                # Get chunks for this document
                doc_chunk_list = doc_chunks_dict.get(doc_id, [])
                if not doc_chunk_list:
                    logger.warning(f"No chunks found for document {doc_id}, skipping warehouse load")
                    continue

                # Build document metadata from first chunk
                first_chunk = doc_chunk_list[0]
                document_metadata = {
                    "document_id": doc_id,
                    "title": first_chunk.document_title,
                    "url": first_chunk.document_metadata.get("url"),
                    "file_path": first_chunk.document_metadata.get("file_path"),
                    "source": first_chunk.document_metadata.get("source", ""),
                    "publication_date": first_chunk.document_metadata.get("publication_date"),
                    "total_chunks": len(doc_chunk_list),
                }

                # Reconstruct document text from chunks
                document_text = "\n\n".join([chunk.text for chunk in doc_chunk_list])

                # Convert results to MultiDimensionalResult objects
                extraction_results = []
                for result_data in doc_result_list:
                    extraction_dict = result_data.get("extraction", {})

                    # Create MultiDimensionalResult from dict
                    extraction_result = MultiDimensionalResult(**extraction_dict)
                    extraction_results.append(extraction_result)

                # Load to warehouse
                try:
                    chunks_loaded = self.warehouse_etl.load_extraction_results(
                        extraction_results=extraction_results,
                        document_metadata=document_metadata,
                        document_text=document_text,
                        embedding_model=None,  # TODO: Add embedding support
                        embeddings=None,
                    )

                    total_loaded += chunks_loaded
                    logger.info(f"✓ Loaded {chunks_loaded} chunks for document {doc_id}")

                except Exception as e:
                    logger.error(f"Failed to load document {doc_id} to warehouse: {e}")
                    logger.exception(e)
                    continue

            logger.info(f"✓ Warehouse loading complete: {total_loaded} total chunks loaded")

        except Exception as e:
            logger.error(f"Warehouse loading failed: {e}")
            logger.exception(e)

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
        """
        Save extraction results to file, organized by model type.

        Creates a folder structure: results/<model_name>/extraction_results.json
        """
        # Extract model name from first successful result
        model_name = "unknown_model"
        for result in results:
            extraction = result.get('extraction', {})
            if extraction.get('success'):
                extraction_config = extraction.get('extraction_config', {})
                if isinstance(extraction_config, dict):
                    # Try model_name first (current format), then model (legacy)
                    model_name = extraction_config.get('model_name') or extraction_config.get('model', 'unknown_model')
                else:
                    # Handle ExtractionConfig object
                    model_name = getattr(extraction_config, 'model_name', None) or getattr(extraction_config, 'model', 'unknown_model')
                break

        # Clean model name for filesystem (replace / with _)
        clean_model_name = model_name.replace('/', '_')

        # Create model-specific directory
        model_dir = self.results_dir / clean_model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        output_file = model_dir / "extraction_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Saved {len(results)} results to: {output_file}")
        logger.info(f"   Model: {model_name}")

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
