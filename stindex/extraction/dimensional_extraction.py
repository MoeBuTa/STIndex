"""
Enhanced extraction with multi-dimensional support.

Provides flexible dimensional extraction capabilities beyond temporal and spatial.
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.utils import extract_json_from_text
from stindex.llm.manager import LLMManager
from stindex.llm.prompts.dimensional_extraction import DimensionalExtractionPrompt
from stindex.llm.response.dimension_models import (
    DimensionType,
    MultiDimensionalResult,
    NormalizedDimensionEntity,
    GeocodedDimensionEntity,
    CategoricalDimensionEntity,
)
from stindex.llm.response.models import ExtractionConfig
from stindex.postprocess.spatial.geocoder import GeocoderService
from stindex.utils.dimension_loader import DimensionConfigLoader
from stindex.utils.config import load_config_from_file


class DimensionalExtractor:
    """
    Multi-dimensional extractor that supports configurable dimensions.

    Can extract any combination of dimensions defined in YAML config:
    - temporal, spatial (default)
    - event_type, venue_type, disease (health surveillance)
    - custom dimensions for other domains
    """

    def __init__(
        self,
        config_path: str = "extract",
        dimension_config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        auto_start: bool = True,
        extraction_context: Optional[ExtractionContext] = None,
    ):
        """
        Initialize dimensional extractor.

        Args:
            config_path: Path to main config file (llm provider, geocoding, etc.)
            dimension_config_path: Path to dimension config file (default: cfg/dimensions.yml)
                                  Can be domain-specific: "case_studies/public_health/extraction/config/health_dimensions"
            model_name: Override model name from config
            auto_start: Auto-start servers if not running (vLLM only)
            extraction_context: Optional ExtractionContext for context-aware extraction
        """
        # Load main configuration
        config = load_config_from_file(config_path)
        self.config = config

        # Create LLM manager
        llm_config = config.get("llm", {})
        if model_name:
            llm_config["model_name"] = model_name
            logger.info(f"Using runtime model override: {model_name}")
        if "auto_start" not in llm_config:
            llm_config["auto_start"] = auto_start

        self.llm_manager = LLMManager(llm_config)

        # Initialize geocoder
        geocoding_config = config.get("geocoding", {})
        self.geocoder = GeocoderService(
            user_agent=geocoding_config.get("user_agent", "stindex-extraction/1.0"),
            enable_cache=True,
        )

        # Load dimension configuration
        dimension_config_path = dimension_config_path or "dimensions"
        self.dimension_loader = DimensionConfigLoader()
        self.dimension_config = self.dimension_loader.load_dimension_config(dimension_config_path)
        self.dimensions = self.dimension_loader.get_enabled_dimensions(self.dimension_config)

        # Context manager for context-aware extraction
        self.extraction_context = extraction_context

        logger.info(f"✓ DimensionalExtractor initialized with {len(self.dimensions)} dimensions")
        logger.info(f"  Dimensions: {list(self.dimensions.keys())}")
        if self.extraction_context:
            logger.info("  Context-aware extraction: ENABLED")

    def extract(
        self,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> MultiDimensionalResult:
        """
        Extract multi-dimensional information from text.

        Args:
            text: Input text to extract from
            document_metadata: Optional document metadata
                - publication_date: ISO 8601 date (for relative temporal resolution)
                - source_location: Geographic context (for spatial disambiguation)
                - source_url: Original document URL
                - Any other metadata

        Returns:
            MultiDimensionalResult with entities for all dimensions
        """
        start_time = time.time()
        raw_output = None
        document_metadata = document_metadata or {}

        # Update extraction context if available
        if self.extraction_context:
            # Merge document metadata
            self.extraction_context.document_metadata.update(document_metadata)
            logger.debug(
                f"Using extraction context with {len(self.extraction_context.prior_temporal_refs)} "
                f"temporal refs, {len(self.extraction_context.prior_spatial_refs)} spatial refs"
            )

        try:
            # Step 1: Build prompt with dimension config and metadata
            logger.info("Building dimensional extraction prompt...")
            prompt_builder = DimensionalExtractionPrompt(
                dimensions=self.dimensions,
                document_metadata=document_metadata,
                extraction_context=self.extraction_context  # Pass context
            )

            # Build JSON schema for all dimensions
            json_schema = self.dimension_loader.build_json_schema(self.dimensions)

            # Build messages
            use_few_shot = self.dimension_config.get("extraction", {}).get("use_few_shot", False)
            messages = prompt_builder.build_messages_with_schema(
                text.strip(),
                json_schema=json_schema,
                use_few_shot=use_few_shot
            )

            # Step 2: Generate with LLM
            logger.info(f"Extracting {len(self.dimensions)} dimensions with LLM...")
            llm_response = self.llm_manager.generate(messages)

            if not llm_response.success:
                raise ValueError(f"LLM generation failed: {llm_response.error_msg}")

            raw_output = llm_response.content
            logger.debug(f"Raw LLM output: {raw_output[:200]}...")

            # Step 3: Extract and validate JSON
            # We need to parse it as a generic dict first since the structure is dynamic
            extraction_dict = extract_json_from_text(raw_output, None, return_dict=True)

            logger.info(f"✓ LLM extracted dimensions: {list(extraction_dict.keys())}")

            # Step 4: Process each dimension
            processed_entities = {}

            for dim_name, dim_config in self.dimensions.items():
                mentions = extraction_dict.get(dim_name, [])
                if not mentions:
                    continue

                logger.info(f"Processing {len(mentions)} {dim_name} mentions...")

                # Process based on dimension type
                extraction_type = DimensionType(dim_config.extraction_type)

                if extraction_type == DimensionType.NORMALIZED:
                    entities = self._process_normalized(mentions, dim_name, dim_config, document_metadata)
                elif extraction_type == DimensionType.GEOCODED:
                    entities = self._process_geocoded(mentions, dim_name, text, document_metadata)
                elif extraction_type == DimensionType.CATEGORICAL:
                    entities = self._process_categorical(mentions, dim_name, dim_config)
                elif extraction_type == DimensionType.STRUCTURED:
                    entities = self._process_structured(mentions, dim_name)
                else:
                    entities = self._process_free_text(mentions, dim_name)

                if entities:
                    processed_entities[dim_name] = [e.model_dump() for e in entities]

            # Step 5: Update extraction context memory if available
            if self.extraction_context:
                self.extraction_context.update_memory(processed_entities)
                logger.debug("✓ Updated extraction context memory")

            processing_time = time.time() - start_time

            # Build extraction config
            extraction_config = {
                "llm_provider": self.config.get("llm", {}).get("llm_provider", "unknown"),
                "model_name": self.config.get("llm", {}).get("model_name", "unknown"),
                "temperature": self.config.get("llm", {}).get("temperature"),
                "max_tokens": self.config.get("llm", {}).get("max_tokens"),
                "raw_llm_output": raw_output,
                "dimension_config_path": self.dimension_config.get("config_path", "dimensions"),
                "enabled_dimensions": list(self.dimensions.keys()),
                "context_aware": self.extraction_context is not None
            }

            # Build dimension metadata
            dimension_metadata = {
                dim_name: dim_config.to_metadata().model_dump()
                for dim_name, dim_config in self.dimensions.items()
            }

            return MultiDimensionalResult(
                input_text=text,
                entities=processed_entities,
                temporal_entities=processed_entities.get("temporal", []),  # Backward compat
                spatial_entities=processed_entities.get("spatial", []),    # Backward compat
                success=True,
                processing_time=processing_time,
                document_metadata=document_metadata,
                extraction_config=extraction_config,
                dimension_configs=dimension_metadata
            )

        except Exception as e:
            logger.error(f"Dimensional extraction failed: {str(e)}")

            extraction_config = None
            if raw_output:
                extraction_config = {
                    "llm_provider": self.config.get("llm", {}).get("llm_provider", "unknown"),
                    "model_name": self.config.get("llm", {}).get("model_name", "unknown"),
                    "raw_llm_output": raw_output,
                    "error": str(e)
                }

            return MultiDimensionalResult(
                input_text=text,
                entities={},
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                document_metadata=document_metadata,
                extraction_config=extraction_config
            )

    def _process_normalized(
        self,
        mentions: List[Dict],
        dim_name: str,
        dim_config,
        document_metadata: Dict
    ) -> List[NormalizedDimensionEntity]:
        """Process normalized dimensions (e.g., temporal)."""
        entities = []

        for mention in mentions:
            # TODO: Add relative temporal resolution here if needed
            # For now, assume LLM already normalized

            entity = NormalizedDimensionEntity(
                text=mention.get("text", ""),
                dimension_name=dim_name,
                normalized=mention.get("normalized", ""),
                normalization_type=mention.get(list(mention.keys())[2] if len(mention) > 2 else "type", ""),
                confidence=mention.get("confidence", 0.95)
            )
            entities.append(entity)

        return entities

    def _process_geocoded(
        self,
        mentions: List[Dict],
        dim_name: str,
        document_text: str,
        document_metadata: Dict
    ) -> List[GeocodedDimensionEntity]:
        """Process geocoded dimensions (e.g., spatial)."""
        entities = []

        for mention in mentions:
            location_text = mention.get("text", "")
            parent_region = mention.get("parent_region")

            # Geocode
            try:
                coords = self.geocoder.get_coordinates(
                    location=location_text,
                    context=document_text,
                    parent_region=parent_region
                )

                if coords:
                    lat, lon = coords
                    entity = GeocodedDimensionEntity(
                        text=location_text,
                        dimension_name=dim_name,
                        latitude=lat,
                        longitude=lon,
                        location_type=mention.get("location_type"),
                        confidence=0.95
                    )
                    entities.append(entity)
                else:
                    logger.warning(f"Geocoding failed for: {location_text}")

            except Exception as e:
                logger.warning(f"Error geocoding '{location_text}': {e}")

        return entities

    def _process_categorical(
        self,
        mentions: List[Dict],
        dim_name: str,
        dim_config
    ) -> List[CategoricalDimensionEntity]:
        """Process categorical dimensions (e.g., event_type, disease)."""
        entities = []

        for mention in mentions:
            entity = CategoricalDimensionEntity(
                text=mention.get("text", ""),
                dimension_name=dim_name,
                category=mention.get("category", "unknown"),
                category_confidence=mention.get("confidence", mention.get("category_confidence", 1.0)),
                confidence=mention.get("confidence", 1.0)
            )
            entities.append(entity)

        return entities

    def _process_structured(
        self,
        mentions: List[Dict],
        dim_name: str
    ) -> List:
        """Process structured dimensions."""
        # For now, return as-is (will be enhanced later)
        entities = []
        for mention in mentions:
            entities.append({
                "text": mention.get("text", ""),
                "dimension_name": dim_name,
                "fields": mention.get("fields", mention),
                "confidence": mention.get("confidence", 1.0)
            })
        return entities

    def _process_free_text(
        self,
        mentions: List[Dict],
        dim_name: str
    ) -> List:
        """Process free-text dimensions."""
        entities = []
        for mention in mentions:
            entities.append({
                "text": mention.get("text", ""),
                "dimension_name": dim_name,
                "confidence": mention.get("confidence", 1.0)
            })
        return entities
