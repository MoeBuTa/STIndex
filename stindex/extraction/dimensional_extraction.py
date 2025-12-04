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
from stindex.postprocess.spatial.geocoder import GeocoderService
from stindex.postprocess.spatial.osm_context import OSMContextProvider
from stindex.postprocess.temporal.relative_resolver import RelativeTemporalResolver
from stindex.postprocess.categorical_validator import CategoricalValidator
from stindex.extraction.dimension_loader import DimensionConfigLoader
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

        # Initialize spatial post-processors (loads from cfg/extraction/postprocess/spatial.yml)
        self.geocoder = GeocoderService()

        # Initialize OSM context provider for nearby location context
        spatial_config = config.get("spatial", {})
        enable_osm_context = spatial_config.get("enable_osm_context", False)
        if enable_osm_context:
            osm_radius = spatial_config.get("osm_radius_km", 100)
            osm_max_results = spatial_config.get("osm_max_results", 10)
            self.osm_context = OSMContextProvider(
                max_results=osm_max_results
            )
            self.osm_radius_km = osm_radius
            logger.debug(f"✓ OSM context provider enabled (radius: {osm_radius}km, max_results: {osm_max_results})")
        else:
            self.osm_context = None
            self.osm_radius_km = 100  # Default fallback

        # Initialize temporal post-processor
        temporal_config = config.get("temporal", {})
        enable_relative_resolution = temporal_config.get("enable_relative_resolution", True)
        timezone = temporal_config.get("timezone", "UTC")
        if enable_relative_resolution:
            self.temporal_resolver = RelativeTemporalResolver(timezone=timezone)
            logger.debug(f"✓ Temporal resolver enabled (timezone: {timezone})")
        else:
            self.temporal_resolver = None

        # Initialize categorical validator
        categorical_config = config.get("categorical", {})
        enable_categorical_validation = categorical_config.get("enable_validation", True)
        if enable_categorical_validation:
            strict_mode = categorical_config.get("strict_mode", False)
            self.categorical_validator = CategoricalValidator(strict_mode=strict_mode)
            logger.debug(f"✓ Categorical validator enabled (strict_mode: {strict_mode})")
        else:
            self.categorical_validator = None

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
        document_metadata: Optional[Dict[str, Any]] = None,
        update_context: bool = True
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
            update_context: Whether to update extraction context with results (default: True)
                           Set to False if reflection will be applied afterward

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
            logger.debug(f"Raw LLM output: {raw_output}...")

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

            # Step 5: Update extraction context memory if requested
            # Note: Set update_context=False if reflection will be applied afterward
            if self.extraction_context and update_context:
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
        """Process normalized dimensions (e.g., temporal) with relative resolution."""
        entities = []

        for mention in mentions:
            text = mention.get("text", "")
            normalized = mention.get("normalized", "")
            normalization_type = mention.get(list(mention.keys())[2] if len(mention) > 2 else "type", "")

            # Apply relative temporal resolution if available and dimension is temporal
            if self.temporal_resolver and dim_name == "temporal" and normalized:
                try:
                    # Get document publication date for anchor
                    publication_date = document_metadata.get("publication_date")

                    # Resolve relative expressions to absolute dates
                    resolved_normalized, resolved_type = self.temporal_resolver.resolve(
                        temporal_text=normalized,
                        document_date=publication_date,
                        temporal_type=normalization_type
                    )

                    # Update normalized value and type if resolution succeeded
                    if resolved_normalized != normalized:
                        logger.debug(f"Resolved temporal: '{normalized}' → '{resolved_normalized}'")
                        normalized = resolved_normalized
                        normalization_type = resolved_type

                except Exception as e:
                    logger.warning(f"Temporal resolution failed for '{normalized}': {e}")
                    # Continue with original normalized value

            entity = NormalizedDimensionEntity(
                text=text,
                dimension_name=dim_name,
                normalized=normalized,
                normalization_type=normalization_type,
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
        """Process geocoded dimensions (e.g., spatial) with optional OSM context."""
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

                    # Get nearby location context if OSM provider is enabled
                    nearby_locations = None
                    if self.osm_context:
                        try:
                            nearby_locations = self.osm_context.get_nearby_locations(
                                location=coords,
                                radius_km=self.osm_radius_km
                            )
                            if nearby_locations:
                                logger.debug(f"Found {len(nearby_locations)} nearby locations for '{location_text}'")
                        except Exception as e:
                            logger.debug(f"OSM context retrieval failed for '{location_text}': {e}")
                            # Continue without nearby context

                    entity = GeocodedDimensionEntity(
                        text=location_text,
                        dimension_name=dim_name,
                        latitude=lat,
                        longitude=lon,
                        location_type=mention.get("location_type"),
                        confidence=0.95
                    )

                    # Add nearby locations as metadata if available
                    if nearby_locations and hasattr(entity, 'metadata'):
                        entity.metadata = entity.metadata or {}
                        entity.metadata['nearby_locations'] = nearby_locations

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
        """
        Process categorical dimensions (e.g., event_type, disease).

        Validates that extracted categories match predefined allowed values
        from dimension config, with normalization and fuzzy matching.
        """
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

        # Validate categories against allowed values if validator enabled
        if self.categorical_validator and entities:
            # Convert to dict for validation
            entity_dicts = [e.model_dump() for e in entities]

            # Validate
            validated_dicts = self.categorical_validator.validate_entities(
                entity_dicts,
                dim_config,
                dim_name
            )

            # Convert back to Pydantic models
            entities = [
                CategoricalDimensionEntity(**validated_dict)
                for validated_dict in validated_dicts
            ]

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

    def update_context_memory(self, entities: Dict[str, List[Dict]]):
        """
        Update extraction context with processed/reflected entities.

        Use this method after reflection to update context with filtered entities.
        This ensures context memory only contains high-quality extractions.

        Args:
            entities: Dictionary of {dimension_name: [entity_dicts]}

        Example:
            # Extract without context update
            result = extractor.extract(text, update_context=False)

            # Apply reflection
            reflected_entities = reflector.reflect_on_extractions(text, result.entities)

            # Update context with reflected entities
            extractor.update_context_memory(reflected_entities)
        """
        if self.extraction_context:
            self.extraction_context.update_memory(entities)
            logger.debug(f"✓ Updated extraction context with {sum(len(v) for v in entities.values())} entities")
        else:
            logger.warning("Cannot update context: extraction_context is not initialized")
