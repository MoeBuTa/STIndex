"""
Enhanced extraction with multi-dimensional support.

Provides flexible dimensional extraction capabilities beyond temporal and spatial.
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.utils import extract_json_from_text
from stindex.llm.base import create_client
from stindex.llm.prompts.dimensional_extraction import DimensionalExtractionPrompt
from stindex.extraction.dimension_models import (
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
        dimension_overrides: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
        auto_start: bool = True,
        extraction_context: Optional[ExtractionContext] = None,
        prompt_mode: str = "default",  # "default" or "corpus"
        enable_dimension_discovery: bool = False,
        discovery_confidence_threshold: float = 0.9,
        schema_output_path: Optional[str] = None,
    ):
        """
        Initialize dimensional extractor.

        Args:
            config_path: Path to main config file (llm provider, geocoding, etc.)
            dimension_config_path: Path to dimension config file (default: cfg/dimensions.yml)
            dimension_overrides: Optional path to override config (e.g., to disable dimensions)
            model_name: Override model name from config
            temperature: Override sampling temperature from config
            max_tokens: Override max tokens from config
            base_url: Override LLM server base URL from config (for hf provider)
            auto_start: Auto-start servers if not running (vLLM only)
            extraction_context: Optional ExtractionContext for context-aware extraction
            prompt_mode: Prompt template mode - "default" for general text, "corpus" for corpus/textbook documents
            enable_dimension_discovery: Whether to allow LLM to propose new dimensions during extraction
            discovery_confidence_threshold: Minimum confidence (0.0-1.0) for accepting proposed dimensions
            schema_output_path: Path to schema file for auto-update when new dimensions discovered
        """
        # Load main configuration
        config = load_config_from_file(config_path)
        self.config = config
        self.prompt_mode = prompt_mode

        # Create LLM manager — apply any runtime overrides
        llm_config = config.get("llm", {})
        if model_name is not None:
            llm_config["model_name"] = model_name
            logger.info(f"Using runtime model override: {model_name}")
        if temperature is not None:
            llm_config["temperature"] = temperature
        if max_tokens is not None:
            llm_config["max_tokens"] = max_tokens
        if base_url is not None:
            llm_config["base_url"] = base_url
        if "auto_start" not in llm_config:
            llm_config["auto_start"] = auto_start

        self.llm_client = create_client(llm_config)

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
        self.dimension_config = self.dimension_loader.load_dimension_config(
            dimension_config_path,
            override_config_path=dimension_overrides
        )
        self.dimensions = self.dimension_loader.get_enabled_dimensions(self.dimension_config)
        self.dimension_overrides_path = dimension_overrides

        # Dimension discovery settings
        self.enable_dimension_discovery = enable_dimension_discovery
        self.discovery_confidence_threshold = discovery_confidence_threshold
        self.schema_output_path = schema_output_path
        self.proposed_dimensions: Dict[str, Any] = {}  # Accumulated proposed dimensions

        # Context manager for context-aware extraction
        self.extraction_context = extraction_context

        logger.info(f"✓ DimensionalExtractor initialized with {len(self.dimensions)} dimensions")
        logger.info(f"  Dimensions: {list(self.dimensions.keys())}")
        if self.extraction_context:
            logger.info("  Context-aware extraction: ENABLED")
        if self.enable_dimension_discovery:
            logger.info(f"  Dimension discovery: ENABLED (threshold: {self.discovery_confidence_threshold})")
            if self.schema_output_path:
                logger.info(f"  Schema auto-update: {self.schema_output_path}")

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
        component_timings = {}  # Track component timing

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
                extraction_context=self.extraction_context,  # Pass context
                mode=self.prompt_mode,  # Pass corpus/default mode
                enable_dimension_discovery=self.enable_dimension_discovery,
                discovery_confidence_threshold=self.discovery_confidence_threshold
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
            _sys = next((m["content"] for m in messages if m["role"] == "system"), "")
            _usr = next((m["content"] for m in messages if m["role"] == "user"), "")
            raw_output = self.llm_client.generate(_sys, _usr)
            logger.debug(f"Raw LLM output: {raw_output}...")

            # Step 3: Extract and validate JSON
            # We need to parse it as a generic dict first since the structure is dynamic
            extraction_dict = extract_json_from_text(raw_output, None, return_dict=True)

            logger.info(f"✓ LLM extracted dimensions: {list(extraction_dict.keys())}")

            # Step 3.25: Validate extracted dimensions against schema
            extraction_dict = self._validate_and_filter_extraction(extraction_dict)

            # Step 3.5: Process proposed dimensions if discovery enabled
            proposed_dimensions_result = {}
            if self.enable_dimension_discovery:
                proposed = extraction_dict.pop("_proposed_dimensions", {})
                if proposed:
                    proposed_dimensions_result = self._process_proposed_dimensions(proposed, raw_output)

            # Step 4: Process each dimension
            processed_entities = {}
            postprocess_start = time.time()  # Start tracking postprocessing

            for dim_name, dim_config in self.dimensions.items():
                mentions = extraction_dict.get(dim_name, [])
                if not mentions:
                    continue

                logger.info(f"Processing {len(mentions)} {dim_name} mentions...")

                # Process based on dimension type
                extraction_type = DimensionType(dim_config.extraction_type)

                if extraction_type == DimensionType.NORMALIZED:
                    # Track temporal resolution time if this is the temporal dimension
                    if dim_name == "temporal":
                        temporal_start = time.time()
                    entities = self._process_normalized(mentions, dim_name, dim_config, document_metadata)
                    if dim_name == "temporal":
                        component_timings["temporal_resolution_seconds"] = round(time.time() - temporal_start, 3)
                elif extraction_type == DimensionType.GEOCODED:
                    # Track geocoding time
                    geocoding_start = time.time()
                    entities = self._process_geocoded(mentions, dim_name, text, document_metadata)
                    component_timings["geocoding_seconds"] = round(time.time() - geocoding_start, 3)
                elif extraction_type == DimensionType.CATEGORICAL:
                    # Track categorical validation time
                    validation_start = time.time()
                    entities = self._process_categorical(mentions, dim_name, dim_config)
                    component_timings["categorical_validation_seconds"] = round(time.time() - validation_start, 3)
                elif extraction_type == DimensionType.STRUCTURED:
                    entities = self._process_structured(mentions, dim_name)
                else:
                    entities = self._process_free_text(mentions, dim_name)

                if entities:
                    # Handle both Pydantic models and dicts
                    if entities and hasattr(entities[0], 'model_dump'):
                        # Pydantic models (from _process_geocoded, _process_normalized, _process_categorical)
                        processed_entities[dim_name] = [e.model_dump() for e in entities]
                    else:
                        # Already dicts (from _process_structured, _process_free_text)
                        processed_entities[dim_name] = entities

            # Track total postprocessing time
            component_timings["postprocessing_seconds"] = round(time.time() - postprocess_start, 3)

            # Step 4.5: Retry if no entities extracted (aggressive extraction)
            MAX_RETRIES = 2
            retry_count = 0

            while not processed_entities and retry_count < MAX_RETRIES:
                retry_count += 1
                logger.warning(f"No entities extracted, retrying ({retry_count}/{MAX_RETRIES})...")

                # Add error message to prompt
                retry_msg = "\n\nPREVIOUS ATTEMPT FAILED: No entities were extracted. You MUST extract at least one entity. Look more carefully at the text and identify concepts that match any dimension."

                # Rebuild messages with error appended
                messages_with_retry = prompt_builder.build_messages_with_schema(
                    text.strip() + retry_msg,
                    json_schema=json_schema,
                    use_few_shot=use_few_shot
                )

                # Retry LLM call
                try:
                    _sys = next((m["content"] for m in messages_with_retry if m["role"] == "system"), "")
                    _usr = next((m["content"] for m in messages_with_retry if m["role"] == "user"), "")
                    raw_output = self.llm_client.generate(_sys, _usr)
                    logger.debug(f"Retry {retry_count} raw output: {raw_output[:200]}...")

                    extraction_dict = extract_json_from_text(raw_output, None, return_dict=True)
                    logger.info(f"Retry {retry_count} extracted: {list(extraction_dict.keys())}")

                    # Re-process entities
                    for dim_name, dim_config in self.dimensions.items():
                        mentions = extraction_dict.get(dim_name, [])
                        if not mentions:
                            continue

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
                            if hasattr(entities[0], 'model_dump'):
                                processed_entities[dim_name] = [e.model_dump() for e in entities]
                            else:
                                processed_entities[dim_name] = entities
                except Exception as e:
                    logger.warning(f"Retry {retry_count} failed: {e}")
                    continue

            if not processed_entities:
                logger.warning(f"No entities extracted after {MAX_RETRIES} retries")

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
                "context_aware": self.extraction_context is not None,
                "proposed_dimensions": proposed_dimensions_result if proposed_dimensions_result else None
            }

            # Build dimension metadata
            dimension_metadata = {}
            for dim_name, dim_config in self.dimensions.items():
                try:
                    # Handle DimensionConfig objects
                    if hasattr(dim_config, 'to_metadata'):
                        dimension_metadata[dim_name] = dim_config.to_metadata().model_dump()
                    elif isinstance(dim_config, dict):
                        # Handle dict case (shouldn't happen but be defensive)
                        dimension_metadata[dim_name] = dim_config
                    else:
                        logger.warning(f"Unexpected dimension config type for {dim_name}: {type(dim_config)}")
                except Exception as e:
                    logger.warning(f"Failed to build metadata for dimension {dim_name}: {e}")
                    # Continue without this dimension's metadata

            return MultiDimensionalResult(
                input_text=text,
                entities=processed_entities,
                temporal_entities=processed_entities.get("temporal", []),  # Backward compat
                spatial_entities=processed_entities.get("spatial", []),    # Backward compat
                success=True,
                processing_time=processing_time,
                component_timings=component_timings,  # Add component timing breakdown
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
            # Skip if mention is not a dict (malformed LLM output)
            if not isinstance(mention, dict):
                logger.warning(f"Skipping non-dict mention in {dim_name}: {type(mention)}")
                continue

            text = mention.get("text", "")
            normalized = mention.get("normalized", "")
            # For temporal hierarchy format, fall back to timestamp → date if normalized is empty
            if not normalized and dim_name == "temporal":
                normalized = (
                    mention.get("timestamp") or
                    mention.get("datetime") or
                    mention.get("date") or
                    ""
                )
            # Get normalization type from explicit key or infer from hierarchy fields present
            normalization_type = mention.get("type") or mention.get("normalization_type") or ""
            if not normalization_type and dim_name == "temporal":
                if mention.get("timestamp") or mention.get("datetime"):
                    normalization_type = "datetime"
                elif mention.get("date"):
                    normalization_type = "date"
                elif mention.get("month"):
                    normalization_type = "month"
                elif mention.get("year"):
                    normalization_type = "year"

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
            # Skip if mention is not a dict (malformed LLM output)
            if not isinstance(mention, dict):
                logger.warning(f"Skipping non-dict mention in {dim_name}: {type(mention)}")
                continue

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
            # Skip if mention is not a dict (malformed LLM output)
            if not isinstance(mention, dict):
                logger.warning(f"Skipping non-dict mention in {dim_name}: {type(mention)}")
                continue

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
        """
        Process structured dimensions with hierarchy fields.

        Flattens all hierarchy fields to top level for labeler compatibility.
        The labeler expects entity.get("level_name") not entity["fields"]["level_name"].
        """
        entities = []
        for mention in mentions:
            # Skip if mention is not a dict (malformed LLM output)
            if not isinstance(mention, dict):
                logger.warning(f"Skipping non-dict mention in {dim_name}: {type(mention)}")
                continue

            # Start with base fields
            entity = {
                "text": mention.get("text", ""),
                "dimension_name": dim_name,
                "confidence": mention.get("confidence", 1.0)
            }
            # Flatten all hierarchy fields to top level for labeler compatibility
            for key, value in mention.items():
                if key not in ["text", "confidence"]:
                    entity[key] = value
            entities.append(entity)
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

    def _validate_and_filter_extraction(
        self,
        extraction_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate extracted dimensions and hierarchy fields against schema.

        - Filters out dimensions not in schema
        - Filters out invalid hierarchy fields per entity
        - Logs warnings for mismatches

        Args:
            extraction_dict: Raw LLM extraction output

        Returns:
            Filtered extraction dict with only valid dimensions and fields
        """
        valid_dim_names = set(self.dimensions.keys())
        validated = {}

        for dim_name, entities in extraction_dict.items():
            # Skip internal keys
            if dim_name.startswith("_"):
                validated[dim_name] = entities
                continue

            # Check if dimension exists in schema
            if dim_name not in valid_dim_names:
                logger.warning(f"⚠ Invalid dimension '{dim_name}' not in schema - skipping")
                continue

            if not entities or not isinstance(entities, list):
                validated[dim_name] = entities
                continue

            # Get valid hierarchy fields for this dimension
            dim_config = self.dimensions[dim_name]
            valid_fields = {"text", "confidence"}  # Always valid

            # Add hierarchy level names as valid fields
            if dim_config.hierarchy:
                for level in dim_config.hierarchy:
                    if isinstance(level, dict):
                        valid_fields.add(level.get("level", ""))
                    else:
                        valid_fields.add(str(level))

            # Also add fields from config
            if dim_config.fields:
                for field in dim_config.fields:
                    if isinstance(field, dict):
                        valid_fields.add(field.get("name", ""))
                    else:
                        valid_fields.add(str(field))

            # Validate each entity
            validated_entities = []
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                validated_entity = {}
                for key, value in entity.items():
                    if key in valid_fields:
                        validated_entity[key] = value
                    else:
                        logger.debug(f"⚠ Invalid field '{key}' for dimension '{dim_name}' - removing")

                if validated_entity:
                    validated_entities.append(validated_entity)

            validated[dim_name] = validated_entities

        # Log summary
        removed_dims = set(extraction_dict.keys()) - set(validated.keys()) - {"_proposed_dimensions"}
        if removed_dims:
            logger.info(f"Schema validation: removed {len(removed_dims)} invalid dimensions: {removed_dims}")

        return validated

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

    def _process_proposed_dimensions(
        self,
        proposed: Dict[str, Any],
        raw_output: str
    ) -> Dict[str, Any]:
        """
        Process and validate proposed dimensions from LLM.

        Args:
            proposed: Dict of proposed dimensions from LLM output
            raw_output: Raw LLM output (for logging)

        Returns:
            Dict with accepted/rejected dimension info
        """
        from stindex.discovery.models import DiscoveredDimensionSchema

        result = {}

        for dim_name, dim_data in proposed.items():
            confidence = dim_data.get("confidence", 0.0)
            justification = dim_data.get("justification", "")
            hierarchy = dim_data.get("hierarchy", [])
            entities = dim_data.get("entities", [])

            # Log proposal (always, even if rejected)
            logger.info(f"Proposed dimension '{dim_name}': confidence={confidence:.2f}")
            logger.debug(f"  Hierarchy: {hierarchy}")
            logger.debug(f"  Justification: {justification}")
            logger.debug(f"  Entities: {len(entities)}")

            # Filter by confidence threshold
            if confidence < self.discovery_confidence_threshold:
                logger.info(f"  ✗ Rejected (confidence {confidence:.2f} < {self.discovery_confidence_threshold})")
                result[dim_name] = {
                    "accepted": False,
                    "confidence": confidence,
                    "reason": f"confidence below threshold ({self.discovery_confidence_threshold})"
                }
                continue

            # Check if dimension already exists
            if dim_name in self.dimensions:
                logger.info(f"  ✗ Rejected (dimension '{dim_name}' already exists)")
                result[dim_name] = {
                    "accepted": False,
                    "confidence": confidence,
                    "reason": "dimension already exists"
                }
                continue

            # Accept dimension
            logger.info(f"  ✓ Accepted (confidence {confidence:.2f})")

            # Create schema from proposed dimension
            try:
                schema = DiscoveredDimensionSchema(
                    hierarchy=hierarchy if hierarchy else [dim_name.lower().replace(' ', '_')],
                    description=f"Auto-discovered: {dim_name}",
                    examples=[e.get("text", "") for e in entities[:5]]
                )

                # Store in accumulated proposed dimensions
                self.proposed_dimensions[dim_name] = schema

                result[dim_name] = {
                    "accepted": True,
                    "confidence": confidence,
                    "hierarchy": schema.hierarchy,
                    "entities_count": len(entities)
                }

                # Update schema file if path provided
                if self.schema_output_path:
                    self._update_schema_file(dim_name, schema)

            except Exception as e:
                logger.error(f"  ✗ Failed to create schema for '{dim_name}': {e}")
                result[dim_name] = {
                    "accepted": False,
                    "confidence": confidence,
                    "reason": f"schema creation failed: {str(e)}"
                }

        return result

    def _update_schema_file(self, dim_name: str, schema: 'DiscoveredDimensionSchema'):
        """
        Append new dimension to extraction schema YAML file.

        Args:
            dim_name: Name of the new dimension
            schema: DiscoveredDimensionSchema for the dimension
        """
        import yaml
        from pathlib import Path

        schema_path = Path(self.schema_output_path)
        if not schema_path.suffix:
            schema_path = schema_path.with_suffix('.yml')

        # Load existing schema
        if schema_path.exists():
            with open(schema_path) as f:
                existing = yaml.safe_load(f) or {}
        else:
            existing = {}

        # Add new dimension (use snake_case key)
        dim_key = dim_name.lower().replace(' ', '_')
        existing[dim_key] = {
            'hierarchy': schema.hierarchy
        }

        # Write back
        with open(schema_path, 'w') as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

        logger.info(f"  → Updated schema file: {schema_path}")

        # Also update in-memory dimensions for next extraction
        self._reload_dimensions()

    def _reload_dimensions(self):
        """Reload dimensions from updated schema file."""
        if self.schema_output_path:
            try:
                self.dimension_config = self.dimension_loader.load_dimension_config(
                    self.schema_output_path,
                    override_config_path=self.dimension_overrides_path
                )
                self.dimensions = self.dimension_loader.get_enabled_dimensions(self.dimension_config)
                logger.info(f"  → Reloaded {len(self.dimensions)} dimensions")
            except Exception as e:
                logger.warning(f"  → Failed to reload dimensions: {e}")

    def get_proposed_dimensions(self) -> Dict[str, Any]:
        """
        Get all proposed dimensions accumulated during extraction.

        Returns:
            Dict of dimension_name → DiscoveredDimensionSchema
        """
        return self.proposed_dimensions.copy()

