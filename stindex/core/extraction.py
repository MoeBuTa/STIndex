"""
Clean spatiotemporal extractor with native LLM providers.

Simple, direct extraction with LLM doing all the heavy lifting.
"""

import time
from typing import Any, Dict

from loguru import logger

from stindex.core.utils import extract_json_from_text
from stindex.llm.manager import LLMManager
from stindex.llm.prompts.extraction import ExtractionPrompt
from stindex.llm.response.models import (
    ExtractionConfig,
    ExtractionResult,
    SpatialEntity,
    SpatioTemporalResult,
    TemporalEntity,
)
from stindex.spatio.geocoder import GeocoderService
from stindex.utils.config import load_config_from_file


class STIndexExtractor:
    """
    Simple spatiotemporal extractor using native LLM providers.

    Clean architecture: LLM extracts and normalizes, geocoder adds coordinates.
    """

    def __init__(
        self,
        config_path: str = "extract",
    ):
        """
        Initialize extractor.

        Args:
            config_path: Path to config file (default: "extract" loads cfg/extract.yml)
                        The config file specifies the LLM provider (hf/openai/anthropic)
                        and automatically loads provider-specific settings
        """
        # Load configuration from file
        config = load_config_from_file(config_path)
        self.config = config

        # Create LLM manager with config from llm section
        llm_config = config.get("llm", {})
        self.llm_manager = LLMManager(llm_config)

        # Initialize geocoder with config
        geocoding_config = config.get("geocoding", {})
        self.geocoder = GeocoderService(
            user_agent=geocoding_config.get("user_agent", "stindex-extraction/1.0"),
            enable_cache=True,
        )

        logger.info("✓ STIndexExtractor initialized")

    def extract(self, text: str) -> SpatioTemporalResult:
        """
        Extract spatiotemporal information from text.

        Args:
            text: Input text to extract from

        Returns:
            SpatioTemporalResult with temporal and spatial entities
        """
        start_time = time.time()

        try:
            # Step 1: Build messages using prompt module
            logger.info("Extracting entities with LLM...")
            messages = ExtractionPrompt.build_messages_with_schema(
                text.strip(),
                response_model=ExtractionResult,
                use_few_shot=False
            )

            # Step 2: Generate with LLM
            llm_response = self.llm_manager.generate(messages)

            # Check if LLM call was successful
            if not llm_response.success:
                raise ValueError(f"LLM generation failed: {llm_response.error_msg}")

            raw_output = llm_response.content
            logger.debug(f"LLM usage: {llm_response.usage}")
            logger.debug(f"Raw output: {raw_output}...")

            # Step 3: Extract and validate JSON
            extraction = extract_json_from_text(raw_output, ExtractionResult)

            logger.info(
                f"✓ LLM extracted {len(extraction.temporal_mentions)} temporal "
                f"and {len(extraction.spatial_mentions)} spatial mentions"
            )

            # Step 4: Process temporal entities (just add metadata)
            temporal_entities = self._process_temporal(extraction.temporal_mentions, text)

            # Step 5: Process spatial entities (geocode)
            spatial_entities = self._process_spatial(extraction.spatial_mentions, text)

            processing_time = time.time() - start_time

            # Build extraction config info with raw LLM output
            extraction_config = ExtractionConfig(
                llm_provider=self.config.get("llm", {}).get("llm_provider", "unknown"),
                model_name=self.config.get("llm", {}).get("model_name", "unknown"),
                temperature=self.config.get("llm", {}).get("temperature"),
                max_tokens=self.config.get("llm", {}).get("max_tokens"),
                raw_llm_output=raw_output,
            )

            return SpatioTemporalResult(
                input_text=text,
                temporal_entities=temporal_entities,
                spatial_entities=spatial_entities,
                success=True,
                processing_time=processing_time,
                extraction_config=extraction_config,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return SpatioTemporalResult(
                input_text=text,
                temporal_entities=[],
                spatial_entities=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _process_temporal(self, mentions, document_text: str) -> list[TemporalEntity]:
        """Process temporal mentions."""
        entities = []

        for mention in mentions:
            entity = TemporalEntity(
                text=mention.text,
                normalized=mention.normalized,
                temporal_type=mention.temporal_type,
                confidence=0.95,
            )
            entities.append(entity)

        return entities

    def _process_spatial(self, mentions, document_text: str) -> list[SpatialEntity]:
        """Process spatial mentions (geocode to coordinates)."""
        entities = []

        for mention in mentions:
            # Geocode with parent region and context
            # Context allows geocoder to extract parent region via spaCy if LLM missed it
            try:
                coords = self.geocoder.get_coordinates(
                    location=mention.text,
                    context=document_text,
                    parent_region=mention.parent_region,
                )

                if coords:
                    lat, lon = coords

                    entity = SpatialEntity(
                        text=mention.text,
                        latitude=lat,
                        longitude=lon,
                        location_type=mention.location_type,
                        confidence=0.95,
                    )
                    entities.append(entity)
                else:
                    logger.warning(f"Geocoding failed for: {mention.text}")

            except Exception as e:
                logger.warning(f"Error geocoding '{mention.text}': {e}")
                continue

        return entities

    def extract_batch(self, texts: list[str], use_batch_api: bool = False) -> list[SpatioTemporalResult]:
        """
        Extract from multiple texts.

        Args:
            texts: List of input texts
            use_batch_api: If True, uses LLM batch API for parallel processing (faster with multi-GPU)
                          If False, processes one-by-one (default, safer)

        Returns:
            List of SpatioTemporalResult objects
        """
        if not use_batch_api:
            # Single-item mode: process one-by-one (current behavior)
            return [self.extract(text) for text in texts]

        # Batch mode: use LLM batch API for parallel processing
        try:
            start_time = time.time()

            # Step 1: Build messages for all texts
            messages_batch = []
            for text in texts:
                messages = ExtractionPrompt.build_messages_with_schema(
                    text.strip(),
                    response_model=ExtractionResult,
                    use_few_shot=False
                )
                messages_batch.append(messages)

            # Step 2: Send batch to LLM
            llm_responses = self.llm_manager.generate_batch(messages_batch)

            # Step 3: Process each response
            results = []
            for i, (text, llm_response) in enumerate(zip(texts, llm_responses)):
                item_start_time = time.time()

                try:
                    # Check if LLM call was successful
                    if not llm_response.success:
                        raise ValueError(f"LLM generation failed: {llm_response.error_msg}")

                    raw_output = llm_response.content

                    # Extract and validate JSON
                    extraction = extract_json_from_text(raw_output, ExtractionResult)

                    # Process temporal entities
                    temporal_entities = self._process_temporal(extraction.temporal_mentions, text)

                    # Process spatial entities (geocode)
                    spatial_entities = self._process_spatial(extraction.spatial_mentions, text)

                    processing_time = time.time() - item_start_time

                    # Build extraction config info
                    extraction_config = ExtractionConfig(
                        llm_provider=self.config.get("llm", {}).get("llm_provider", "unknown"),
                        model_name=self.config.get("llm", {}).get("model_name", "unknown"),
                        temperature=self.config.get("llm", {}).get("temperature"),
                        max_tokens=self.config.get("llm", {}).get("max_tokens"),
                        raw_llm_output=raw_output,
                    )

                    results.append(SpatioTemporalResult(
                        input_text=text,
                        temporal_entities=temporal_entities,
                        spatial_entities=spatial_entities,
                        success=True,
                        processing_time=processing_time,
                        extraction_config=extraction_config,
                    ))

                except Exception as e:
                    logger.error(f"Extraction failed for text {i+1}: {str(e)}")
                    results.append(SpatioTemporalResult(
                        input_text=text,
                        temporal_entities=[],
                        spatial_entities=[],
                        success=False,
                        error=str(e),
                        processing_time=time.time() - item_start_time,
                    ))

            return results

        except Exception as e:
            # Fallback to single-item mode if batch fails
            logger.error(f"Batch extraction failed: {str(e)}, falling back to single-item mode")
            return [self.extract(text) for text in texts]
