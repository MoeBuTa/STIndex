"""
Clean spatiotemporal extractor using Instructor framework.

Simple, direct extraction with LLM doing all the heavy lifting.
"""

import time
from typing import Any, Dict, Optional

from loguru import logger

from stindex.agents.llm.client import UnifiedLLMClient, create_llm_client
from stindex.agents.prompts.extraction import ExtractionPrompt
from stindex.agents.response.models import (
    ExtractionResult,
    SpatialEntity,
    SpatioTemporalResult,
    TemporalEntity,
)
from stindex.spatio.geocoder import GeocoderService
from stindex.utils.config import load_config_from_file


class STIndexExtractor:
    """
    Simple spatiotemporal extractor using Instructor.

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

        # Create LLM client from config
        llm_config = config.get("llm", {})
        self.llm = create_llm_client(llm_config)

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
            # Step 1: LLM extraction (single call, structured output)
            logger.info("Extracting entities with LLM...")
            messages = ExtractionPrompt.build_messages(text.strip(), use_few_shot=False)

            extraction = self.llm.extract(
                messages=messages,
                response_model=ExtractionResult,
            )

            logger.info(
                f"✓ LLM extracted {len(extraction.temporal_mentions)} temporal "
                f"and {len(extraction.spatial_mentions)} spatial mentions"
            )

            # Step 2: Process temporal entities (just add metadata)
            temporal_entities = self._process_temporal(extraction.temporal_mentions, text)

            # Step 3: Process spatial entities (geocode)
            spatial_entities = self._process_spatial(extraction.spatial_mentions, text)

            processing_time = time.time() - start_time

            return SpatioTemporalResult(
                temporal_entities=temporal_entities,
                spatial_entities=spatial_entities,
                success=True,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return SpatioTemporalResult(
                temporal_entities=[],
                spatial_entities=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _process_temporal(self, mentions, document_text: str) -> list[TemporalEntity]:
        """Process temporal mentions (add character positions)."""
        entities = []

        for mention in mentions:
            start_char = document_text.find(mention.text)
            end_char = start_char + len(mention.text) if start_char != -1 else None

            entity = TemporalEntity(
                text=mention.text,
                normalized=mention.normalized,
                temporal_type=mention.temporal_type,
                confidence=0.95,
                start_char=start_char if start_char != -1 else None,
                end_char=end_char,
            )
            entities.append(entity)

        logger.info(f"✓ Processed {len(entities)} temporal entities")
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
                    start_char = document_text.find(mention.text)
                    end_char = start_char + len(mention.text) if start_char != -1 else None

                    entity = SpatialEntity(
                        text=mention.text,
                        latitude=lat,
                        longitude=lon,
                        location_type=mention.location_type,
                        confidence=0.95,
                        start_char=start_char if start_char != -1 else None,
                        end_char=end_char,
                    )
                    entities.append(entity)
                else:
                    logger.warning(f"Geocoding failed for: {mention.text}")

            except Exception as e:
                logger.warning(f"Error geocoding '{mention.text}': {e}")
                continue

        logger.info(f"✓ Processed {len(entities)} spatial entities")
        return entities

    def extract_batch(self, texts: list[str]) -> list[SpatioTemporalResult]:
        """
        Extract from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of SpatioTemporalResult objects
        """
        return [self.extract(text) for text in texts]
