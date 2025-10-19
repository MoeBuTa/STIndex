"""
Main STIndex extractor - facade for the extraction pipeline.

This is the main entry point for spatiotemporal extraction.
Uses the new agentic architecture with observe-reason-act pattern.
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv

from stindex.models.schemas import ExtractionConfig, SpatioTemporalResult, TemporalEntity, SpatialEntity
from stindex.pipeline import ExtractionPipeline

# Load environment variables
load_dotenv()


class STIndexExtractor:
    """
    Main extractor for spatiotemporal indices.

    This is a facade that wraps the new agentic ExtractionPipeline.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize STIndexExtractor.

        Args:
            config: Configuration object. If None, uses defaults from environment.
        """
        # Load configuration
        if config is None:
            config = self._load_config_from_env()

        self.config = config

        # Initialize pipeline with config dict
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
        self.pipeline = ExtractionPipeline(config_dict)

    def _load_config_from_env(self) -> ExtractionConfig:
        """Load configuration from environment variables."""
        config_dict = {
            "llm_provider": os.getenv("STINDEX_LLM_PROVIDER", "local"),
            "model_name": os.getenv("STINDEX_MODEL_NAME", "Qwen/Qwen3-8B"),
            "temperature": float(os.getenv("STINDEX_TEMPERATURE", "0.0")),
            "reference_date": os.getenv("STINDEX_REFERENCE_DATE"),
            "enable_temporal": os.getenv("STINDEX_ENABLE_TEMPORAL", "true").lower() == "true",
            "enable_spatial": os.getenv("STINDEX_ENABLE_SPATIAL", "true").lower() == "true",
            "geocoder": os.getenv("STINDEX_GEOCODER", "nominatim"),
            "user_agent": os.getenv("STINDEX_USER_AGENT", "stindex"),
            "enable_cache": os.getenv("STINDEX_ENABLE_CACHE", "true").lower() == "true",
            "rate_limit_calls": int(os.getenv("STINDEX_RATE_LIMIT_CALLS", "1")),
            "rate_limit_period": float(os.getenv("STINDEX_RATE_LIMIT_PERIOD", "1.0")),
            "include_offsets": os.getenv("STINDEX_INCLUDE_OFFSETS", "true").lower() == "true",
            "min_confidence": float(os.getenv("STINDEX_MIN_CONFIDENCE", "0.5")),
        }

        # Add local model specific config
        if config_dict["llm_provider"] == "local":
            config_dict["model_path"] = os.getenv("STINDEX_MODEL_PATH")
            config_dict["device"] = os.getenv("STINDEX_DEVICE", "auto")

        return ExtractionConfig(**config_dict)

    def extract(self, text: str) -> SpatioTemporalResult:
        """
        Extract spatiotemporal indices from text.

        Args:
            text: Input text

        Returns:
            SpatioTemporalResult with extracted entities
        """
        # Run pipeline
        result = self.pipeline.extract(text)

        # Convert to legacy format for compatibility
        temporal_entities = [
            TemporalEntity(**e) for e in result.temporal_entities
        ]

        spatial_entities = [
            SpatialEntity(**e) for e in result.spatial_entities
        ]

        return SpatioTemporalResult(
            text=text,
            temporal_entities=temporal_entities,
            spatial_entities=spatial_entities,
            processing_time=result.processing_time,
            metadata=result.metadata,
        )

    def extract_batch(self, texts: List[str]) -> List[SpatioTemporalResult]:
        """
        Extract spatiotemporal indices from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of SpatioTemporalResult objects
        """
        batch_result = self.pipeline.extract_batch(texts)

        results = []
        for r in batch_result.results:
            temporal_entities = [TemporalEntity(**e) for e in r.temporal_entities]
            spatial_entities = [SpatialEntity(**e) for e in r.spatial_entities]

            results.append(
                SpatioTemporalResult(
                    text=r.text,
                    temporal_entities=temporal_entities,
                    spatial_entities=spatial_entities,
                    processing_time=r.processing_time,
                    metadata=r.metadata,
                )
            )

        return results

    def extract_from_file(self, file_path: str) -> SpatioTemporalResult:
        """
        Extract spatiotemporal indices from a text file.

        Args:
            file_path: Path to text file

        Returns:
            SpatioTemporalResult with extracted entities
        """
        result = self.pipeline.extract_from_file(file_path)

        temporal_entities = [TemporalEntity(**e) for e in result.temporal_entities]
        spatial_entities = [SpatialEntity(**e) for e in result.spatial_entities]

        return SpatioTemporalResult(
            text=result.text,
            temporal_entities=temporal_entities,
            spatial_entities=spatial_entities,
            processing_time=result.processing_time,
            metadata=result.metadata,
        )

    def extract_temporal_only(self, text: str) -> List[TemporalEntity]:
        """
        Extract only temporal entities.

        Args:
            text: Input text

        Returns:
            List of TemporalEntity objects
        """
        result = self.extract(text)
        return result.temporal_entities

    def extract_spatial_only(self, text: str) -> List[SpatialEntity]:
        """
        Extract only spatial entities.

        Args:
            text: Input text

        Returns:
            List of SpatialEntity objects
        """
        result = self.extract(text)
        return result.spatial_entities

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        config_dict = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict()
        config_dict.update(kwargs)
        self.config = ExtractionConfig(**config_dict)

        # Reinitialize pipeline
        self.pipeline = ExtractionPipeline(config_dict)

    def clear_cache(self):
        """Clear all caches."""
        # Access geocoder through pipeline -> agent -> tool_registry
        if hasattr(self.pipeline.agent, 'tool_registry'):
            if hasattr(self.pipeline.agent.tool_registry, 'geocoder'):
                self.pipeline.agent.tool_registry.geocoder.clear_cache()
