"""
Main STIndex extractor combining temporal and spatial extraction.
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv

from stindex.extractors.spatial import SpatialExtractor
from stindex.extractors.temporal import TemporalExtractor
from stindex.models.schemas import ExtractionConfig, SpatioTemporalResult

# Load environment variables
load_dotenv()


class STIndexExtractor:
    """
    Main extractor for spatiotemporal indices.

    Combines temporal and spatial extraction into a unified pipeline.
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

        # Initialize extractors
        self.temporal_extractor = None
        self.spatial_extractor = None

        if config.enable_temporal:
            self.temporal_extractor = TemporalExtractor(
                llm_provider=config.llm_provider,
                model_name=config.model_name,
                model_path=getattr(config, 'model_path', None),
                temperature=config.temperature,
                reference_date=config.reference_date,
                device=getattr(config, 'device', 'auto'),
            )

        if config.enable_spatial:
            self.spatial_extractor = SpatialExtractor(
                spacy_model="en_core_web_sm",
                geocoder_provider=config.geocoder,
                user_agent=config.user_agent,
                rate_limit=config.rate_limit_period,  # Updated parameter name
                enable_geocoding=True,
                enable_cache=config.enable_cache,
            )

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
        start_time = time.time()

        # Extract temporal entities
        temporal_entities = []
        if self.config.enable_temporal and self.temporal_extractor:
            temporal_entities = self.temporal_extractor.extract(text)

            # Filter by confidence
            temporal_entities = [
                e for e in temporal_entities if e.confidence >= self.config.min_confidence
            ]

        # Extract spatial entities
        spatial_entities = []
        if self.config.enable_spatial and self.spatial_extractor:
            spatial_entities = self.spatial_extractor.extract(text)

            # Filter by confidence
            spatial_entities = [
                e for e in spatial_entities if e.confidence >= self.config.min_confidence
            ]

        processing_time = time.time() - start_time

        # Create result
        result = SpatioTemporalResult(
            text=text,
            temporal_entities=temporal_entities,
            spatial_entities=spatial_entities,
            processing_time=processing_time,
            metadata={
                "config": self.config.model_dump(),
                "extractors_used": {
                    "temporal": self.config.enable_temporal,
                    "spatial": self.config.enable_spatial,
                },
            },
        )

        return result

    def extract_batch(self, texts: List[str]) -> List[SpatioTemporalResult]:
        """
        Extract spatiotemporal indices from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of SpatioTemporalResult objects
        """
        results = []

        for text in texts:
            result = self.extract(text)
            results.append(result)

        return results

    def extract_from_file(self, file_path: str) -> SpatioTemporalResult:
        """
        Extract spatiotemporal indices from a text file.

        Args:
            file_path: Path to text file

        Returns:
            SpatioTemporalResult with extracted entities
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.extract(text)

    def extract_temporal_only(self, text: str) -> List:
        """
        Extract only temporal entities.

        Args:
            text: Input text

        Returns:
            List of TemporalEntity objects
        """
        if not self.temporal_extractor:
            raise ValueError("Temporal extraction is not enabled")

        return self.temporal_extractor.extract(text)

    def extract_spatial_only(self, text: str) -> List:
        """
        Extract only spatial entities.

        Args:
            text: Input text

        Returns:
            List of SpatialEntity objects
        """
        if not self.spatial_extractor:
            raise ValueError("Spatial extraction is not enabled")

        return self.spatial_extractor.extract(text)

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        config_dict = self.config.model_dump()
        config_dict.update(kwargs)
        self.config = ExtractionConfig(**config_dict)

        # Reinitialize extractors if needed
        if "llm_provider" in kwargs or "model_name" in kwargs:
            if self.config.enable_temporal:
                self.temporal_extractor = TemporalExtractor(
                    llm_provider=self.config.llm_provider,
                    model_name=self.config.model_name,
                    model_path=getattr(self.config, 'model_path', None),
                    temperature=self.config.temperature,
                    reference_date=self.config.reference_date,
                    device=getattr(self.config, 'device', 'auto'),
                )

    def clear_cache(self):
        """Clear all caches."""
        if self.spatial_extractor and self.spatial_extractor.geocoder:
            self.spatial_extractor.geocoder.clear_cache()
