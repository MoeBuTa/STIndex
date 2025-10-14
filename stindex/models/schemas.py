"""
Pydantic models for spatiotemporal index extraction.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TemporalType(str, Enum):
    """Types of temporal expressions."""

    DATE = "date"  # Specific date
    TIME = "time"  # Specific time
    DATETIME = "datetime"  # Date and time
    DURATION = "duration"  # Time duration
    INTERVAL = "interval"  # Time interval/range
    RELATIVE = "relative"  # Relative time expression


class TemporalEntity(BaseModel):
    """Represents an extracted temporal entity."""

    text: str = Field(..., description="Original temporal expression in text")
    normalized: str = Field(..., description="Normalized ISO 8601 format")
    temporal_type: TemporalType = Field(..., description="Type of temporal expression")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    start_char: Optional[int] = Field(None, description="Character offset start position")
    end_char: Optional[int] = Field(None, description="Character offset end position")

    # Additional fields for intervals
    start_date: Optional[str] = Field(None, description="Start date for intervals")
    end_date: Optional[str] = Field(None, description="End date for intervals")

    @field_validator("normalized")
    @classmethod
    def validate_normalized_format(cls, v: str) -> str:
        """Validate that normalized time follows ISO 8601 format."""
        try:
            # Try parsing as datetime
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            # Allow duration formats like P1D, P1M, etc.
            if v.startswith("P"):
                return v
            # Allow interval formats like 2023-01-17/2023-01-19
            if "/" in v:
                parts = v.split("/")
                if len(parts) == 2:
                    try:
                        # Validate both parts are valid ISO dates
                        datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                        datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
                        return v
                    except ValueError:
                        pass
            raise ValueError(f"Invalid ISO 8601 format: {v}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "March 15, 2022",
                "normalized": "2022-03-15",
                "temporal_type": "date",
                "confidence": 0.95,
                "start_char": 3,
                "end_char": 18,
            }
        }


class SpatialEntity(BaseModel):
    """Represents an extracted spatial entity."""

    text: str = Field(..., description="Original location mention in text")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    location_type: str = Field(
        default="LOCATION", description="Type of location (GPE, LOC, FAC, etc.)"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    start_char: Optional[int] = Field(None, description="Character offset start position")
    end_char: Optional[int] = Field(None, description="Character offset end position")

    # Geocoding details
    address: Optional[str] = Field(None, description="Full formatted address")
    country: Optional[str] = Field(None, description="Country name")
    admin_area: Optional[str] = Field(None, description="Administrative area (state/province)")
    locality: Optional[str] = Field(None, description="City or locality")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Broome, Western Australia",
                "latitude": -17.9614,
                "longitude": 122.2359,
                "location_type": "GPE",
                "confidence": 0.98,
                "address": "Broome WA 6725, Australia",
                "country": "Australia",
                "admin_area": "Western Australia",
                "locality": "Broome",
            }
        }


class SpatioTemporalResult(BaseModel):
    """Complete result of spatiotemporal extraction."""

    text: str = Field(..., description="Original input text")
    temporal_entities: List[TemporalEntity] = Field(
        default_factory=list, description="Extracted temporal entities"
    )
    spatial_entities: List[SpatialEntity] = Field(
        default_factory=list, description="Extracted spatial entities"
    )
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

    @property
    def temporal_count(self) -> int:
        """Number of temporal entities extracted."""
        return len(self.temporal_entities)

    @property
    def spatial_count(self) -> int:
        """Number of spatial entities extracted."""
        return len(self.spatial_entities)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "temporal": [t.model_dump() for t in self.temporal_entities],
            "spatial": [s.model_dump() for s in self.spatial_entities],
            "counts": {"temporal": self.temporal_count, "spatial": self.spatial_count},
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }

    class Config:
        json_schema_extra = {
            "example": {
                "text": "On March 15, 2022, a cyclone hit Broome, Western Australia.",
                "temporal_entities": [
                    {
                        "text": "March 15, 2022",
                        "normalized": "2022-03-15",
                        "temporal_type": "date",
                        "confidence": 0.95,
                    }
                ],
                "spatial_entities": [
                    {
                        "text": "Broome, Western Australia",
                        "latitude": -17.9614,
                        "longitude": 122.2359,
                        "location_type": "GPE",
                        "confidence": 0.98,
                    }
                ],
                "processing_time": 1.23,
            }
        }


class ExtractionConfig(BaseModel):
    """Configuration for extraction pipeline."""

    # LLM settings
    llm_provider: str = Field(default="local", description="LLM provider (openai/anthropic/local)")
    model_name: str = Field(default="Qwen/Qwen3-8B", description="Model name")
    model_path: Optional[str] = Field(None, description="Local model path (for local provider)")
    device: str = Field(default="auto", description="Device for local models (cuda/cpu/auto)")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="LLM temperature")

    # Temporal extraction settings
    enable_temporal: bool = Field(default=True, description="Enable temporal extraction")
    reference_date: Optional[str] = Field(None, description="Reference date for relative times")

    # Spatial extraction settings
    enable_spatial: bool = Field(default=True, description="Enable spatial extraction")
    geocoder: str = Field(default="nominatim", description="Geocoding provider")
    user_agent: str = Field(default="stindex", description="User agent for geocoding")

    # Performance settings
    enable_cache: bool = Field(default=True, description="Enable caching")
    rate_limit_calls: int = Field(default=1, description="Rate limit calls per period")
    rate_limit_period: float = Field(default=1.0, description="Rate limit period in seconds")

    # Output settings
    include_offsets: bool = Field(default=True, description="Include character offsets")
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "llm_provider": "local",
                "model_name": "Qwen/Qwen3-8B",
                "temperature": 0.0,
                "geocoder": "nominatim",
                "user_agent": "stindex_v0.2.0",
                "enable_cache": True,
            }
        }
