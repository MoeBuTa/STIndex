"""
Simplified Pydantic models for spatiotemporal extraction using Instructor.

Clean, type-safe schemas for structured LLM outputs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# LLM RESPONSE MODELS
# ============================================================================

class TokenUsage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        default=0, description="Number of tokens in the completion"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")


class LLMResponse(BaseModel):
    """Standardized response model for all LLM providers."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model name used for the response")
    input: List[Dict[str, str]] = Field(
        ..., description="Input messages sent to the model"
    )
    status: str = Field(..., description="Response status (processed, error)")
    response: Optional[Dict[str, Any]] = Field(
        None, description="Parsed response content"
    )
    usage: Optional[TokenUsage] = Field(None, description="Token usage statistics")
    error_msg: str = Field(default="", description="Error message if status is error")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp when response was created",
    )
    content: str = Field(default="", description="Raw text content of the response")
    success: bool = Field(
        default=False, description="Whether the request was successful"
    )


# ============================================================================
# ENUMS
# ============================================================================

class TemporalType(str, Enum):
    """Types of temporal expressions."""
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    DURATION = "duration"
    INTERVAL = "interval"


class LocationType(str, Enum):
    """Types of spatial locations."""
    COUNTRY = "country"
    CITY = "city"
    REGION = "region"
    LANDMARK = "landmark"
    FEATURE = "feature"
    OTHER = "other"


# ============================================================================
# EXTRACTION MODELS (LLM Output)
# ============================================================================

class TemporalMention(BaseModel):
    """Temporal mention extracted and normalized by LLM."""

    text: str = Field(description="Original temporal expression from text")
    normalized: str = Field(description="ISO 8601 normalized format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
    temporal_type: TemporalType = Field(description="Type of temporal expression")

    @field_validator("normalized")
    @classmethod
    def validate_iso8601(cls, v: str) -> str:
        """Validate ISO 8601 format."""
        # Allow durations (P1D, P1M, etc.)
        if v.startswith("P"):
            return v

        # Allow intervals (date/date)
        if "/" in v:
            parts = v.split("/")
            if len(parts) == 2:
                try:
                    datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
                    datetime.fromisoformat(parts[1].replace("Z", "+00:00"))
                    return v
                except ValueError:
                    raise ValueError(f"Invalid ISO 8601 interval: {v}")

        # Regular datetime/date
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 format: {v}. Expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")


class SpatialMention(BaseModel):
    """Spatial mention extracted and disambiguated by LLM."""

    text: str = Field(description="Original location name from text")
    location_type: LocationType = Field(description="Type of location")
    parent_region: Optional[str] = Field(
        default=None,
        description="Parent region for disambiguation (e.g., 'Western Australia' for 'Broome')"
    )


class ExtractionResult(BaseModel):
    """Complete extraction result from LLM with CoT reasoning."""

    temporal_mentions: List[TemporalMention] = Field(
        default_factory=list,
        description="All temporal expressions found and normalized to ISO 8601"
    )
    spatial_mentions: List[SpatialMention] = Field(
        default_factory=list,
        description="All spatial mentions found with parent regions for disambiguation"
    )


# ============================================================================
# FINAL OUTPUT MODELS (After Geocoding)
# ============================================================================

class TemporalEntity(BaseModel):
    """Final temporal entity with metadata."""

    text: str
    normalized: str
    temporal_type: TemporalType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SpatialEntity(BaseModel):
    """Final spatial entity with geocoded coordinates."""

    text: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_type: LocationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    address: Optional[str] = None


class ExtractionConfig(BaseModel):
    """LLM configuration used for extraction."""

    llm_provider: str = Field(description="LLM provider used (openai, anthropic, hf)")
    model_name: str = Field(description="Model name/ID")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    raw_llm_output: Optional[str] = Field(
        default=None,
        description="Raw output from LLM before parsing (for debugging)"
    )


class SpatioTemporalResult(BaseModel):
    """Complete spatiotemporal extraction result."""

    input_text: str = Field(default="", description="Original input text that was processed")
    temporal_entities: List[TemporalEntity] = Field(default_factory=list)
    spatial_entities: List[SpatialEntity] = Field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    processing_time: float = 0.0
    extraction_config: Optional[ExtractionConfig] = Field(
        default=None,
        description="LLM configuration and raw output used for this extraction"
    )
