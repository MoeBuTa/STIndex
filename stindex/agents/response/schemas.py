"""
Response models and schemas for spatiotemporal extraction agents.

These Pydantic models structure the observe-reason-act flow and define entity schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# ENTITY TYPE ENUMS
# ============================================================================

class TemporalType(str, Enum):
    """Types of temporal expressions."""

    DATE = "date"  # Specific date
    TIME = "time"  # Specific time
    DATETIME = "datetime"  # Date and time
    DURATION = "duration"  # Time duration
    INTERVAL = "interval"  # Time interval/range
    RELATIVE = "relative"  # Relative time expression


# ============================================================================
# ENTITY SCHEMAS (used by tools and utils)
# ============================================================================

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


# ============================================================================
# OBSERVATION MODELS (Observe phase)
# ============================================================================

class ExtractionObservation(BaseModel):
    """
    What the extractor agent observes from the environment.

    This is the output of the OBSERVE phase.
    """
    # Input
    original_text: str = Field(description="Original input text")
    cleaned_text: str = Field(description="Cleaned and normalized text")

    # Document-level context
    temporal_context: Dict = Field(
        default_factory=dict,
        description="Document-level temporal context (years, relative dates)"
    )
    spatial_context: Dict = Field(
        default_factory=dict,
        description="Document-level spatial context (regions, countries)"
    )

    # Metadata
    language: str = Field(default="en", description="Detected language")
    char_count: int = Field(description="Character count")
    word_count: int = Field(description="Word count")


# ============================================================================
# REASONING MODELS (Reason phase - LLM output)
# ============================================================================

class TemporalMention(BaseModel):
    """Temporal mention extracted by LLM."""
    text: str = Field(description="Exact temporal expression found")
    context: str = Field(description="Surrounding context (up to 20 words)")


class SpatialMention(BaseModel):
    """Spatial mention extracted by LLM."""
    text: str = Field(description="Exact location name found")
    context: str = Field(description="Surrounding context (up to 20 words)")
    type: str = Field(
        default="other",
        description="Location type: country, city, region, landmark, address, feature, other"
    )


class ExtractionReasoning(BaseModel):
    """
    LLM reasoning output from the extractor agent.

    This is the output of the REASON phase.
    """
    # LLM extracted mentions
    temporal_mentions: List[TemporalMention] = Field(
        default_factory=list,
        description="Temporal expressions extracted by LLM"
    )
    spatial_mentions: List[SpatialMention] = Field(
        default_factory=list,
        description="Spatial mentions extracted by LLM"
    )

    # Raw LLM output (for debugging/tracing)
    raw_output: str = Field(default="", description="Raw LLM response")

    # Success flag
    success: bool = Field(default=True, description="Whether extraction succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# ACTION MODELS (Act phase - final output)
# ============================================================================

class ExtractionActionResponse(BaseModel):
    """
    Final response from the extractor agent.

    This is the output of the ACT phase, containing:
    - Normalized temporal entities
    - Geocoded spatial entities
    """
    # Structured entities (after tool calling postprocessing)
    temporal_entities: List[Dict] = Field(
        default_factory=list,
        description="Temporal entities with ISO 8601 normalization"
    )
    spatial_entities: List[Dict] = Field(
        default_factory=list,
        description="Spatial entities with lat/lon coordinates"
    )

    # Metadata
    success: bool = Field(default=True, description="Whether extraction succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    processing_time: float = Field(default=0.0, description="Processing time in seconds")

    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata (config, counts, etc.)"
    )


# ============================================================================
# TOOL CALL MODELS
# ============================================================================

class ToolCallRequest(BaseModel):
    """Request to call a tool."""
    tool_name: str = Field(description="Name of the tool to call")
    parameters: Dict = Field(default_factory=dict, description="Tool parameters")


class ToolCallResponse(BaseModel):
    """Response from a tool call."""
    success: bool = Field(description="Whether tool call succeeded")
    result: Optional[Dict] = Field(default=None, description="Tool result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
