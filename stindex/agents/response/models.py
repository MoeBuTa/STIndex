"""
Response models for spatiotemporal extraction agents.

These Pydantic models structure the observe-reason-act flow.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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
