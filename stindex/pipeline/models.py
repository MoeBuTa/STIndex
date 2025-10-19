"""
Pipeline models for STIndex.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Result from extraction pipeline."""

    text: str = Field(description="Original input text")
    temporal_entities: List[Dict] = Field(
        default_factory=list,
        description="Extracted temporal entities"
    )
    spatial_entities: List[Dict] = Field(
        default_factory=list,
        description="Extracted spatial entities"
    )

    success: bool = Field(default=True, description="Whether extraction succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    processing_time: float = Field(default=0.0, description="Processing time in seconds")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class BatchExtractionResult(BaseModel):
    """Result from batch extraction pipeline."""

    results: List[ExtractionResult] = Field(
        default_factory=list,
        description="List of extraction results"
    )

    total_count: int = Field(description="Total number of texts processed")
    success_count: int = Field(description="Number of successful extractions")
    failure_count: int = Field(description="Number of failed extractions")

    total_processing_time: float = Field(
        default=0.0,
        description="Total processing time in seconds"
    )
