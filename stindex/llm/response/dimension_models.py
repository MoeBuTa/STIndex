"""
Generic dimension models for multi-dimensional extraction.

Provides base models for different dimension types:
- NormalizedDimension: For dimensions requiring normalization (e.g., temporal → ISO 8601)
- GeocodedDimension: For spatial dimensions requiring geocoding
- CategoricalDimension: For dimensions with predefined categories
- StructuredDimension: For dimensions with multiple structured fields

These models enable STIndex to extract any configurable dimension from YAML config.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# GENERIC DIMENSION BASE MODELS
# ============================================================================

class DimensionType(str, Enum):
    """Types of dimensions that can be extracted."""
    NORMALIZED = "normalized"      # Requires normalization (e.g., temporal)
    GEOCODED = "geocoded"          # Requires geocoding (e.g., spatial)
    CATEGORICAL = "categorical"    # From predefined categories
    STRUCTURED = "structured"      # Multiple structured fields
    FREE_TEXT = "free_text"        # Free-form text


class BaseDimensionMention(BaseModel):
    """Base class for all dimension mentions."""
    text: str = Field(description="Original text from document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    hierarchy: Optional[Dict[str, str]] = Field(
        default=None,
        description="Hierarchy level values (level_name → value)"
    )


class NormalizedDimensionMention(BaseDimensionMention):
    """
    Dimension requiring normalization to a standard format.

    Example: Temporal expressions → ISO 8601
    """
    normalized: str = Field(description="Normalized representation")
    normalization_type: Optional[str] = Field(
        default=None,
        description="Type of normalized value (e.g., 'date', 'datetime', 'duration')"
    )


class GeocodedDimensionMention(BaseDimensionMention):
    """
    Dimension requiring geocoding to coordinates.

    Example: Spatial locations → lat/lon
    """
    location_type: Optional[str] = Field(default=None)
    parent_region: Optional[str] = Field(
        default=None,
        description="Parent region for disambiguation"
    )


class CategoricalDimensionMention(BaseDimensionMention):
    """
    Dimension with predefined categories.

    Example: event_type → ["exposure_site", "case_report", ...]
    """
    category: str = Field(description="Category label")
    category_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in category assignment"
    )


class StructuredDimensionMention(BaseDimensionMention):
    """
    Dimension with multiple structured fields.

    Example: patient_demographics → {age_group, vaccination_status, ...}
    """
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured field values"
    )


class FreeTextDimensionMention(BaseDimensionMention):
    """
    Free-form text dimension.

    Example: event_description, symptoms, etc.
    """
    pass


# ============================================================================
# DIMENSION METADATA
# ============================================================================

class DimensionMetadata(BaseModel):
    """Metadata about a dimension definition."""
    name: str
    enabled: bool = True
    description: str
    extraction_type: DimensionType
    schema_type: str
    fields: List[Dict[str, Any]] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# DYNAMIC EXTRACTION RESULT
# ============================================================================

class DimensionalExtractionResult(BaseModel):
    """
    Dynamic extraction result that can hold any combination of dimensions.

    The 'dimensions' dict maps dimension names to lists of mentions:
    {
        "temporal": [TemporalMention(...), ...],
        "spatial": [SpatialMention(...), ...],
        "event_type": [CategoricalDimensionMention(...), ...],
        ...
    }
    """
    model_config = ConfigDict(extra="allow")  # Allow extra fields

    dimensions: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Map of dimension names to lists of extracted mentions"
    )

    # For backward compatibility with existing code
    temporal_mentions: Optional[List[Dict[str, Any]]] = Field(default=None)
    spatial_mentions: Optional[List[Dict[str, Any]]] = Field(default=None)

    def __init__(self, **data):
        """Initialize with backward compatibility."""
        super().__init__(**data)

        # If using new format, populate backward-compatible fields
        if self.dimensions and not self.temporal_mentions:
            self.temporal_mentions = self.dimensions.get("temporal", [])
        if self.dimensions and not self.spatial_mentions:
            self.spatial_mentions = self.dimensions.get("spatial", [])


# ============================================================================
# PROCESSED DIMENSION ENTITIES (After post-processing)
# ============================================================================

class DimensionEntity(BaseModel):
    """Base class for processed dimension entities."""
    text: str
    dimension_name: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    hierarchy: Optional[Dict[str, str]] = Field(
        default=None,
        description="Hierarchy level values (level_name → value)"
    )


class NormalizedDimensionEntity(DimensionEntity):
    """Processed normalized dimension (e.g., temporal with ISO 8601)."""
    normalized: str
    normalization_type: Optional[str] = None


class GeocodedDimensionEntity(DimensionEntity):
    """Processed geocoded dimension (e.g., spatial with coordinates)."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    location_type: Optional[str] = None
    address: Optional[str] = None


class CategoricalDimensionEntity(DimensionEntity):
    """Processed categorical dimension."""
    category: str
    category_confidence: Optional[float] = None


class StructuredDimensionEntity(DimensionEntity):
    """Processed structured dimension."""
    fields: Dict[str, Any] = Field(default_factory=dict)


class FreeTextDimensionEntity(DimensionEntity):
    """Processed free-text dimension."""
    pass


# ============================================================================
# FINAL MULTI-DIMENSIONAL RESULT
# ============================================================================

class MultiDimensionalResult(BaseModel):
    """
    Complete multi-dimensional extraction result.

    Contains both raw extractions and processed entities for all dimensions.
    """
    input_text: str = Field(default="", description="Original input text")

    # Multi-dimensional entities (post-processed)
    entities: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Map of dimension names to processed entities"
    )

    # Backward compatibility
    temporal_entities: List[Dict[str, Any]] = Field(default_factory=list)
    spatial_entities: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    success: bool = True
    error: Optional[str] = None
    processing_time: float = 0.0

    # Document metadata (for context)
    document_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata (publication_date, source_location, etc.)"
    )

    # Extraction configuration
    extraction_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM configuration and dimension config used"
    )

    # Dimension metadata
    dimension_configs: Optional[Dict[str, DimensionMetadata]] = Field(
        default=None,
        description="Metadata about extracted dimensions"
    )


# ============================================================================
# HELPER FUNCTIONS FOR DYNAMIC MODEL CREATION
# ============================================================================

def create_dimension_mention_model(
    dimension_name: str,
    dimension_config: Dict[str, Any]
) -> type[BaseModel]:
    """
    Create a Pydantic model for a dimension based on config.

    Args:
        dimension_name: Name of the dimension
        dimension_config: Dimension configuration dict

    Returns:
        Pydantic model class
    """
    extraction_type = dimension_config.get("extraction_type", "categorical")
    fields_config = dimension_config.get("fields", [])

    # Choose base class
    if extraction_type == "normalized":
        base_class = NormalizedDimensionMention
    elif extraction_type == "geocoded":
        base_class = GeocodedDimensionMention
    elif extraction_type == "categorical":
        base_class = CategoricalDimensionMention
    elif extraction_type == "structured":
        base_class = StructuredDimensionMention
    else:
        base_class = FreeTextDimensionMention

    # For now, return base class (could dynamically create with custom fields)
    return base_class


def create_extraction_result_model(
    dimension_configs: Dict[str, Dict[str, Any]]
) -> type[BaseModel]:
    """
    Create a dynamic ExtractionResult model with fields for all enabled dimensions.

    Args:
        dimension_configs: Dict of dimension name → dimension config

    Returns:
        Pydantic model class with fields for each dimension
    """
    # Build field definitions
    field_defs = {}

    for dim_name, dim_config in dimension_configs.items():
        if not dim_config.get("enabled", True):
            continue

        # Create field with list of mentions
        field_defs[dim_name] = (
            List[Dict[str, Any]],
            Field(
                default_factory=list,
                description=dim_config.get("description", f"Extracted {dim_name} mentions")
            )
        )

    # Create dynamic model
    DynamicExtractionResult = type(
        "DynamicExtractionResult",
        (BaseModel,),
        {
            "__annotations__": field_defs,
            **field_defs
        }
    )

    return DynamicExtractionResult


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def convert_to_dimension_entity(
    mention: Dict[str, Any],
    dimension_name: str,
    dimension_type: DimensionType
) -> DimensionEntity:
    """
    Convert a raw mention dict to a typed DimensionEntity.

    Args:
        mention: Raw mention dict from LLM
        dimension_name: Name of the dimension
        dimension_type: Type of dimension

    Returns:
        Typed DimensionEntity
    """
    base_data = {
        "text": mention.get("text", ""),
        "dimension_name": dimension_name,
        "confidence": mention.get("confidence", 1.0)
    }

    if dimension_type == DimensionType.NORMALIZED:
        return NormalizedDimensionEntity(
            **base_data,
            normalized=mention.get("normalized", ""),
            normalization_type=mention.get("normalization_type")
        )
    elif dimension_type == DimensionType.GEOCODED:
        # Will be populated after geocoding
        return None
    elif dimension_type == DimensionType.CATEGORICAL:
        return CategoricalDimensionEntity(
            **base_data,
            category=mention.get("category", "unknown"),
            category_confidence=mention.get("category_confidence")
        )
    elif dimension_type == DimensionType.STRUCTURED:
        return StructuredDimensionEntity(
            **base_data,
            fields=mention.get("fields", {})
        )
    else:
        return FreeTextDimensionEntity(**base_data)
