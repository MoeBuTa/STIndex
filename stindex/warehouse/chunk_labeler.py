"""
Dimensional chunk labeling for data warehouse integration.

Converts multi-dimensional extraction results into hierarchical labels and paths
for warehouse fact table. Supports:
- Temporal hierarchy (Year → Quarter → Month → Day)
- Spatial hierarchy (Continent → Country → State → Region → City)
- Event hierarchy (Category → Type → Subtype)
- Entity hierarchy (Category → Type)
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from stindex.llm.response.dimension_models import (
    MultiDimensionalResult,
)


@dataclass
class ChunkDimensionalLabels:
    """
    Dimensional labels for a document chunk.

    Based on extracted multi-dimensional entities, generates hierarchical labels
    and paths for each dimension to enable efficient warehouse queries.

    Attributes:
        doc_id: Unique identifier for document/chunk (string UUID from extraction)
        chunk_text: Original chunk text
        chunk_hash: SHA-256 hash of chunk text

        # Dimensional IDs (for warehouse foreign keys)
        temporal_dim_ids: List of temporal dimension IDs
        spatial_dim_ids: List of spatial dimension IDs
        event_dim_ids: List of event dimension IDs
        entity_dim_ids: List of entity dimension IDs

        # Hierarchical labels for each dimension (array for filtering)
        temporal_labels: List[str] - ["2022", "2022-Q1", "2022-03", "2022-03-15"]
        spatial_labels: List[str] - ["Australia", "Western Australia", "Kimberley", "Broome"]
        event_labels: List[str] - ["natural_disaster", "storm", "cyclone"]
        entity_labels: List[str] - ["organization", "government_agency", "WHO"]

        # Hierarchy paths (for drill-down queries)
        temporal_path: str - "2022 > Q1 > March > 2022-03-15"
        spatial_path: str - "Australia > Western Australia > Kimberley > Broome"
        event_path: str - "natural_disaster > storm > cyclone"
        entity_path: str - "organization > government_agency > WHO"

        # Geographic coordinates (for PostGIS)
        latitude: Optional[float]
        longitude: Optional[float]

        # Confidence scores
        confidence_score: float - Overall confidence (average across dimensions)
        dimension_confidences: Dict[str, float] - Per-dimension confidence

        # Entity counts
        entity_counts: Dict[str, int] - Count per dimension

        # Document metadata
        document_id: Optional[int] - Foreign key to dim_document
        chunk_index: int - Position in document

        # Section hierarchy (from preprocessing)
        section_hierarchy: Optional[str] - "Report > Section 3 > Subsection 3.2"
    """

    # Chunk identification
    doc_id: Optional[str] = None
    chunk_text: str = ""
    chunk_hash: str = ""

    # Dimensional IDs (for warehouse foreign keys)
    temporal_dim_ids: List[int] = field(default_factory=list)
    spatial_dim_ids: List[int] = field(default_factory=list)
    event_dim_ids: List[int] = field(default_factory=list)
    entity_dim_ids: List[int] = field(default_factory=list)

    # Hierarchical labels (arrays for GIN index filtering)
    temporal_labels: List[str] = field(default_factory=list)
    spatial_labels: List[str] = field(default_factory=list)
    event_labels: List[str] = field(default_factory=list)
    entity_labels: List[str] = field(default_factory=list)

    # Hierarchy paths (human-readable)
    temporal_path: Optional[str] = None
    spatial_path: Optional[str] = None
    event_path: Optional[str] = None
    entity_path: Optional[str] = None

    # Geographic coordinates (for spatial queries)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Confidence scores
    confidence_score: float = 1.0
    dimension_confidences: Dict[str, float] = field(default_factory=dict)

    # Entity counts per dimension
    entity_counts: Dict[str, int] = field(default_factory=dict)

    # Document metadata
    document_id: Optional[int] = None
    chunk_index: int = 0

    # Section hierarchy
    section_hierarchy: Optional[str] = None

    # Generic dimensions (for discovered dimensions beyond the 4 hardcoded ones)
    # NEW: Store per-entity hierarchies as list of lists for RAG-optimized output
    generic_dimension_entities: Dict[str, List[List[str]]] = field(default_factory=dict)
    # Legacy: flat labels for backward compatibility
    generic_dimension_labels: Dict[str, List[str]] = field(default_factory=dict)
    generic_dimension_paths: Dict[str, str] = field(default_factory=dict)
    generic_dimension_confidences: Dict[str, float] = field(default_factory=dict)

    # Enabled dimensions (for filtering output fields)
    enabled_dimensions: Optional[set] = field(default=None)

    def compute_chunk_hash(self) -> str:
        """Compute SHA-256 hash of chunk text."""
        return hashlib.sha256(self.chunk_text.encode('utf-8')).hexdigest()

    def compute_confidence_score(self) -> float:
        """Compute average confidence across all dimensions."""
        if not self.dimension_confidences:
            return 1.0

        return sum(self.dimension_confidences.values()) / len(self.dimension_confidences)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert labels to RAG-optimized dictionary.

        Output format:
        {
            "temporal": [["2022", "2022-Q1", "2022-03"]],
            "spatial": [["Australia", "Western Australia", "Broome"]],
            "drug": [["aspirin", "NSAID"], ["morphine", "opioid"]],
            ...
        }

        Returns:
            Dict with dimensional labels only (no metadata, no null values)
        """
        result = {}

        # Temporal (only if has values)
        if self.temporal_labels:
            result["temporal"] = [self.temporal_labels]

        # Spatial (only if has values)
        if self.spatial_labels:
            result["spatial"] = [self.spatial_labels]

        # Generic dimensions (drug, procedure, diagnosis, etc.)
        for dim_name, entities in self.generic_dimension_entities.items():
            if entities:
                result[dim_name] = entities

        return result


class DimensionalChunkLabeler:
    """
    Generate hierarchical dimensional labels from extraction results.

    Converts extraction results into warehouse-compatible labels and paths.
    Handles temporal, spatial, event, entity hierarchies (hardcoded),
    plus arbitrary discovered dimensions with hierarchy support.
    """

    def __init__(
        self,
        dimension_config: Optional[Dict[str, Any]] = None,
        enabled_dimensions: Optional[set] = None
    ):
        """
        Initialize chunk labeler.

        Args:
            dimension_config: Optional dimension configuration dict
                            Can include discovered dimensions with hierarchies
                            Supports both formats:
                              - {"dimensions": {"drug": {...}}}  (from DimensionConfigLoader)
                              - {"drug": {"hierarchy": [...]}}   (flat discovered schema)
            enabled_dimensions: Set of enabled dimension names (e.g., {'temporal', 'spatial', 'vaccine'})
                              If None, defaults to all hardcoded dimensions for backward compatibility
        """
        self.dimension_config = dimension_config or {}
        self.dimensions = {}
        self.enabled_dimensions = enabled_dimensions or {'temporal', 'spatial', 'event', 'entity'}

        # Handle both config formats
        dims_dict = None
        if dimension_config:
            if "dimensions" in dimension_config:
                # Format: {"dimensions": {"drug": {...}}}
                dims_dict = dimension_config["dimensions"]
            else:
                # Flat format: {"drug": {"hierarchy": [...]}} (discovered schema)
                # Filter out metadata keys
                dims_dict = {k: v for k, v in dimension_config.items()
                           if isinstance(v, dict) and k not in ['config_path', 'base_config_path']}

        # Load dimensions
        if dims_dict:
            for dim_name, dim_dict in dims_dict.items():
                if isinstance(dim_dict, dict):
                    # Store hierarchy info for generic dimensions
                    # Don't require full DimensionConfig validation
                    self.dimensions[dim_name] = dim_dict

    def label_chunk(
        self,
        chunk_text: str,
        extraction_result: MultiDimensionalResult,
        chunk_index: int = 0,
        document_id: Optional[int] = None,
        section_hierarchy: Optional[str] = None,
    ) -> ChunkDimensionalLabels:
        """
        Generate dimensional labels for a chunk.

        Args:
            chunk_text: Original chunk text
            extraction_result: Multi-dimensional extraction result
            chunk_index: Position of chunk in document
            document_id: Foreign key to document dimension
            section_hierarchy: Document section path

        Returns:
            ChunkDimensionalLabels with all hierarchical labels and paths
        """
        labels = ChunkDimensionalLabels(
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            document_id=document_id,
            section_hierarchy=section_hierarchy,
            enabled_dimensions=self.enabled_dimensions,  # Pass enabled dimensions for output filtering
        )

        # Compute chunk hash
        labels.chunk_hash = labels.compute_chunk_hash()

        # Process hardcoded dimensions
        self._process_temporal_dimension(extraction_result, labels)
        self._process_spatial_dimension(extraction_result, labels)
        self._process_event_dimension(extraction_result, labels)
        self._process_entity_dimension(extraction_result, labels)

        # Process discovered dimensions (generic dimensions beyond the 4 hardcoded)
        if self.dimensions:
            for dim_name, dim_config in self.dimensions.items():
                # Skip hardcoded dimensions (already processed above)
                if dim_name not in ['temporal', 'spatial', 'event', 'entity']:
                    self._process_generic_dimension(dim_name, dim_config, extraction_result, labels)

        # Compute overall confidence score
        labels.confidence_score = labels.compute_confidence_score()

        logger.debug(
            f"Chunk labeled: {len(labels.temporal_labels)} temporal, "
            f"{len(labels.spatial_labels)} spatial, "
            f"{len(labels.event_labels)} event, "
            f"{len(labels.entity_labels)} entity labels, "
            f"{len(labels.generic_dimension_labels)} generic dimensions"
        )

        return labels

    def _process_generic_dimension(
        self,
        dim_name: str,
        dim_config: Any,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> None:
        """
        Process a discovered dimension with hierarchy support.

        Stores per-entity hierarchies as list of lists for RAG-optimized output.

        Args:
            dim_name: Dimension name (snake_case)
            dim_config: DimensionConfig object or dict with hierarchy
            extraction_result: Multi-dimensional extraction result
            labels: ChunkDimensionalLabels to populate
        """
        # Get entities for this dimension from extraction result
        dimension_entities = extraction_result.entities.get(dim_name, [])

        if not dimension_entities:
            return

        # Get hierarchy levels from config (handle both dict and DimensionConfig)
        hierarchy_levels = []
        if isinstance(dim_config, dict):
            # Dict format: {"hierarchy": ["drug_name", "drug_class"]}
            hierarchy_levels = dim_config.get('hierarchy', [])
        elif hasattr(dim_config, 'hierarchy') and dim_config.hierarchy:
            hierarchy_levels = dim_config.hierarchy
        elif hasattr(dim_config, 'fields') and dim_config.fields:
            # Fallback: extract hierarchy from fields (skip 'text' field)
            hierarchy_levels = [
                f.get("name")
                for f in dim_config.fields
                if f.get("name") != "text"
            ]

        # Collect per-entity hierarchies (list of lists)
        entity_hierarchies = []
        for entity in dimension_entities:
            entity_labels = []

            for level_name in hierarchy_levels:
                # Handle both string level names and dict {"level": "name"} format
                if isinstance(level_name, dict):
                    level_name = level_name.get("level", "")

                # Get value for this hierarchy level from entity
                value = entity.get(level_name)
                if value and isinstance(value, str):
                    entity_labels.append(value)

            if entity_labels:
                entity_hierarchies.append(entity_labels)

        # Store in generic_dimension_entities (new format for RAG)
        if entity_hierarchies:
            labels.generic_dimension_entities[dim_name] = entity_hierarchies

            # Also store flat labels for backward compatibility
            flat_labels = []
            for entity_labels in entity_hierarchies:
                for label in entity_labels:
                    if label not in flat_labels:
                        flat_labels.append(label)

            labels.generic_dimension_labels[dim_name] = flat_labels
            labels.generic_dimension_paths[dim_name] = " > ".join(flat_labels[:3])

            # Store average confidence
            confidences = [e.get("confidence", 1.0) for e in dimension_entities]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
            labels.generic_dimension_confidences[dim_name] = avg_confidence
            labels.dimension_confidences[dim_name] = avg_confidence

        # Store entity count
        labels.entity_counts[dim_name] = len(dimension_entities)

        logger.debug(
            f"Processed generic dimension '{dim_name}': "
            f"{len(entity_hierarchies)} entities, "
            f"{len(labels.generic_dimension_labels.get(dim_name, []))} unique labels"
        )

    def _process_temporal_dimension(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> None:
        """Process temporal dimension and generate hierarchy."""
        temporal_entities = extraction_result.temporal_entities

        if not temporal_entities:
            return

        # Get first temporal entity (primary temporal reference)
        entity = temporal_entities[0]

        # Generate temporal hierarchy
        normalized_value = entity.get("normalized", "")
        temporal_type = entity.get("normalization_type", "date")

        if normalized_value:
            hierarchy = self._create_temporal_hierarchy(normalized_value, temporal_type)
            labels.temporal_labels = hierarchy

            # Create hierarchy path
            labels.temporal_path = " > ".join(hierarchy)

            # Store confidence
            confidence = entity.get("confidence", 1.0)
            labels.dimension_confidences["temporal"] = confidence

        # Store entity count
        labels.entity_counts["temporal"] = len(temporal_entities)

    def _create_temporal_hierarchy(
        self,
        normalized_value: str,
        temporal_type: str,
    ) -> List[str]:
        """
        Create temporal hierarchy labels from normalized value.

        Args:
            normalized_value: ISO 8601 temporal value (e.g., "2022-03-15")
            temporal_type: Type of temporal value (date, datetime, time, duration, period)

        Returns:
            List of hierarchical labels (e.g., ["2022", "2022-Q1", "2022-03", "2022-03-15"])
        """
        labels = []

        try:
            if temporal_type in ("date", "datetime"):
                # Parse ISO 8601 date
                if "T" in normalized_value:
                    # datetime format: 2022-03-15T14:30:00
                    dt = datetime.fromisoformat(normalized_value)
                else:
                    # date format: 2022-03-15
                    dt = datetime.fromisoformat(normalized_value + "T00:00:00")

                # Year level
                year = dt.strftime("%Y")
                labels.append(year)

                # Quarter level
                quarter = (dt.month - 1) // 3 + 1
                labels.append(f"{year}-Q{quarter}")

                # Month level
                month = dt.strftime("%Y-%m")
                labels.append(month)

                # Day level
                day = dt.strftime("%Y-%m-%d")
                labels.append(day)

                # If datetime, add time hierarchy
                if temporal_type == "datetime" and "T" in normalized_value:
                    # Hour level
                    hour = dt.strftime("%Y-%m-%dT%H")
                    labels.append(hour)

            elif temporal_type == "year":
                # Just year: 2022
                labels.append(normalized_value)

            elif temporal_type == "month":
                # Year-month: 2022-03
                year = normalized_value[:4]
                labels.append(year)

                quarter = (int(normalized_value[5:7]) - 1) // 3 + 1
                labels.append(f"{year}-Q{quarter}")

                labels.append(normalized_value)  # 2022-03

            elif temporal_type == "quarter":
                # Year-quarter: 2022-Q1
                year = normalized_value[:4]
                labels.append(year)
                labels.append(normalized_value)

        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse temporal value '{normalized_value}': {e}")
            # Fallback: just add the raw value
            labels.append(normalized_value)

        return labels

    def _process_spatial_dimension(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> None:
        """Process spatial dimension and generate hierarchy."""
        spatial_entities = extraction_result.spatial_entities

        if not spatial_entities:
            return

        # Get first spatial entity (primary location)
        entity = spatial_entities[0]

        # Generate spatial hierarchy from geocoded result
        latitude = entity.get("latitude")
        longitude = entity.get("longitude")
        location_text = entity.get("text", "")

        if latitude is not None and longitude is not None:
            # Store coordinates
            labels.latitude = latitude
            labels.longitude = longitude

            # Generate hierarchy from reverse geocoding
            hierarchy = self._create_spatial_hierarchy(
                latitude, longitude, location_text
            )
            labels.spatial_labels = hierarchy

            # Create hierarchy path
            labels.spatial_path = " > ".join(hierarchy)

            # Store confidence
            confidence = entity.get("confidence", 1.0)
            labels.dimension_confidences["spatial"] = confidence

        # Store entity count
        labels.entity_counts["spatial"] = len(spatial_entities)

    def _create_spatial_hierarchy(
        self,
        latitude: float,
        longitude: float,
        location_text: str,
    ) -> List[str]:
        """
        Create spatial hierarchy labels from geocoded coordinates.

        This is a simplified version. In production, you would:
        1. Use reverse geocoding to get administrative hierarchy
        2. Query OSM/Google Maps API for location details
        3. Use a gazetteer database for lookups

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            location_text: Original location text

        Returns:
            List of hierarchical labels (e.g., ["Australia", "Western Australia", "Kimberley", "Broome"])
        """
        # For now, return simplified hierarchy based on location text
        # In Phase 3 ETL, we'll implement full reverse geocoding

        labels = []

        # Simple heuristic: split location text by comma
        if "," in location_text:
            parts = [p.strip() for p in location_text.split(",")]
            labels.extend(reversed(parts))  # Reverse for hierarchy (country > state > city)
        else:
            labels.append(location_text)

        # TODO Phase 3: Implement reverse geocoding with Nominatim/Google Maps
        # hierarchy = reverse_geocode(latitude, longitude)
        # labels = [
        #     hierarchy.get("country"),
        #     hierarchy.get("state"),
        #     hierarchy.get("region"),
        #     hierarchy.get("city"),
        # ]

        return [l for l in labels if l]  # Filter out None values

    def _process_event_dimension(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> None:
        """Process event dimension and generate hierarchy."""
        # Get event entities from generic entities dict
        event_entities = extraction_result.entities.get("event", [])

        if not event_entities:
            return

        # Get first event (primary event)
        entity = event_entities[0]

        # Generate event hierarchy from category
        category = entity.get("category", "")

        if category:
            # Parse category hierarchy (e.g., "natural_disaster > storm > cyclone")
            if ">" in category:
                hierarchy = [c.strip() for c in category.split(">")]
            else:
                hierarchy = [category]

            labels.event_labels = hierarchy

            # Create hierarchy path
            labels.event_path = " > ".join(hierarchy)

            # Store confidence
            confidence = entity.get("confidence", 1.0)
            labels.dimension_confidences["event"] = confidence

        # Store entity count
        labels.entity_counts["event"] = len(event_entities)

    def _process_entity_dimension(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> None:
        """Process named entity dimension and generate hierarchy."""
        # Get entity entities from generic entities dict
        entity_entities = extraction_result.entities.get("entity", [])

        if not entity_entities:
            return

        # Collect all entity labels (can have multiple entities)
        all_labels = []

        for entity in entity_entities:
            category = entity.get("category", "")
            entity_name = entity.get("text", "")

            if category:
                # Parse category hierarchy (e.g., "organization > government_agency")
                if ">" in category:
                    hierarchy = [c.strip() for c in category.split(">")]
                else:
                    hierarchy = [category]

                # Add entity name at end
                if entity_name:
                    hierarchy.append(entity_name)

                all_labels.extend(hierarchy)

        if all_labels:
            # Deduplicate while preserving order
            labels.entity_labels = list(dict.fromkeys(all_labels))

            # Create hierarchy path (use first entity for path)
            labels.entity_path = " > ".join(labels.entity_labels)

            # Store average confidence
            confidences = [e.get("confidence", 1.0) for e in entity_entities]
            labels.dimension_confidences["entity"] = sum(confidences) / len(confidences)

        # Store entity count
        labels.entity_counts["entity"] = len(entity_entities)


# ============================================================================
# Helper Functions
# ============================================================================

def extract_year_from_date(normalized_value: str) -> Optional[str]:
    """Extract year from ISO 8601 date string."""
    try:
        if len(normalized_value) >= 4:
            return normalized_value[:4]
    except (IndexError, TypeError):
        pass
    return None


def extract_quarter_from_date(normalized_value: str) -> Optional[int]:
    """Extract quarter (1-4) from ISO 8601 date string."""
    try:
        if len(normalized_value) >= 7:
            month = int(normalized_value[5:7])
            return (month - 1) // 3 + 1
    except (IndexError, ValueError, TypeError):
        pass
    return None


def extract_month_from_date(normalized_value: str) -> Optional[str]:
    """Extract year-month from ISO 8601 date string."""
    try:
        if len(normalized_value) >= 7:
            return normalized_value[:7]  # YYYY-MM
    except (IndexError, TypeError):
        pass
    return None
