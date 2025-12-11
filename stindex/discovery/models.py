"""
Pydantic models for schema discovery pipeline.

Provides type-safe data structures for:
- Discovered dimensional schemas
- Cluster-level discovery results
- Entity extraction results
- Merged cross-cluster schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field, field_validator
import difflib


# ============================================================================
# DIMENSION SCHEMA MODELS
# ============================================================================

class DiscoveredDimensionSchema(BaseModel):
    """
    Schema for a single discovered dimension.

    Represents the dimensional structure discovered by LLM from questions.
    """
    hierarchy: List[str] = Field(
        description="Hierarchy levels from specific to general (e.g., ['specific_symptom', 'symptom_category'])",
        min_length=1
    )
    description: str = Field(description="Natural language description of what this dimension captures")
    examples: List[str] = Field(
        default_factory=list,
        description="Example entities illustrating this dimension"
    )

    @field_validator('hierarchy')
    @classmethod
    def validate_hierarchy(cls, v: List[str]) -> List[str]:
        """Validate hierarchy structure."""
        if not v:
            raise ValueError("Hierarchy cannot be empty")
        # Normalize to snake_case
        return [level.strip().lower().replace(' ', '_') for level in v]

    def get_field_definitions(self) -> List[Dict[str, str]]:
        """Convert hierarchy to field definitions for prompts."""
        return [
            {
                'name': level,
                'type': 'string',
                'description': f'{level.replace("_", " ").title()} level'
            }
            for level in self.hierarchy
        ]


# ============================================================================
# ENTITY MODELS
# ============================================================================

class HierarchicalEntity(BaseModel):
    """
    Entity with hierarchical fields.

    Represents an extracted entity with values at different hierarchy levels.
    """
    text: str = Field(description="Primary entity text (canonical name)")
    hierarchy_values: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Map of hierarchy level → value (e.g., {'specific_symptom': 'fever', 'symptom_category': 'systemic'})"
    )
    dimension: str = Field(description="Dimension name this entity belongs to")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate entity text."""
        if not v or not v.strip():
            raise ValueError("Entity text cannot be empty")
        return v.strip()

    def get_hierarchy_value(self, level: str) -> Optional[str]:
        """Get value at specific hierarchy level."""
        return self.hierarchy_values.get(level)

    def matches_entity(self, other: 'HierarchicalEntity', similarity_threshold: float = 0.85) -> bool:
        """Check if this entity matches another based on text similarity."""
        similarity = difflib.SequenceMatcher(
            None,
            self.text.lower(),
            other.text.lower()
        ).ratio()
        return similarity >= similarity_threshold


# ============================================================================
# CLUSTER DISCOVERY RESULT MODELS
# ============================================================================

class ClusterSchemaDiscoveryResult(BaseModel):
    """
    Result of cluster-level schema discovery.

    Contains both discovered dimensions and extracted entities from all questions
    in the cluster.
    """
    cluster_id: int = Field(ge=0)
    n_questions: int = Field(ge=0, description="Total questions processed in cluster")

    # Discovered dimensional schemas
    discovered_dimensions: Dict[str, DiscoveredDimensionSchema] = Field(
        default_factory=dict,
        description="Map of dimension name → discovered schema"
    )

    # Extracted entities per dimension
    entities: Dict[str, List[HierarchicalEntity]] = Field(
        default_factory=dict,
        description="Map of dimension name → list of extracted entities"
    )

    # Metadata
    reasoning: Optional[str] = Field(default=None, description="LLM reasoning for schema discovery")
    extraction_time: float = Field(default=0.0, ge=0.0, description="Time taken for discovery + extraction (seconds)")

    def get_entity_count(self, dimension: str) -> int:
        """Get number of entities for a dimension."""
        return len(self.entities.get(dimension, []))

    def get_total_entities(self) -> int:
        """Get total entities across all dimensions."""
        return sum(len(entities) for entities in self.entities.values())

    def get_dimension_stats(self) -> Dict[str, int]:
        """Get entity counts per dimension."""
        return {dim: len(entities) for dim, entities in self.entities.items()}


# ============================================================================
# MERGED SCHEMA MODELS
# ============================================================================

class DimensionSource(BaseModel):
    """Track which clusters contributed to a dimension."""
    cluster_ids: Set[int] = Field(default_factory=set, description="Cluster IDs that discovered this dimension")
    entity_counts_per_cluster: Dict[int, int] = Field(
        default_factory=dict,
        description="Map of cluster ID → entity count"
    )

    def add_cluster(self, cluster_id: int, entity_count: int = 0):
        """Add a cluster as a source."""
        self.cluster_ids.add(cluster_id)
        if entity_count > 0:
            self.entity_counts_per_cluster[cluster_id] = entity_count

    def get_total_contributing_clusters(self) -> int:
        """Get number of clusters that contributed to this dimension."""
        return len(self.cluster_ids)


class MergedDimensionSchema(BaseModel):
    """
    Merged dimensional schema from multiple clusters.

    Combines schemas discovered by different clusters with entity aggregation.
    """
    hierarchy: List[str] = Field(description="Merged hierarchy levels")
    description: str = Field(description="Merged description")
    examples: List[str] = Field(default_factory=list, description="Combined examples from all clusters")

    # Aggregated entities
    entities: List[HierarchicalEntity] = Field(
        default_factory=list,
        description="Deduplicated entities from all clusters"
    )

    # Statistics
    total_entity_count: int = Field(default=0, ge=0)
    sources: DimensionSource = Field(default_factory=DimensionSource, description="Cluster sources for this dimension")

    # Alternative dimension names (for fuzzy matching)
    alternative_names: Set[str] = Field(
        default_factory=set,
        description="Alternative names from different clusters (e.g., 'disease', 'medical_condition')"
    )

    def add_entities_from_cluster(self, cluster_id: int, entities: List[HierarchicalEntity]):
        """Add entities from a cluster (deduplication handled by caller)."""
        self.entities.extend(entities)
        self.total_entity_count = len(self.entities)
        self.sources.add_cluster(cluster_id, len(entities))

    def merge_schema_from_cluster(self, cluster_schema: DiscoveredDimensionSchema, cluster_id: int):
        """Merge schema information from a cluster."""
        # Add examples
        self.examples.extend([ex for ex in cluster_schema.examples if ex not in self.examples])

        # Track source
        self.sources.add_cluster(cluster_id, 0)


class FinalSchema(BaseModel):
    """
    Final merged schema across all clusters.

    Contains all discovered dimensions with deduplicated entities and metadata.
    """
    dimensions: Dict[str, MergedDimensionSchema] = Field(
        default_factory=dict,
        description="Map of dimension name → merged schema"
    )

    # Pipeline metadata
    n_clusters_processed: int = Field(ge=0)
    total_questions_processed: int = Field(ge=0)

    # Statistics
    pipeline_time: float = Field(default=0.0, ge=0.0, description="Total pipeline time (seconds)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Pipeline completion timestamp (ISO 8601)")

    def get_dimension_names(self) -> List[str]:
        """Get sorted list of dimension names."""
        return sorted(self.dimensions.keys())

    def get_total_entities(self) -> int:
        """Get total entities across all dimensions."""
        return sum(dim.total_entity_count for dim in self.dimensions.values())

    def get_dimension_entity_counts(self) -> Dict[str, int]:
        """Get entity counts per dimension."""
        return {
            name: dim.total_entity_count
            for name, dim in self.dimensions.items()
        }

    def to_yaml_dict(self) -> Dict:
        """Convert to YAML-friendly dict for saving."""
        output = {}
        for dim_name, dim_schema in self.dimensions.items():
            output[dim_name] = {
                'hierarchy': dim_schema.hierarchy,
                'description': dim_schema.description,
                'examples': dim_schema.examples,
                'count': dim_schema.total_entity_count,
                'entities': [
                    {
                        'text': entity.text,
                        **entity.hierarchy_values
                    }
                    for entity in dim_schema.entities
                ],
                'sources': {
                    f"cluster_{cid}": dim_schema.sources.entity_counts_per_cluster.get(cid, 0)
                    for cid in sorted(dim_schema.sources.cluster_ids)
                }
            }
        return output


# ============================================================================
# BATCH EXTRACTION RESULT (for progressive extraction)
# ============================================================================

class BatchExtractionResult(BaseModel):
    """
    Result from processing a batch of questions.

    Used internally during progressive entity extraction.
    """
    batch_idx: int = Field(ge=0)
    n_questions: int = Field(ge=0)

    # Extracted entities in this batch
    entities: Dict[str, List[HierarchicalEntity]] = Field(default_factory=dict)

    # Newly discovered dimensions in this batch (if allow_new_dimensions=True)
    new_dimensions: Dict[str, DiscoveredDimensionSchema] = Field(default_factory=dict)

    # Metadata
    reasoning: Optional[str] = Field(default=None)
    extraction_time: float = Field(default=0.0, ge=0.0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_dimension_schemas(
    schemas: List[DiscoveredDimensionSchema],
    dimension_name: str
) -> DiscoveredDimensionSchema:
    """
    Merge multiple discovered schemas for the same dimension.

    Used when different clusters discover the same dimension with slight variations.

    Args:
        schemas: List of schemas to merge
        dimension_name: Name of the dimension

    Returns:
        Merged schema
    """
    if not schemas:
        raise ValueError("Cannot merge empty list of schemas")

    if len(schemas) == 1:
        return schemas[0]

    # Use the first schema as base
    base = schemas[0]

    # Merge hierarchy (use longest one)
    merged_hierarchy = max([s.hierarchy for s in schemas], key=len)

    # Merge descriptions (concatenate unique descriptions)
    unique_descs = list(dict.fromkeys([s.description for s in schemas]))
    merged_description = "; ".join(unique_descs)

    # Merge examples (deduplicate)
    all_examples = []
    for schema in schemas:
        all_examples.extend(schema.examples)
    merged_examples = list(dict.fromkeys(all_examples))

    return DiscoveredDimensionSchema(
        hierarchy=merged_hierarchy,
        description=merged_description,
        examples=merged_examples
    )
