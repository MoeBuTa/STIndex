"""
Extraction context management for context-aware extraction.

Implements context engineering best practices:
- cinstr: Instruction context (task definition, schemas)
- ctools: Tool context (available post-processing)
- cmem: Memory context (prior extractions)
- cstate: State context (document metadata, position)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ExtractionContext:
    """
    Manages all context components for extraction.

    Implements context engineering best practices from 2025 survey:
    - cinstr: Instruction context (task definition, schemas)
    - ctools: Tool context (available post-processing)
    - cmem: Memory context (prior extractions)
    - cstate: State context (document metadata, position)

    This enables context-aware extraction that:
    - Resolves relative temporal expressions using prior references
    - Disambiguates spatial mentions using document location context
    - Maintains consistency across document chunks
    """

    # Instruction context (cinstr)
    dimension_schemas: Dict[str, Any] = field(default_factory=dict)
    few_shot_examples: List[Dict] = field(default_factory=list)

    # Tool context (ctools)
    available_tools: Dict[str, Any] = field(default_factory=dict)
    geocoding_provider: str = "nominatim"
    rate_limits: Dict[str, float] = field(default_factory=dict)

    # Memory context (cmem) - Prior extractions across chunks
    # Dynamic: supports any dimension (temporal, spatial, symptom, disease, etc.)
    prior_refs: Dict[str, List[Dict]] = field(default_factory=dict)

    # State context (cstate)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    current_chunk_index: int = 0
    total_chunks: int = 0
    section_hierarchy: str = ""

    # Configuration
    max_memory_refs: int = 10  # Keep last N references
    enable_nearby_locations: bool = False  # OSM nearby locations feature

    # Backward compatibility properties for existing code
    @property
    def prior_temporal_refs(self) -> List[Dict]:
        """Backward compatibility: access temporal refs from prior_refs."""
        return self.prior_refs.get('temporal', [])

    @prior_temporal_refs.setter
    def prior_temporal_refs(self, value: List[Dict]):
        """Backward compatibility: set temporal refs in prior_refs."""
        self.prior_refs['temporal'] = value

    @property
    def prior_spatial_refs(self) -> List[Dict]:
        """Backward compatibility: access spatial refs from prior_refs."""
        return self.prior_refs.get('spatial', [])

    @prior_spatial_refs.setter
    def prior_spatial_refs(self, value: List[Dict]):
        """Backward compatibility: set spatial refs in prior_refs."""
        self.prior_refs['spatial'] = value

    @property
    def prior_events(self) -> List[Dict]:
        """Backward compatibility: access event refs from prior_refs."""
        return self.prior_refs.get('event', [])

    @prior_events.setter
    def prior_events(self, value: List[Dict]):
        """Backward compatibility: set event refs in prior_refs."""
        self.prior_refs['event'] = value

    def to_prompt_context(self) -> str:
        """
        Convert context to prompt string for LLM.

        Builds context sections:
        1. Document context (publication date, source location, cluster ID, position)
        2. Previous dimensional references (dynamic - any dimension)

        Supports both document extraction (temporal, spatial, event) and
        schema discovery (symptom, disease, treatment, etc.)

        Returns:
            Formatted context string for LLM prompt
        """
        sections = []

        # Document context (cstate)
        if self.document_metadata:
            sections.append("# Document Context")

            # Publication date (for document extraction)
            pub_date = self.document_metadata.get('publication_date')
            if pub_date and pub_date != 'Unknown':
                sections.append(f"Publication Date: {pub_date}")

            # Source location (for document extraction)
            source_loc = self.document_metadata.get('source_location')
            if source_loc and source_loc != 'Unknown':
                sections.append(f"Source Location: {source_loc}")

            # Cluster ID (for schema discovery)
            cluster_id = self.document_metadata.get('cluster_id')
            if cluster_id is not None:
                sections.append(f"Cluster ID: {cluster_id}")

            # Position
            if self.total_chunks > 0:
                sections.append(f"Current Position: Chunk {self.current_chunk_index + 1} of {self.total_chunks}")

            # Section hierarchy (for long documents)
            if self.section_hierarchy:
                sections.append(f"Section: {self.section_hierarchy}")

            sections.append("")

        # Memory context (cmem) - DYNAMIC dimensions
        # Sort dimensions for consistent ordering
        sorted_dimensions = sorted(self.prior_refs.keys())

        for dimension_name in sorted_dimensions:
            refs = self.prior_refs[dimension_name]
            if not refs:
                continue

            # Format dimension name for display
            dim_display = dimension_name.replace('_', ' ').title()

            # Different formatting for different dimension types
            if dimension_name == 'temporal':
                # Temporal: show text → normalized
                sections.append(f"# Previous {dim_display} References")
                sections.append("Use these references to resolve relative temporal expressions:")
                for ref in refs[-5:]:  # Last 5
                    normalized = ref.get('normalized', '')
                    if normalized:
                        sections.append(f"- {ref['text']} → {normalized}")
                    else:
                        sections.append(f"- {ref['text']}")
                sections.append("")

            elif dimension_name == 'spatial':
                # Spatial: show location with parent region
                sections.append(f"# Previous {dim_display} References")
                sections.append("Locations already mentioned:")
                for ref in refs[-5:]:  # Last 5
                    parent = ref.get('parent_region', '')
                    parent_str = f" ({parent})" if parent else ""
                    sections.append(f"- {ref['text']}{parent_str}")
                sections.append("")

            else:
                # Generic dimensions (symptom, disease, treatment, event, etc.)
                sections.append(f"# Previous {dim_display} Extractions")
                sections.append("Maintain consistent terminology:")
                for ref in refs[-5:]:  # Last 5
                    # Show text and category/type if available
                    text = ref.get('text', '')
                    category = ref.get('category', '') or ref.get('type', '')
                    if category:
                        sections.append(f"- {text} ({category})")
                    else:
                        sections.append(f"- {text}")
                sections.append("")

        return "\n".join(sections)

    def update_memory(self, extraction_result: Dict[str, Any]):
        """
        Update memory context with new extraction results.

        This method is called after each chunk/question extraction to maintain
        running context across document processing or question clustering.

        Supports DYNAMIC dimensions - works for any dimension type:
        - Document extraction: temporal, spatial, event
        - Schema discovery: symptom, disease, treatment, body_system, etc.

        Args:
            extraction_result: Dictionary with extraction results
                Keys are dimension names (e.g., 'temporal', 'symptom', 'disease')
                Values are lists of entity dictionaries
                Skip meta fields like 'new_dimensions', 'metadata'
        """
        # Meta fields to skip (not dimensions)
        skip_fields = {'new_dimensions', 'metadata', 'raw_llm_output', 'extraction_config'}

        # Update memory for each dimension dynamically
        for dimension_name, entities in extraction_result.items():
            # Skip meta fields
            if dimension_name in skip_fields:
                continue

            # Skip if not a list of entities
            if not isinstance(entities, list):
                continue

            # Initialize dimension tracking if first time
            if dimension_name not in self.prior_refs:
                self.prior_refs[dimension_name] = []

            # Add each entity to memory
            for entity in entities:
                if not isinstance(entity, dict):
                    continue

                # Build reference entry
                ref_entry = {
                    'text': entity.get('text', ''),
                    'chunk_index': self.current_chunk_index
                }

                # Add dimension-specific fields
                if 'normalized' in entity:
                    ref_entry['normalized'] = entity['normalized']
                if 'parent_region' in entity:
                    ref_entry['parent_region'] = entity['parent_region']
                if 'category' in entity:
                    ref_entry['category'] = entity['category']
                if 'type' in entity:
                    ref_entry['type'] = entity['type']

                self.prior_refs[dimension_name].append(ref_entry)

            # Keep only last N references per dimension (sliding window)
            self.prior_refs[dimension_name] = self.prior_refs[dimension_name][-self.max_memory_refs:]

        # Log update
        if self.prior_refs:
            dimension_counts = {dim: len(refs) for dim, refs in self.prior_refs.items()}
            logger.debug(f"Updated context memory: {dimension_counts}")
        else:
            logger.debug("Context memory update: no dimensions tracked")

    def get_anchor_date(self) -> Optional[str]:
        """
        Get anchor date for relative temporal resolution.

        Priority order:
        1. Most recent temporal reference in prior extractions
        2. Document publication date
        3. None

        Returns:
            ISO 8601 date string or None
        """
        # Try to use most recent prior temporal reference
        if self.prior_temporal_refs:
            return self.prior_temporal_refs[-1].get('normalized')

        # Fall back to document publication date
        return self.document_metadata.get('publication_date')

    def get_spatial_context(self) -> Optional[str]:
        """
        Get spatial context for location disambiguation.

        Priority order:
        1. Document source_location metadata
        2. Most recent spatial reference
        3. None

        Returns:
            Location string or None
        """
        source_loc = self.document_metadata.get('source_location')
        if source_loc:
            return source_loc

        # Fall back to most recent spatial reference
        if self.prior_spatial_refs:
            return self.prior_spatial_refs[-1].get('text')

        return None

    def get_nearby_locations_context(self, location_coords=None) -> str:
        """
        Get nearby locations context for spatial disambiguation.

        Requires OSMContextProvider (imported dynamically to avoid circular dependency).

        Args:
            location_coords: (lat, lon) tuple for reference location

        Returns:
            Formatted nearby locations context string
        """
        if not self.enable_nearby_locations:
            return ""

        try:
            # Import here to avoid circular dependency
            from stindex.postprocess.spatial.osm_context import OSMContextProvider

            # If no coords provided, try to geocode source_location
            if not location_coords and self.document_metadata.get('source_location'):
                from stindex.postprocess.spatial.geocoder import GeocoderService
                geocoder = GeocoderService()
                location_coords = geocoder.get_coordinates(
                    self.document_metadata['source_location']
                )

            if location_coords:
                osm = OSMContextProvider()
                nearby = osm.get_nearby_locations(location_coords, radius_km=100)

                if nearby:
                    context = "# Nearby Geographic Features\n"
                    for poi in nearby[:5]:  # Top 5
                        context += (
                            f"- {poi['name']} ({poi['type']}): "
                            f"{poi['distance_km']}km {poi['direction']}\n"
                        )
                    return context

        except Exception as e:
            logger.warning(f"Failed to get nearby locations context: {e}")

        return ""

    def reset_memory(self):
        """Reset memory context (for new document or cluster)."""
        self.prior_refs = {}
        self.current_chunk_index = 0
        logger.debug("Context memory reset")

    def set_chunk_position(self, chunk_index: int, total_chunks: int, section_hierarchy: str = ""):
        """
        Update current chunk position.

        Args:
            chunk_index: Current chunk index (0-based)
            total_chunks: Total number of chunks
            section_hierarchy: Section hierarchy string (e.g., "Introduction > Background")
        """
        self.current_chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.section_hierarchy = section_hierarchy

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize context to dictionary.

        Returns:
            Dictionary representation of context
        """
        return {
            "document_metadata": self.document_metadata,
            "current_chunk_index": self.current_chunk_index,
            "total_chunks": self.total_chunks,
            "section_hierarchy": self.section_hierarchy,
            "prior_refs": self.prior_refs,  # Dynamic dimensions
            "geocoding_provider": self.geocoding_provider,
            "enable_nearby_locations": self.enable_nearby_locations,
            "max_memory_refs": self.max_memory_refs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionContext":
        """
        Deserialize context from dictionary.

        Supports both new format (prior_refs) and old format (prior_temporal_refs, etc.)
        for backward compatibility.

        Args:
            data: Dictionary representation

        Returns:
            ExtractionContext instance
        """
        # Handle new format (prior_refs dict)
        if "prior_refs" in data:
            return cls(
                document_metadata=data.get("document_metadata", {}),
                current_chunk_index=data.get("current_chunk_index", 0),
                total_chunks=data.get("total_chunks", 0),
                section_hierarchy=data.get("section_hierarchy", ""),
                prior_refs=data.get("prior_refs", {}),
                geocoding_provider=data.get("geocoding_provider", "nominatim"),
                enable_nearby_locations=data.get("enable_nearby_locations", False),
                max_memory_refs=data.get("max_memory_refs", 10),
            )

        # Handle old format (prior_temporal_refs, prior_spatial_refs, prior_events)
        # Convert to new format automatically
        prior_refs = {}
        if "prior_temporal_refs" in data:
            prior_refs["temporal"] = data["prior_temporal_refs"]
        if "prior_spatial_refs" in data:
            prior_refs["spatial"] = data["prior_spatial_refs"]
        if "prior_events" in data:
            prior_refs["event"] = data["prior_events"]

        return cls(
            document_metadata=data.get("document_metadata", {}),
            current_chunk_index=data.get("current_chunk_index", 0),
            total_chunks=data.get("total_chunks", 0),
            section_hierarchy=data.get("section_hierarchy", ""),
            prior_refs=prior_refs,
            geocoding_provider=data.get("geocoding_provider", "nominatim"),
            enable_nearby_locations=data.get("enable_nearby_locations", False),
            max_memory_refs=data.get("max_memory_refs", 10),
        )
