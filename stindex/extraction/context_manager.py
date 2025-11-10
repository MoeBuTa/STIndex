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
    prior_temporal_refs: List[Dict] = field(default_factory=list)
    prior_spatial_refs: List[Dict] = field(default_factory=list)
    prior_events: List[Dict] = field(default_factory=list)

    # State context (cstate)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    current_chunk_index: int = 0
    total_chunks: int = 0
    section_hierarchy: str = ""

    # Configuration
    max_memory_refs: int = 10  # Keep last N references
    enable_nearby_locations: bool = False  # OSM nearby locations feature

    def to_prompt_context(self) -> str:
        """
        Convert context to prompt string for LLM.

        Builds context sections:
        1. Document context (publication date, source location, position)
        2. Previous temporal references (last 5)
        3. Previous spatial references (last 5)
        4. Previous events (if available)

        Returns:
            Formatted context string for LLM prompt
        """
        sections = []

        # Document context (cstate)
        if self.document_metadata:
            sections.append("# Document Context")
            pub_date = self.document_metadata.get('publication_date', 'Unknown')
            source_loc = self.document_metadata.get('source_location', 'Unknown')
            sections.append(f"Publication Date: {pub_date}")
            sections.append(f"Source Location: {source_loc}")
            sections.append(f"Current Position: Chunk {self.current_chunk_index + 1} of {self.total_chunks}")

            if self.section_hierarchy:
                sections.append(f"Section: {self.section_hierarchy}")

            sections.append("")

        # Memory context (cmem) - Prior extractions
        if self.prior_temporal_refs:
            sections.append("# Previous Temporal References")
            sections.append("Use these references to resolve relative temporal expressions:")
            for ref in self.prior_temporal_refs[-5:]:  # Last 5
                sections.append(f"- {ref['text']} → {ref['normalized']}")
            sections.append("")

        if self.prior_spatial_refs:
            sections.append("# Previous Spatial References")
            sections.append("Locations already mentioned in this document:")
            for ref in self.prior_spatial_refs[-5:]:  # Last 5
                parent = ref.get('parent_region', '')
                parent_str = f" ({parent})" if parent else ""
                sections.append(f"- {ref['text']}{parent_str}")
            sections.append("")

        if self.prior_events:
            sections.append("# Previous Events")
            for event in self.prior_events[-3:]:  # Last 3
                sections.append(f"- {event.get('text', '')}")
            sections.append("")

        return "\n".join(sections)

    def update_memory(self, extraction_result: Dict[str, Any]):
        """
        Update memory context with new extraction results.

        This method is called after each chunk extraction to maintain
        running context across document processing.

        Args:
            extraction_result: Dictionary with extraction results
                Expected keys: temporal_entities, spatial_entities, etc.
        """
        # Update temporal memory
        temporal_entities = extraction_result.get('temporal_entities', [])
        for entity in temporal_entities:
            self.prior_temporal_refs.append({
                'text': entity.get('text', ''),
                'normalized': entity.get('normalized', ''),
                'chunk_index': self.current_chunk_index
            })

        # Update spatial memory
        spatial_entities = extraction_result.get('spatial_entities', [])
        for entity in spatial_entities:
            self.prior_spatial_refs.append({
                'text': entity.get('text', ''),
                'parent_region': entity.get('parent_region'),
                'chunk_index': self.current_chunk_index
            })

        # Update event memory (if available)
        event_entities = extraction_result.get('event', [])
        for entity in event_entities:
            self.prior_events.append({
                'text': entity.get('text', ''),
                'category': entity.get('category'),
                'chunk_index': self.current_chunk_index
            })

        # Keep only last N references (sliding window)
        self.prior_temporal_refs = self.prior_temporal_refs[-self.max_memory_refs:]
        self.prior_spatial_refs = self.prior_spatial_refs[-self.max_memory_refs:]
        self.prior_events = self.prior_events[-self.max_memory_refs:]

        logger.debug(
            f"Updated context memory: {len(self.prior_temporal_refs)} temporal, "
            f"{len(self.prior_spatial_refs)} spatial refs"
        )

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
        """Reset memory context (for new document)."""
        self.prior_temporal_refs = []
        self.prior_spatial_refs = []
        self.prior_events = []
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
            "prior_temporal_refs": self.prior_temporal_refs,
            "prior_spatial_refs": self.prior_spatial_refs,
            "prior_events": self.prior_events,
            "geocoding_provider": self.geocoding_provider,
            "enable_nearby_locations": self.enable_nearby_locations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionContext":
        """
        Deserialize context from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ExtractionContext instance
        """
        return cls(
            document_metadata=data.get("document_metadata", {}),
            current_chunk_index=data.get("current_chunk_index", 0),
            total_chunks=data.get("total_chunks", 0),
            section_hierarchy=data.get("section_hierarchy", ""),
            prior_temporal_refs=data.get("prior_temporal_refs", []),
            prior_spatial_refs=data.get("prior_spatial_refs", []),
            prior_events=data.get("prior_events", []),
            geocoding_provider=data.get("geocoding_provider", "nominatim"),
            enable_nearby_locations=data.get("enable_nearby_locations", False),
        )
