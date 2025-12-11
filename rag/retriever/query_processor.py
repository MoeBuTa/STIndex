"""
Query Processor for RAG retrieval.

Handles query preprocessing, expansion, and analysis:
- Query cleaning and normalization
- Entity extraction (temporal, spatial)
- Query decomposition for multi-hop
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original_query: str
    cleaned_query: str
    temporal_hints: List[str] = field(default_factory=list)
    spatial_hints: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    query_type: str = "simple"  # simple, temporal, spatial, comparison, multi-hop


class QueryProcessor:
    """
    Process queries for improved retrieval.

    Extracts hints for dimensional filtering and decomposes
    complex queries into sub-queries.
    """

    def __init__(self):
        """Initialize query processor."""
        # Temporal patterns
        self.temporal_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b(Q[1-4])\b',  # Quarters
            r'\b(before|after|during|in|on|since)\s+\d',  # Temporal prepositions
        ]

        # Spatial patterns
        self.spatial_patterns = [
            r'\b(in|at|from|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Locations
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2,})\b',  # City, State/Country
        ]

        # Multi-hop indicators
        self.multihop_patterns = [
            r'\bwho\s+(?:is|was)\s+the\s+\w+\s+of\s+the\s+\w+\s+that\b',  # Who is the X of the Y that...
            r'\bwhat\s+(?:is|was)\s+the\s+\w+\s+of\s+the\s+\w+\s+that\b',
            r'\band\s+(?:also|then)\b',  # Multi-part questions
        ]

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query for retrieval.

        Args:
            query: Raw query string

        Returns:
            ProcessedQuery with extracted information
        """
        # Clean query
        cleaned = self._clean_query(query)

        # Extract temporal hints
        temporal_hints = self._extract_temporal_hints(query)

        # Extract spatial hints
        spatial_hints = self._extract_spatial_hints(query)

        # Extract entities
        entities = self._extract_entities(query)

        # Determine query type
        query_type = self._classify_query(query, temporal_hints, spatial_hints)

        # Decompose if multi-hop
        sub_queries = []
        if query_type == "multi-hop":
            sub_queries = self._decompose_query(query)

        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            temporal_hints=temporal_hints,
            spatial_hints=spatial_hints,
            entities=entities,
            sub_queries=sub_queries,
            query_type=query_type,
        )

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Strip whitespace
        cleaned = query.strip()

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove question marks for search
        cleaned = cleaned.rstrip('?')

        return cleaned

    def _extract_temporal_hints(self, query: str) -> List[str]:
        """Extract temporal entities from query."""
        hints = []

        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            hints.extend(m if isinstance(m, str) else m[0] for m in matches)

        return list(set(hints))

    def _extract_spatial_hints(self, query: str) -> List[str]:
        """Extract spatial entities from query."""
        hints = []

        for pattern in self.spatial_patterns:
            matches = re.findall(pattern, query)
            for m in matches:
                if isinstance(m, tuple):
                    # Take the location part (usually second group)
                    hints.append(m[1] if len(m) > 1 else m[0])
                else:
                    hints.append(m)

        return list(set(hints))

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        # Simple heuristic: capitalized words (excluding sentence start)
        words = query.split()
        entities = []

        for i, word in enumerate(words):
            # Skip first word (sentence start)
            if i == 0:
                continue

            # Check for capitalized word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper():
                entities.append(clean_word)

        return list(set(entities))

    def _classify_query(
        self,
        query: str,
        temporal_hints: List[str],
        spatial_hints: List[str],
    ) -> str:
        """Classify query type."""
        # Check for multi-hop
        for pattern in self.multihop_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return "multi-hop"

        # Check for comparison
        if re.search(r'\b(compare|difference|versus|vs\.?|between)\b', query, re.IGNORECASE):
            return "comparison"

        # Check for temporal focus
        if temporal_hints and not spatial_hints:
            return "temporal"

        # Check for spatial focus
        if spatial_hints and not temporal_hints:
            return "spatial"

        # Default to simple
        return "simple"

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a multi-hop query into sub-queries.

        Simple heuristic decomposition. For better results,
        use LLM-based decomposition.
        """
        sub_queries = [query]

        # Extract "the X that Y" patterns
        pattern = r'the\s+(\w+(?:\s+\w+)*)\s+that\s+([^?]+)'
        match = re.search(pattern, query, re.IGNORECASE)

        if match:
            entity_type = match.group(1)
            condition = match.group(2).strip()

            # First query: find the entity satisfying the condition
            sub_queries.insert(0, f"What {entity_type} {condition}?")

        return sub_queries

    def suggest_filters(
        self,
        processed: ProcessedQuery,
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Suggest temporal and spatial filters based on query analysis.

        Args:
            processed: ProcessedQuery from process()

        Returns:
            Tuple of (temporal_filter_dict, spatial_filter_dict)
        """
        temporal_filter = None
        spatial_filter = None

        # Build temporal filter from hints
        if processed.temporal_hints:
            for hint in processed.temporal_hints:
                # Check if it's a year
                if re.match(r'^(19|20)\d{2}$', hint):
                    temporal_filter = {"year": int(hint)}
                    break
                # Check if it's a quarter
                if re.match(r'^Q[1-4]$', hint, re.IGNORECASE):
                    temporal_filter = {"quarter": int(hint[1])}
                    break

        # Build spatial filter from hints
        if processed.spatial_hints:
            # Use first spatial hint as region filter
            spatial_filter = {"region": processed.spatial_hints[0]}

        return temporal_filter, spatial_filter
