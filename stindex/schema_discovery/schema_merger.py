"""
Schema merger for combining entity lists from all clusters.

Merges entity lists from all clusters with fuzzy deduplication to handle
variations like "fever" vs "Fever" or "influenza" vs "flu".
"""

import difflib
from typing import Dict, List
from loguru import logger


class SchemaMerger:
    """
    Merge entity lists from all clusters.

    Simple approach: concatenate + fuzzy deduplication using string similarity.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize schema merger.

        Args:
            similarity_threshold: Fuzzy matching threshold (0.0-1.0)
                Example: 0.85 means strings with ≥85% similarity are considered duplicates
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized SchemaMerger with similarity threshold = {similarity_threshold}")

    def merge_clusters(
        self,
        cluster_results: List[Dict],
        global_dimensions: Dict[str, Dict]
    ) -> Dict:
        """
        Merge entity lists from all clusters.

        Args:
            cluster_results: List of cluster extraction results:
                [
                    {'cluster_id': 0, 'entities': {'symptom': [...], ...}, ...},
                    {'cluster_id': 1, 'entities': {'symptom': [...], ...}, ...},
                    ...
                ]
            global_dimensions: Global dimensional schemas (for hierarchy, description)

        Returns:
            Final merged schema:
            {
                'symptom': {
                    'hierarchy': ['specific_symptom', 'symptom_category'],
                    'description': '...',
                    'examples': ['fever', 'cough', ...],  # From global discovery
                    'entities': ['fever', 'cough', ...],  # Deduplicated from clusters
                    'count': 127,
                    'sources': {  # Optional: track which clusters contributed
                        'cluster_0': 45,
                        'cluster_1': 38,
                        ...
                    }
                },
                ...
            }
        """
        logger.info(f"Merging entity lists from {len(cluster_results)} clusters")

        # Collect all entities per dimension across clusters
        dimension_entities = {}  # dim_name -> list of (entity_dict, cluster_id) tuples

        for cluster_result in cluster_results:
            cluster_id = cluster_result['cluster_id']
            entities = cluster_result.get('entities', {})

            for dim_name, entity_list in entities.items():
                if dim_name not in dimension_entities:
                    dimension_entities[dim_name] = []

                # Track source cluster for each entity (now dicts with hierarchy)
                for entity in entity_list:
                    # Handle both old format (strings) and new format (dicts with hierarchy)
                    if isinstance(entity, str):
                        # Old format: plain string
                        entity_dict = {'text': entity}
                    else:
                        # New format: dict with hierarchy fields
                        entity_dict = entity

                    dimension_entities[dim_name].append((entity_dict, cluster_id))

        # Merge and deduplicate per dimension
        merged_dims = {}

        for dim_name, dim_schema in global_dimensions.items():
            if dim_name not in dimension_entities:
                # No entities extracted for this dimension
                logger.warning(f"  • {dim_name}: No entities extracted")
                merged_dims[dim_name] = {
                    **dim_schema,  # Include hierarchy, description, examples
                    'entities': [],
                    'count': 0,
                    'sources': {}
                }
                continue

            # Get all entities for this dimension (with source clusters)
            all_entities_with_sources = dimension_entities[dim_name]
            all_entity_dicts = [entity_dict for entity_dict, _ in all_entities_with_sources]

            # Fuzzy deduplication based on 'text' field
            unique_entity_dicts = self._fuzzy_deduplicate_dicts(all_entity_dicts)

            # Track source clusters for entities
            sources = {}
            for entity_dict in unique_entity_dicts:
                entity_text = entity_dict.get('text', '')
                # Find which clusters contributed this entity (or similar variants)
                contributing_clusters = set()
                for other_dict, cluster_id in all_entities_with_sources:
                    other_text = other_dict.get('text', '')
                    if self._are_similar(entity_text, other_text):
                        contributing_clusters.add(cluster_id)

                for cid in contributing_clusters:
                    cluster_key = f"cluster_{cid}"
                    sources[cluster_key] = sources.get(cluster_key, 0) + 1

            # Add to merged schema
            merged_dims[dim_name] = {
                **dim_schema,  # Include hierarchy, description, examples from global discovery
                'entities': sorted(unique_entity_dicts, key=lambda x: x.get('text', '').lower()),
                'count': len(unique_entity_dicts),
                'sources': sources
            }

            logger.info(f"  • {dim_name}: {len(unique_entity_dicts)} unique entities (from {len(all_entity_dicts)} total)")

        logger.info("✓ Merge complete")

        return merged_dims

    def _fuzzy_deduplicate_dicts(self, entities: List[Dict]) -> List[Dict]:
        """
        Deduplicate entity dictionaries using fuzzy string matching on 'text' field.

        Preserves hierarchy fields from the first occurrence of each unique entity.

        Examples:
        - {"text": "fever", "specific_symptom": "fever"} and {"text": "Fever", ...} → keep first
        - {"text": "influenza", ...} and {"text": "flu", ...} → keep both (similarity < threshold)

        Args:
            entities: List of entity dicts with 'text' field and hierarchy fields

        Returns:
            List of unique entity dicts (deduplicated, hierarchy preserved)
        """
        unique = []
        seen_lower = set()

        for entity_dict in entities:
            # Extract text field for comparison
            entity_text = entity_dict.get('text', '').strip()
            entity_lower = entity_text.lower()

            # Skip empty entities
            if not entity_text:
                continue

            # Exact match (case-insensitive)
            if entity_lower in seen_lower:
                continue

            # Fuzzy match against existing unique entities
            is_duplicate = False
            for existing_dict in unique:
                existing_text = existing_dict.get('text', '').lower()
                if self._are_similar(entity_lower, existing_text):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity_dict)
                seen_lower.add(entity_lower)

        return unique

    def _fuzzy_deduplicate(self, entities: List[str]) -> List[str]:
        """
        Deduplicate using fuzzy string matching.

        Examples:
        - "fever" and "Fever " → keep "fever" (exact match after normalization)
        - "influenza" and "flu" → keep both (similarity < threshold)
        - "myocardial infarction" and "myocardial infarction " → keep one

        Args:
            entities: List of entity strings (may contain duplicates/variants)

        Returns:
            List of unique entities (deduplicated)
        """
        unique = []
        seen_lower = set()

        for entity in entities:
            # Normalize: strip whitespace
            entity_clean = entity.strip()
            entity_lower = entity_clean.lower()

            # Skip empty strings
            if not entity_clean:
                continue

            # Exact match (case-insensitive)
            if entity_lower in seen_lower:
                continue

            # Fuzzy match against existing unique entities
            is_duplicate = False
            for existing in unique:
                if self._are_similar(entity_lower, existing.lower()):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity_clean)
                seen_lower.add(entity_lower)

        return unique

    def _are_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two strings are similar based on threshold.

        Uses difflib.SequenceMatcher for character-level similarity.

        Args:
            text1: First string (normalized to lowercase)
            text2: Second string (normalized to lowercase)

        Returns:
            True if similarity >= threshold
        """
        similarity = difflib.SequenceMatcher(
            None,
            text1.lower(),
            text2.lower()
        ).ratio()

        return similarity >= self.similarity_threshold
