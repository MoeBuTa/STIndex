"""
Schema merger for combining entity lists from all clusters.

Merges entity lists from all clusters with fuzzy deduplication to handle
variations like "fever" vs "Fever" or "influenza" vs "flu".

Pure cluster-level merging without global baseline.
"""

import difflib
from typing import Dict, List, Tuple, Set
from loguru import logger

from stindex.discovery.models import (
    ClusterSchemaDiscoveryResult,
    DiscoveredDimensionSchema,
    HierarchicalEntity,
    MergedDimensionSchema,
    FinalSchema,
    DimensionSource
)


class SchemaMerger:
    """
    Merge schemas and entities from all clusters without global baseline.

    Aligns dimensions across clusters using fuzzy name matching,
    then merges entities with deduplication.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize schema merger.

        Args:
            similarity_threshold: Fuzzy matching threshold (0.0-1.0)
                Used for both dimension name matching and entity deduplication.
                Example: 0.85 means strings with ≥85% similarity are considered duplicates
        """
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized SchemaMerger with similarity threshold = {similarity_threshold}")

    def merge_clusters(
        self,
        cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> FinalSchema:
        """
        Merge schemas and entities from all clusters (no global baseline).

        Args:
            cluster_results: List of ClusterSchemaDiscoveryResult from each cluster

        Returns:
            FinalSchema with merged dimensions and deduplicated entities
        """
        import time
        start_time = time.time()

        logger.info(f"Merging schemas and entities from {len(cluster_results)} clusters")

        # Calculate total questions processed
        total_questions = sum(result.n_questions for result in cluster_results)

        # Step 1: Align dimensions across clusters using fuzzy matching
        dimension_groups = self._align_dimensions(cluster_results)
        logger.info(f"  Aligned into {len(dimension_groups)} canonical dimensions")

        # Step 2: Merge schemas and entities for each dimension
        merged_dimensions = {}

        for canonical_name, cluster_schemas in dimension_groups.items():
            merged_dim = self._merge_dimension_group(canonical_name, cluster_schemas, cluster_results)
            merged_dimensions[canonical_name] = merged_dim

            logger.info(
                f"  • {canonical_name}: {merged_dim.total_entity_count} unique entities "
                f"from {len(merged_dim.sources.cluster_ids)} clusters"
            )

        pipeline_time = time.time() - start_time
        logger.info(f"✓ Merge complete in {pipeline_time:.2f}s")

        return FinalSchema(
            dimensions=merged_dimensions,
            n_clusters_processed=len(cluster_results),
            total_questions_processed=total_questions,
            pipeline_time=pipeline_time
        )

    def _align_dimensions(
        self,
        cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> Dict[str, List[Tuple[int, DiscoveredDimensionSchema]]]:
        """
        Align dimensions across clusters using fuzzy name matching.

        Groups dimensions with similar names into canonical dimension groups.
        No global baseline - purely based on cluster-to-cluster similarity.

        Args:
            cluster_results: List of cluster discovery results

        Returns:
            Dict mapping canonical_name → list of (cluster_id, schema) tuples
        """
        logger.info("  Aligning dimensions across clusters...")

        dimension_groups = {}  # canonical_name → [(cluster_id, schema), ...]
        canonical_to_variants = {}  # canonical_name → set of variant names

        for result in cluster_results:
            cluster_id = result.cluster_id
            for dim_name, dim_schema in result.discovered_dimensions.items():
                # Find matching canonical dimension
                matched_canonical = None

                for canonical_name in dimension_groups.keys():
                    # Check if dim_name matches canonical name or any variant
                    if self._are_similar(dim_name, canonical_name):
                        matched_canonical = canonical_name
                        break

                    # Check against known variants
                    variants = canonical_to_variants.get(canonical_name, set())
                    for variant in variants:
                        if self._are_similar(dim_name, variant):
                            matched_canonical = canonical_name
                            break

                    if matched_canonical:
                        break

                if matched_canonical:
                    # Add to existing group
                    dimension_groups[matched_canonical].append((cluster_id, dim_schema))
                    canonical_to_variants[matched_canonical].add(dim_name)
                    if dim_name != matched_canonical:
                        logger.debug(f"    • '{dim_name}' (cluster {cluster_id}) → '{matched_canonical}'")
                else:
                    # Create new canonical group
                    dimension_groups[dim_name] = [(cluster_id, dim_schema)]
                    canonical_to_variants[dim_name] = {dim_name}
                    logger.debug(f"    • New dimension: '{dim_name}' (cluster {cluster_id})")

        return dimension_groups

    def _merge_dimension_group(
        self,
        canonical_name: str,
        cluster_schemas: List[Tuple[int, DiscoveredDimensionSchema]],
        all_cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> MergedDimensionSchema:
        """
        Merge a dimension group from multiple clusters.

        Args:
            canonical_name: Canonical dimension name
            cluster_schemas: List of (cluster_id, schema) tuples for this dimension
            all_cluster_results: All cluster results (for entity access)

        Returns:
            MergedDimensionSchema with merged schema and deduplicated entities
        """
        # Merge schema information (use longest hierarchy, combine descriptions/examples)
        hierarchies = [schema.hierarchy for _, schema in cluster_schemas]
        merged_hierarchy = max(hierarchies, key=len)

        descriptions = [schema.description for _, schema in cluster_schemas]
        unique_descriptions = list(dict.fromkeys(descriptions))
        merged_description = "; ".join(unique_descriptions)

        all_examples = []
        for _, schema in cluster_schemas:
            all_examples.extend(schema.examples)
        merged_examples = list(dict.fromkeys(all_examples))

        # Collect entities from all contributing clusters
        all_entities_with_sources = []  # [(entity, cluster_id), ...]

        # Build mapping of cluster_id → result for entity lookup
        cluster_id_to_result = {result.cluster_id: result for result in all_cluster_results}

        for cluster_id, _ in cluster_schemas:
            result = cluster_id_to_result[cluster_id]

            # Find entities for this dimension (check canonical name and variants)
            dimension_entities = []
            for dim_name, entities in result.entities.items():
                if self._are_similar(dim_name, canonical_name):
                    dimension_entities.extend(entities)

            for entity in dimension_entities:
                all_entities_with_sources.append((entity, cluster_id))

        # Deduplicate entities using HierarchicalEntity.matches_entity()
        unique_entities = self._deduplicate_hierarchical_entities(
            [entity for entity, _ in all_entities_with_sources]
        )

        # Track sources
        sources = DimensionSource()
        for entity in unique_entities:
            # Find which clusters contributed this entity (or similar variants)
            contributing_clusters = set()
            for other_entity, cluster_id in all_entities_with_sources:
                if entity.matches_entity(other_entity, self.similarity_threshold):
                    contributing_clusters.add(cluster_id)

            for cid in contributing_clusters:
                if cid not in sources.entity_counts_per_cluster:
                    sources.entity_counts_per_cluster[cid] = 0
                sources.entity_counts_per_cluster[cid] += 1
                sources.cluster_ids.add(cid)

        # Create alternative names set
        alternative_names = set()
        for cluster_id, schema in cluster_schemas:
            # Find dimension name variants from discovered_dimensions
            result = cluster_id_to_result[cluster_id]
            for dim_name in result.discovered_dimensions.keys():
                if self._are_similar(dim_name, canonical_name):
                    alternative_names.add(dim_name)

        return MergedDimensionSchema(
            hierarchy=merged_hierarchy,
            description=merged_description,
            examples=merged_examples,
            entities=unique_entities,
            total_entity_count=len(unique_entities),
            sources=sources,
            alternative_names=alternative_names
        )

    def _deduplicate_hierarchical_entities(
        self,
        entities: List[HierarchicalEntity]
    ) -> List[HierarchicalEntity]:
        """
        Deduplicate HierarchicalEntity objects using entity.matches_entity().

        Args:
            entities: List of HierarchicalEntity objects

        Returns:
            List of unique entities (deduplicated)
        """
        unique = []

        for entity in entities:
            # Check if this entity is a duplicate of any existing unique entity
            is_duplicate = False
            for existing in unique:
                if entity.matches_entity(existing, self.similarity_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity)

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
