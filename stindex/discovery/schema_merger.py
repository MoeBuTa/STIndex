"""
Schema merger for combining entity lists from all clusters.

Merges entity lists from all clusters with hybrid deduplication:
- Fuzzy matching: character-level similarity (e.g., "fever" vs "Fever")
- Semantic matching: embedding-based similarity (e.g., "Clinical Sign" vs "Clinical Finding")

Pure cluster-level merging without global baseline.
"""

import difflib
from typing import Dict, List, Tuple, Set, Optional
from loguru import logger
import numpy as np

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

    Uses hybrid deduplication:
    - Fuzzy matching: difflib character-level similarity (threshold: 0.60)
    - Semantic matching: sentence embedding cosine similarity (threshold: 0.65)
    - LLM-based alignment: final pass to merge semantically related dimensions

    Aligns dimensions across clusters using lexical, semantic, and LLM-based similarity,
    then merges entities with deduplication.
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.60,
        semantic_threshold: float = 0.65,
        use_semantic: bool = True,
        use_llm_alignment: bool = True,
        llm_manager=None  # accepts an LLMClient instance; named llm_manager for backward compat
    ):
        """
        Initialize schema merger.

        Args:
            fuzzy_threshold: Character-level similarity threshold (0.0-1.0)
                Example: 0.60 means strings with ≥60% character similarity are duplicates
            semantic_threshold: Embedding cosine similarity threshold (0.0-1.0)
                Example: 0.65 means dimensions with ≥65% semantic similarity are merged
            use_semantic: Enable semantic deduplication (default: True)
                Set to False to use only fuzzy matching (faster, but misses semantic duplicates)
            use_llm_alignment: Enable LLM-based dimension alignment (default: True)
                Final pass to merge semantically related dimensions that fuzzy/semantic miss
            llm_manager: LLMClient instance for LLM-based alignment (optional)
                If None and use_llm_alignment=True, will be created on demand
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.use_semantic = use_semantic
        self.use_llm_alignment = use_llm_alignment
        self._llm_client = llm_manager  # accept LLMClient instance

        # Lazy-load sentence-transformers model
        self._embedding_model = None
        self._embedding_cache = {}  # Cache embeddings to avoid recomputation

        logger.info(
            f"Initialized SchemaMerger with fuzzy_threshold={fuzzy_threshold}, "
            f"semantic_threshold={semantic_threshold}, use_semantic={use_semantic}, "
            f"use_llm_alignment={use_llm_alignment}"
        )

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

        # Step 1: Align dimensions across clusters using fuzzy/semantic matching
        dimension_groups = self._align_dimensions(cluster_results)
        logger.info(f"  Initial alignment: {len(dimension_groups)} canonical dimensions")

        # Step 2: LLM-based alignment to merge semantically related dimensions
        if self.use_llm_alignment and len(dimension_groups) > 20:
            dimension_groups = self._llm_align_dimensions(dimension_groups, cluster_results)
            logger.info(f"  After LLM alignment: {len(dimension_groups)} canonical dimensions")

        # Step 3: Merge schemas and entities for each dimension
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
    ) -> Dict[str, List[Tuple[int, Optional[DiscoveredDimensionSchema]]]]:
        """
        Align dimensions across clusters using fuzzy name matching.

        Groups dimensions with similar names into canonical dimension groups.
        No global baseline - purely based on cluster-to-cluster similarity.

        IMPORTANT: Iterates over BOTH discovered_dimensions AND entities keys
        to ensure all extracted entities are included, even if their dimension
        wasn't explicitly discovered.

        Args:
            cluster_results: List of cluster discovery results

        Returns:
            Dict mapping canonical_name → list of (cluster_id, schema) tuples
            Note: schema may be None for dimensions only found in entities
        """
        logger.info("  Aligning dimensions across clusters...")

        dimension_groups = {}  # canonical_name → [(cluster_id, schema), ...]
        canonical_to_variants = {}  # canonical_name → set of variant names

        for result in cluster_results:
            cluster_id = result.cluster_id

            # Collect ALL dimension names from both discovered_dimensions AND entities
            all_dim_names = set(result.discovered_dimensions.keys()) | set(result.entities.keys())

            for dim_name in all_dim_names:
                # Get schema if it exists in discovered_dimensions
                dim_schema = result.discovered_dimensions.get(dim_name)

                # Skip if no entities for this dimension (only discovered, no data)
                if dim_name not in result.entities or not result.entities[dim_name]:
                    continue

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

    def _llm_align_dimensions(
        self,
        dimension_groups: Dict[str, List[Tuple[int, Optional[DiscoveredDimensionSchema]]]],
        cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> Dict[str, List[Tuple[int, Optional[DiscoveredDimensionSchema]]]]:
        """
        Use LLM to identify and merge semantically related dimensions.

        Domain-agnostic approach: asks LLM to group dimension names that represent
        the same or very similar concepts, regardless of the domain.

        Args:
            dimension_groups: Current dimension groups from fuzzy/semantic alignment
            cluster_results: All cluster results (for rebuilding groups)

        Returns:
            Updated dimension groups with LLM-aligned merges
        """
        import json
        import re

        dimension_names = list(dimension_groups.keys())
        logger.info(f"  Running LLM alignment on {len(dimension_names)} dimensions...")

        # Get or create LLM client
        llm_client = self._get_llm_client()
        if llm_client is None:
            logger.warning("  LLM client not available, skipping LLM alignment")
            return dimension_groups

        # Build domain-agnostic prompt
        prompt = self._build_llm_alignment_prompt(dimension_names)

        try:
            # Call LLM
            llm_output = llm_client.generate("", prompt)

            # Parse LLM response to get merge groups
            merge_groups = self._parse_llm_alignment_response(llm_output, dimension_names)

            if not merge_groups:
                logger.info("  LLM alignment: no additional merges suggested")
                return dimension_groups

            # Apply merges
            merged_dimension_groups = self._apply_llm_merges(
                dimension_groups, merge_groups, cluster_results
            )

            n_merged = len(dimension_groups) - len(merged_dimension_groups)
            logger.info(f"  LLM alignment merged {n_merged} dimension groups")

            return merged_dimension_groups

        except Exception as e:
            logger.warning(f"  LLM alignment failed: {e}, continuing without LLM merges")
            return dimension_groups

    def _get_llm_client(self):
        """Get or create LLM client for alignment."""
        if self._llm_client is not None:
            return self._llm_client

        try:
            from stindex.llm.base import create_client
            from stindex.utils.config import load_config_from_file

            config = load_config_from_file("extract")
            self._llm_client = create_client(config.get("llm", {}))
            return self._llm_client
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
            return None

    def _build_llm_alignment_prompt(self, dimension_names: List[str]) -> str:
        """
        Build domain-agnostic prompt for dimension alignment.

        The prompt asks LLM to identify groups of dimension names that represent
        the same or very similar concepts, without assuming any specific domain.
        """
        dimensions_list = "\n".join(f"- {name}" for name in sorted(dimension_names))

        return f"""You are analyzing a list of dimension names extracted from documents. Your task is to identify groups of dimension names that represent the SAME or VERY SIMILAR concepts and should be merged together.

## Dimension Names
{dimensions_list}

## Instructions
1. Identify groups of dimensions that represent the same underlying concept
2. Only group dimensions that are truly synonymous or near-synonymous
3. Do NOT group dimensions that are merely related but distinct
4. Choose the most general/canonical name for each group

## Examples of what TO merge:
- "Disease", "Medical Condition", "Medical_Diagnosis" → same concept
- "Anatomical Finding", "Anatomical Structure", "Body Part" → same concept
- "Treatment", "Treatment_Plan", "Medical Intervention" → same concept
- "Lab_Test", "Laboratory Study", "Laboratory Finding" → same concept

## Examples of what NOT to merge:
- "Treatment" and "Side Effect" → related but distinct concepts
- "Symptom" and "Disease" → related but distinct concepts
- "Patient Demographics" and "Social_History" → related but distinct concepts

## Output Format
Return a JSON array of merge groups. Each group is an object with:
- "canonical": the best name to use for the merged dimension
- "members": array of dimension names to merge into this canonical name

Only include groups with 2+ members. Dimensions not in any group stay as-is.

```json
[
  {{"canonical": "Medical Condition", "members": ["Disease", "Medical Condition", "Medical_Diagnosis"]}},
  {{"canonical": "Anatomical Structure", "members": ["Anatomical Finding", "Anatomical Structure", "Body Part"]}}
]
```

Return ONLY the JSON array, no other text."""

    def _parse_llm_alignment_response(
        self,
        llm_output: str,
        valid_dimensions: List[str]
    ) -> List[Dict[str, any]]:
        """
        Parse LLM response to extract merge groups.

        Args:
            llm_output: Raw LLM output
            valid_dimensions: List of valid dimension names (for validation)

        Returns:
            List of merge groups: [{"canonical": str, "members": [str, ...]}]
        """
        import json
        import re

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', llm_output)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON array
            json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', llm_output)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("  Could not extract JSON from LLM response")
                return []

        try:
            merge_groups = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"  Failed to parse LLM JSON: {e}")
            return []

        # Validate and filter merge groups
        valid_groups = []
        valid_dims_lower = {d.lower(): d for d in valid_dimensions}

        for group in merge_groups:
            if not isinstance(group, dict):
                continue

            canonical = group.get("canonical", "")
            members = group.get("members", [])

            if not canonical or not isinstance(members, list) or len(members) < 2:
                continue

            # Validate members exist in our dimension list (case-insensitive)
            valid_members = []
            for member in members:
                member_lower = member.lower()
                if member_lower in valid_dims_lower:
                    valid_members.append(valid_dims_lower[member_lower])
                elif member in valid_dimensions:
                    valid_members.append(member)

            if len(valid_members) >= 2:
                # Use first valid member as canonical if LLM's canonical isn't valid
                if canonical not in valid_dimensions:
                    canonical = valid_members[0]

                valid_groups.append({
                    "canonical": canonical,
                    "members": valid_members
                })

        return valid_groups

    def _apply_llm_merges(
        self,
        dimension_groups: Dict[str, List[Tuple[int, Optional[DiscoveredDimensionSchema]]]],
        merge_groups: List[Dict[str, any]],
        cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> Dict[str, List[Tuple[int, Optional[DiscoveredDimensionSchema]]]]:
        """
        Apply LLM-suggested merges to dimension groups.

        Args:
            dimension_groups: Current dimension groups
            merge_groups: LLM-suggested merge groups
            cluster_results: All cluster results

        Returns:
            Updated dimension groups with merges applied
        """
        # Build mapping: old_name → canonical_name
        rename_map = {}
        for group in merge_groups:
            canonical = group["canonical"]
            for member in group["members"]:
                if member != canonical:
                    rename_map[member] = canonical

        if not rename_map:
            return dimension_groups

        logger.info(f"  Applying {len(rename_map)} dimension renames from LLM alignment")
        for old_name, new_name in list(rename_map.items())[:10]:
            logger.debug(f"    • '{old_name}' → '{new_name}'")

        # Merge dimension groups
        new_dimension_groups = {}

        for dim_name, cluster_schemas in dimension_groups.items():
            # Get canonical name (either from rename_map or keep original)
            canonical_name = rename_map.get(dim_name, dim_name)

            if canonical_name not in new_dimension_groups:
                new_dimension_groups[canonical_name] = []

            # Add all cluster schemas to the canonical group
            new_dimension_groups[canonical_name].extend(cluster_schemas)

        return new_dimension_groups

    def _merge_dimension_group(
        self,
        canonical_name: str,
        cluster_schemas: List[Tuple[int, Optional[DiscoveredDimensionSchema]]],
        all_cluster_results: List[ClusterSchemaDiscoveryResult]
    ) -> MergedDimensionSchema:
        """
        Merge a dimension group from multiple clusters.

        Args:
            canonical_name: Canonical dimension name
            cluster_schemas: List of (cluster_id, schema) tuples for this dimension
                Note: schema may be None for dimensions only found in entities
            all_cluster_results: All cluster results (for entity access)

        Returns:
            MergedDimensionSchema with merged schema and deduplicated entities
        """
        # Filter out None schemas for schema merging
        valid_schemas = [(cid, schema) for cid, schema in cluster_schemas if schema is not None]

        # Merge schema information (use longest hierarchy, combine descriptions/examples)
        if valid_schemas:
            hierarchies = [schema.hierarchy for _, schema in valid_schemas]
            merged_hierarchy = max(hierarchies, key=len)

            descriptions = [schema.description for _, schema in valid_schemas]
            unique_descriptions = list(dict.fromkeys(descriptions))
            merged_description = "; ".join(unique_descriptions)

            all_examples = []
            for _, schema in valid_schemas:
                all_examples.extend(schema.examples)
            merged_examples = list(dict.fromkeys(all_examples))
        else:
            # No schema discovered - create default from dimension name
            merged_hierarchy = [canonical_name.lower().replace(' ', '_')]
            merged_description = f"Entities of type: {canonical_name}"
            merged_examples = []

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
                if entity.matches_entity(other_entity, self.fuzzy_threshold):
                    contributing_clusters.add(cluster_id)

            for cid in contributing_clusters:
                if cid not in sources.entity_counts_per_cluster:
                    sources.entity_counts_per_cluster[cid] = 0
                sources.entity_counts_per_cluster[cid] += 1
                sources.cluster_ids.add(cid)

        # Create alternative names set (from both discovered_dimensions AND entities)
        alternative_names = set()
        for cluster_id, schema in cluster_schemas:
            result = cluster_id_to_result[cluster_id]
            # Check discovered_dimensions
            for dim_name in result.discovered_dimensions.keys():
                if self._are_similar(dim_name, canonical_name):
                    alternative_names.add(dim_name)
            # Check entities keys (may have different dimension names)
            for dim_name in result.entities.keys():
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
                if entity.matches_entity(existing, self.fuzzy_threshold):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity)

        return unique

    def _get_embedding_model(self):
        """
        Lazy-load sentence-transformers model.

        Uses all-MiniLM-L6-v2: lightweight, fast, good for semantic similarity.
        Model size: ~90MB, inference: ~10ms per text on CPU.

        Returns:
            SentenceTransformer model
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence embedding model (all-MiniLM-L6-v2)...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded")
        return self._embedding_model

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get cached embedding for text.

        Args:
            text: Text to embed (normalized to lowercase)

        Returns:
            Embedding vector (384-dim)
        """
        normalized = text.lower()
        if normalized not in self._embedding_cache:
            model = self._get_embedding_model()
            self._embedding_cache[normalized] = model.encode(normalized, convert_to_numpy=True)
        return self._embedding_cache[normalized]

    def _are_semantically_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two strings are semantically similar using embeddings.

        Args:
            text1: First string
            text2: Second string

        Returns:
            True if cosine similarity >= semantic_threshold (0.82 by default)
        """
        if not self.use_semantic:
            return False

        # Get embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Compute cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        is_similar = cosine_sim >= self.semantic_threshold

        if is_similar:
            logger.debug(f"  Semantic match: '{text1}' ↔ '{text2}' (similarity: {cosine_sim:.3f})")

        return is_similar

    def _are_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two strings are similar using hybrid matching.

        Uses BOTH fuzzy (character-level) AND semantic (embedding-based) matching.
        Returns True if EITHER method indicates similarity.

        Fuzzy matching: Catches lexical variants (e.g., "fever" vs "Fever")
        Semantic matching: Catches semantic duplicates (e.g., "Clinical Sign" vs "Clinical Finding")

        Args:
            text1: First string (normalized to lowercase)
            text2: Second string (normalized to lowercase)

        Returns:
            True if similarity >= threshold (fuzzy OR semantic)
        """
        # Fuzzy matching (character-level similarity)
        fuzzy_similarity = difflib.SequenceMatcher(
            None,
            text1.lower(),
            text2.lower()
        ).ratio()
        fuzzy_match = fuzzy_similarity >= self.fuzzy_threshold

        # Semantic matching (embedding-based)
        semantic_match = self._are_semantically_similar(text1, text2)

        # Hybrid: match if EITHER fuzzy OR semantic
        is_match = fuzzy_match or semantic_match

        if is_match and not fuzzy_match:
            # Log when semantic matching catches something fuzzy matching missed
            logger.info(
                f"  📊 Semantic deduplication: '{text1}' ↔ '{text2}' "
                f"(fuzzy: {fuzzy_similarity:.2f}, semantic: yes)"
            )

        return is_match
