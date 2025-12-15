"""
Cluster-level schema discoverer with unified entity extraction.

Discovers dimensional schemas and extracts entities in a single unified process.
No separate discovery phase - discovery happens during first batch extraction.
"""

from typing import Dict, List, Optional
import json
from loguru import logger

from stindex.llm.manager import LLMManager
from stindex.discovery.cot_logger import CoTLogger
from stindex.discovery.models import (
    DiscoveredDimensionSchema,
    ClusterSchemaDiscoveryResult,
    HierarchicalEntity
)
from stindex.discovery.cluster_entity_extractor import ClusterEntityExtractor


class ClusterSchemaDiscoverer:
    """
    Discover dimensions and extract entities for a single cluster.

    Unified approach:
    - First batch: Discovers schema + extracts entities (adaptive size)
    - Subsequent batches: Refines schema + extracts entities (with decay)

    Returns Pydantic models for type safety.
    """

    def __init__(
        self,
        llm_manager: LLMManager = None,
        llm_config: Dict = None,
        output_dir: str = None,
        batch_size: int = 50,
        cot_logger: CoTLogger = None
    ):
        """
        Initialize cluster schema discoverer.

        Args:
            llm_manager: Shared LLM manager instance (preferred, for engine reuse)
            llm_config: LLM configuration dict (fallback, creates new manager)
            output_dir: Output directory for CoT logging (optional, deprecated if cot_logger provided)
            batch_size: Batch size for entity extraction (default: 50)
            cot_logger: Shared CoT logger instance (optional, preferred over output_dir)
        """
        # Use provided llm_manager or create new one from config
        if llm_manager is not None:
            self.llm_manager = llm_manager
        elif llm_config is not None:
            self.llm_manager = LLMManager(llm_config)
        else:
            raise ValueError("Must provide either llm_manager or llm_config")

        # Use provided cot_logger, or create new one from output_dir
        self.cot_logger = cot_logger if cot_logger else (CoTLogger(output_dir) if output_dir else None)
        self.batch_size = batch_size

    def _calculate_adaptive_batch_size(
        self,
        cluster_size: int,
        min_size: int = 50,
        max_size: int = 150,
        ratio: float = 0.10
    ) -> int:
        """
        Calculate adaptive first batch size based on cluster size.

        Formula: max(min_size, min(max_size, cluster_size * ratio))

        Args:
            cluster_size: Total number of questions in cluster
            min_size: Minimum batch size (default: 50)
            max_size: Maximum batch size (default: 150)
            ratio: Percentage of cluster to use (default: 0.10 = 10%)

        Returns:
            Adaptive batch size

        Examples:
            cluster_size=100 → 50 (10% = 10, but min is 50)
            cluster_size=500 → 50 (10% = 50, within range)
            cluster_size=2000 → 150 (10% = 200, but max is 150)
        """
        adaptive_size = int(cluster_size * ratio)
        return max(min_size, min(max_size, adaptive_size))

    def discover_and_extract(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        allow_new_dimensions: bool = True,
        adaptive_first_batch: bool = True,
        first_batch_min: int = 50,
        first_batch_max: int = 150,
        first_batch_ratio: float = 0.10,
        decay_config: Optional[Dict] = None
    ) -> ClusterSchemaDiscoveryResult:
        """
        Discover dimensions and extract entities for a single cluster.

        Unified approach:
        - First batch: Discovers schema + extracts entities (adaptive size)
        - Subsequent batches: Refines schema + extracts entities (with decay)

        Args:
            cluster_id: Cluster ID
            cluster_questions: All questions in this cluster
            allow_new_dimensions: Allow discovering new dimensions during extraction
            adaptive_first_batch: Use adaptive sizing for first batch (default: True)
            first_batch_min: Minimum first batch size (default: 50)
            first_batch_max: Maximum first batch size (default: 150)
            first_batch_ratio: First batch as ratio of cluster size (default: 0.10 = 10%)
            decay_config: Decay thresholds config (optional, uses defaults if not provided)

        Returns:
            ClusterSchemaDiscoveryResult with discovered dimensions and extracted entities
        """
        import time
        start_time = time.time()

        logger.info(f"Cluster {cluster_id}: Processing {len(cluster_questions)} questions")

        # Calculate adaptive first batch size
        if adaptive_first_batch:
            first_batch_size = self._calculate_adaptive_batch_size(
                cluster_size=len(cluster_questions),
                min_size=first_batch_min,
                max_size=first_batch_max,
                ratio=first_batch_ratio
            )
            logger.info(f"Cluster {cluster_id}: Adaptive first batch size = {first_batch_size}")
        else:
            first_batch_size = self.batch_size
            logger.info(f"Cluster {cluster_id}: Using standard batch size = {first_batch_size}")

        # Unified extraction (discovery + extraction in one process)
        logger.info(f"Cluster {cluster_id}: Starting unified discovery + extraction")
        result = self._extract_entities_unified(
            cluster_id=cluster_id,
            cluster_questions=cluster_questions,
            first_batch_size=first_batch_size,
            allow_new_dimensions=allow_new_dimensions,
            decay_config=decay_config
        )

        discovered_dims = result['discovered_dimensions']
        entities = result['entities']
        reasoning = result.get('reasoning', '')

        # Log discovered dimensions
        logger.info(f"Cluster {cluster_id}: Discovered {len(discovered_dims)} dimensions")
        for dim_name, dim_schema in discovered_dims.items():
            hierarchy_str = ' → '.join(dim_schema.hierarchy)
            logger.info(f"  • {dim_name}: {hierarchy_str}")

        # Calculate statistics
        entity_stats = {dim: len(ents) for dim, ents in entities.items()}
        logger.info(f"Cluster {cluster_id}: Extracted entities - {entity_stats}")

        extraction_time = time.time() - start_time
        logger.info(f"Cluster {cluster_id}: Complete in {extraction_time:.2f}s")

        return ClusterSchemaDiscoveryResult(
            cluster_id=cluster_id,
            n_questions=len(cluster_questions),
            discovered_dimensions=discovered_dims,
            entities=entities,
            reasoning=reasoning,
            extraction_time=extraction_time
        )

    def _extract_entities_unified(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        first_batch_size: int,
        allow_new_dimensions: bool = True,
        decay_config: Optional[Dict] = None
    ) -> Dict:
        """
        Extract entities using unified approach (discovery + extraction together).

        First batch discovers schema + extracts entities.
        Subsequent batches refine schema + extract entities (with decay).

        Args:
            cluster_id: Cluster ID
            cluster_questions: All questions in cluster
            first_batch_size: Size of first batch (adaptive)
            allow_new_dimensions: Allow discovering new dimensions during extraction
            decay_config: Decay thresholds config (optional)

        Returns:
            Dict with keys:
                - discovered_dimensions: Final discovered dimensions
                - entities: Extracted entities
                - reasoning: Discovery reasoning (from first batch)
        """
        # Use ClusterEntityExtractor for unified processing
        # Start with empty global_dimensions - discovery from scratch
        extractor = ClusterEntityExtractor(
            global_dimensions={},  # Empty - will be discovered in first batch
            llm_manager=self.llm_manager,  # Pass shared manager instance
            batch_size=self.batch_size,
            first_batch_size=first_batch_size,
            allow_new_dimensions=allow_new_dimensions,
            decay_config=decay_config,
            cot_logger=self.cot_logger
        )

        # Extract entities - returns dict with 'entities' and 'discovered_dimensions' keys
        result = extractor.extract_from_cluster(
            cluster_questions=cluster_questions,
            cluster_id=cluster_id
        )

        return result
