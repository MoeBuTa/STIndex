"""
Cluster-level schema discoverer with integrated entity extraction.

Discovers dimensional schemas from questions within a SINGLE cluster,
then extracts entities using the discovered schema.
"""

from typing import Dict, List
import json
import random
from loguru import logger

from stindex.llm.manager import LLMManager
from stindex.llm.prompts.initial_schema_discovery import ClusterSchemaPrompt
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

    Two-phase approach:
    1. Discovery: Sample questions from cluster → discover dimensions
    2. Extraction: Process all questions in batches → extract entities

    Returns Pydantic models for type safety.
    """

    def __init__(
        self,
        llm_config: Dict,
        output_dir: str = None,
        batch_size: int = 50,
        cot_logger: CoTLogger = None
    ):
        """
        Initialize cluster schema discoverer.

        Args:
            llm_config: LLM configuration dict (provider, model, etc.)
            output_dir: Output directory for CoT logging (optional, deprecated if cot_logger provided)
            batch_size: Batch size for entity extraction (default: 50)
            cot_logger: Shared CoT logger instance (optional, preferred over output_dir)
        """
        self.llm_manager = LLMManager(llm_config)
        # Use provided cot_logger, or create new one from output_dir
        self.cot_logger = cot_logger if cot_logger else (CoTLogger(output_dir) if output_dir else None)
        self.batch_size = batch_size

    def discover_and_extract(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        n_samples_for_discovery: int = 20,
        allow_new_dimensions: bool = True
    ) -> ClusterSchemaDiscoveryResult:
        """
        Discover dimensions and extract entities for a single cluster.

        Two-phase approach:
        1. Discovery: Sample questions → discover dimensions
        2. Extraction: All questions in batches → extract entities

        Args:
            cluster_id: Cluster ID
            cluster_questions: All questions in this cluster
            n_samples_for_discovery: Number of questions to sample for discovery (default: 20)
            allow_new_dimensions: Allow discovering new dimensions during extraction

        Returns:
            ClusterSchemaDiscoveryResult with discovered dimensions and extracted entities
        """
        import time
        start_time = time.time()

        logger.info(f"Cluster {cluster_id}: Processing {len(cluster_questions)} questions")

        # Phase 1: Discovery from sample
        logger.info(f"Cluster {cluster_id}: Phase 1 - Discovering dimensions")
        sample_questions = self._sample_questions(cluster_questions, n_samples_for_discovery)
        discovered_dims, reasoning = self._discover_dimensions_for_cluster(cluster_id, sample_questions)

        logger.info(f"Cluster {cluster_id}: Discovered {len(discovered_dims)} dimensions")
        for dim_name, dim_schema in discovered_dims.items():
            hierarchy_str = ' → '.join(dim_schema.hierarchy)
            logger.info(f"  • {dim_name}: {hierarchy_str}")

        # Phase 2: Extraction from all questions
        logger.info(f"Cluster {cluster_id}: Phase 2 - Extracting entities")
        entities = self._extract_entities(
            cluster_id,
            cluster_questions,
            discovered_dims,
            allow_new_dimensions
        )

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

    def _sample_questions(
        self,
        questions: List[str],
        n_samples: int
    ) -> List[str]:
        """
        Sample questions for discovery phase.

        Args:
            questions: All questions in cluster
            n_samples: Number of samples to draw

        Returns:
            Sampled questions (or all if fewer than n_samples)
        """
        if len(questions) <= n_samples:
            return questions

        return random.sample(questions, n_samples)

    def _discover_dimensions_for_cluster(
        self,
        cluster_id: int,
        sample_questions: List[str]
    ) -> tuple[Dict[str, DiscoveredDimensionSchema], str]:
        """
        Discover dimensions from sample questions in a cluster.

        Args:
            cluster_id: Cluster ID
            sample_questions: Sample questions for discovery

        Returns:
            Tuple of (discovered_dimensions, reasoning)
        """
        logger.info(f"Cluster {cluster_id}: Discovering from {len(sample_questions)} sample questions")

        # Create prompt for schema discovery
        prompt = ClusterSchemaPrompt(
            predefined_dimensions=[],  # No predefined dimensions
            cluster_id=cluster_id
        )

        # Build messages
        messages = prompt.build_messages(sample_questions)

        logger.debug(f"Cluster {cluster_id}: System prompt length: {len(messages[0]['content'])} chars")
        logger.debug(f"Cluster {cluster_id}: User prompt length: {len(messages[1]['content'])} chars")

        # Get LLM response
        try:
            response = self.llm_manager.generate(messages)
            logger.info(f"Cluster {cluster_id}: ✓ LLM response received")

        except Exception as e:
            logger.error(f"Cluster {cluster_id}: ✗ LLM generation failed: {e}")
            raise

        # Parse discovered dimensions
        try:
            result = prompt.parse_response(response.content)
            discovered_dims_dict = result['schemas']
            reasoning = result['reasoning']

            # Log CoT
            if self.cot_logger:
                self.cot_logger.log_cluster_discovery(
                    cluster_id=cluster_id,
                    reasoning=reasoning,
                    raw_response=result['raw_response'],
                    n_dimensions=len(discovered_dims_dict)
                )

            # Convert to Pydantic models
            discovered_dims = {}
            for dim_name, dim_dict in discovered_dims_dict.items():
                discovered_dims[dim_name] = DiscoveredDimensionSchema(
                    hierarchy=dim_dict['hierarchy'],
                    description=dim_dict['description'],
                    examples=dim_dict.get('examples', [])
                )

            logger.info(f"Cluster {cluster_id}: ✓ Parsed {len(discovered_dims)} dimensions")

            return discovered_dims, reasoning

        except Exception as e:
            logger.error(f"Cluster {cluster_id}: ✗ Failed to parse LLM response: {e}")
            logger.debug(f"  Raw response: {response.content[:500]}")
            raise

    def _extract_entities(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        discovered_dimensions: Dict[str, DiscoveredDimensionSchema],
        allow_new_dimensions: bool = True
    ) -> Dict[str, List[HierarchicalEntity]]:
        """
        Extract entities from all questions using discovered dimensions.

        Args:
            cluster_id: Cluster ID
            cluster_questions: All questions in cluster
            discovered_dimensions: Discovered dimensional schemas
            allow_new_dimensions: Allow discovering new dimensions during extraction

        Returns:
            Map of dimension name → list of extracted entities
        """
        # Use ClusterEntityExtractor for batch processing
        extractor = ClusterEntityExtractor(
            global_dimensions=discovered_dimensions,  # Using discovered dims as "global" for this cluster
            llm_config=self.llm_manager.config,
            batch_size=self.batch_size,
            allow_new_dimensions=allow_new_dimensions,
            cot_logger=self.cot_logger
        )

        # Extract entities - returns dict with 'entities' key
        result = extractor.extract_from_cluster(
            cluster_questions=cluster_questions,
            cluster_id=cluster_id
        )

        # Extract just the entities field
        return result['entities']
