"""
Two-pass extraction reflection for quality improvement with context-aware reasoning.

Uses a second LLM pass to reflect on and score extraction results,
significantly reducing false positives through context-aware evaluation.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.utils import extract_json_from_text
from stindex.llm.base import LLMClient
from stindex.llm.prompts.reflection import ReflectionPrompt


class ExtractionReflector:
    """
    Two-pass extraction reflector with context-aware reasoning.

    Pass 1: Extract entities (done by DimensionalExtractor)
    Pass 2: Reflect on and score extractions (this class)

    Reduces false positives by scoring each extraction on:
    - Relevance: Is it actually in the text?
    - Accuracy: Does it match the text exactly?
    - Completeness: Were all details extracted?
    - Consistency: Does it align with context and prior extractions?

    Context-aware features:
    - Uses prior temporal/spatial references for consistency checks
    - Considers document metadata (publication date, source location)
    - Evaluates relative expressions against anchor dates
    - Checks spatial disambiguation against known locations
    """

    def __init__(
        self,
        llm_client: LLMClient,
        relevance_threshold: float = 0.7,
        accuracy_threshold: float = 0.7,
        consistency_threshold: float = 0.6,
        extraction_context: Optional[ExtractionContext] = None
    ):
        """
        Initialize extraction reflector.

        Args:
            llm_client: LLM client for reflection
            relevance_threshold: Minimum relevance score (0-1)
            accuracy_threshold: Minimum accuracy score (0-1)
            consistency_threshold: Minimum consistency score (0-1)
            extraction_context: Optional ExtractionContext for context-aware reflection
        """
        self.llm_client = llm_client
        self.relevance_threshold = relevance_threshold
        self.accuracy_threshold = accuracy_threshold
        self.consistency_threshold = consistency_threshold
        self.extraction_context = extraction_context

    def reflect_on_extractions(
        self,
        text: str,
        extraction_result: Dict[str, List[Dict]],
        dimension_schemas: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Reflect on and filter extraction results using context-aware reasoning.

        Args:
            text: Original input text
            extraction_result: Extraction results dict {dimension_name: [entities]}
            dimension_schemas: Optional dimension schemas for context

        Returns:
            Filtered extraction results with only high-confidence entities
        """
        logger.info("Running two-pass reflection with context-aware reasoning...")

        if self.extraction_context:
            logger.debug(
                f"Context: {len(self.extraction_context.prior_temporal_refs)} temporal refs, "
                f"{len(self.extraction_context.prior_spatial_refs)} spatial refs"
            )

        reflected_results = {}

        for dim_name, entities in extraction_result.items():
            if not entities:
                continue

            logger.debug(f"Reflecting on {len(entities)} {dim_name} entities...")

            # Run reflection for this dimension
            scores = self._score_entities(text, dim_name, entities, dimension_schemas)

            # Filter based on scores
            reflected_entities = []
            filtered_count = 0

            for entity, score in zip(entities, scores):
                if self._passes_threshold(score):
                    # Add reflection scores to entity
                    entity['reflection_scores'] = score
                    entity['reflected'] = True
                    reflected_entities.append(entity)
                else:
                    filtered_count += 1
                    logger.debug(
                        f"Filtered out low-confidence entity: {entity.get('text', '')} "
                        f"(relevance={score.get('relevance', 0):.2f}, "
                        f"accuracy={score.get('accuracy', 0):.2f}, "
                        f"consistency={score.get('consistency', 0):.2f}) - "
                        f"Reason: {score.get('reasoning', 'N/A')[:80]}..."
                    )

            reflected_results[dim_name] = reflected_entities
            logger.info(
                f"✓ {dim_name}: {len(reflected_entities)}/{len(entities)} entities passed reflection "
                f"({filtered_count} filtered)"
            )

        return reflected_results

    def _score_entities(
        self,
        text: str,
        dimension_name: str,
        entities: List[Dict],
        dimension_schemas: Optional[Dict] = None
    ) -> List[Dict[str, float]]:
        """
        Score entities using context-aware LLM reflection.

        Args:
            text: Original input text
            dimension_name: Name of dimension being reflected on
            entities: List of extracted entities
            dimension_schemas: Optional dimension schemas

        Returns:
            List of score dicts for each entity
        """
        # Build context-aware reflection prompt
        prompt_builder = ReflectionPrompt(
            text=text,
            dimension_name=dimension_name,
            entities=entities,
            dimension_schemas=dimension_schemas,
            extraction_context=self.extraction_context  # Pass context
        )

        try:
            # Generate reflection scores
            messages = prompt_builder.build_messages()
            _sys = next((m["content"] for m in messages if m["role"] == "system"), "")
            _usr = next((m["content"] for m in messages if m["role"] == "user"), "")
            content = self.llm_client.generate(_sys, _usr)

            # Parse scores from response
            scores = extract_json_from_text(content, None, return_dict=True)

            # Debug: Log what we got
            logger.debug(f"Reflection response type: {type(scores)}")

            # Handle different response formats
            if isinstance(scores, dict):
                # If LLM wrapped array in an object, try to extract it
                if 'entity_scores' in scores:
                    scores = scores['entity_scores']
                    logger.debug("Extracted scores from 'entity_scores' key")
                elif 'scores' in scores:
                    scores = scores['scores']
                    logger.debug("Extracted scores from 'scores' key")
                elif 'results' in scores:
                    scores = scores['results']
                    logger.debug("Extracted scores from 'results' key")
                else:
                    # Try to find any list value
                    for key, value in scores.items():
                        if isinstance(value, list) and len(value) > 0:
                            scores = value
                            logger.debug(f"Extracted scores from key: {key}")
                            break

            # Ensure we have scores for all entities
            if not isinstance(scores, list):
                logger.warning(
                    f"Reflection returned {type(scores).__name__} instead of list. Using default scores."
                )
                logger.debug(f"Raw response (first 500 chars): {content[:500]}")
                return self._default_scores(len(entities))

            if len(scores) != len(entities):
                logger.warning(
                    f"Reflection returned {len(scores)} scores for {len(entities)} entities. Using default scores."
                )
                logger.debug(f"Raw response (first 500 chars): {content[:500]}")
                return self._default_scores(len(entities))

            # Log reflection reasoning for first few entities
            for i, score in enumerate(scores[:3]):  # First 3
                entity_text = entities[i].get('text', '') if i < len(entities) else ''
                reasoning = score.get('reasoning', 'N/A')[:100]
                logger.debug(f"Reflection on '{entity_text}': {reasoning}...")

            return scores

        except Exception as e:
            logger.warning(f"Reflection failed: {e}. Using default scores.")
            return self._default_scores(len(entities))

    def _passes_threshold(self, score: Dict[str, float]) -> bool:
        """
        Check if entity score passes thresholds.

        All three criteria must pass:
        - Relevance >= threshold
        - Accuracy >= threshold
        - Consistency >= threshold (if context-aware)

        Args:
            score: Score dict with relevance, accuracy, completeness, consistency

        Returns:
            True if passes all thresholds
        """
        relevance = score.get('relevance', 0.0)
        accuracy = score.get('accuracy', 0.0)
        consistency = score.get('consistency', 1.0)  # Default to 1.0 if not present

        # Base thresholds
        passes = (
            relevance >= self.relevance_threshold and
            accuracy >= self.accuracy_threshold
        )

        # Add consistency check if context-aware
        if self.extraction_context:
            passes = passes and consistency >= self.consistency_threshold

        return passes

    def _default_scores(self, num_entities: int) -> List[Dict[str, float]]:
        """
        Generate default scores (assume all pass).

        Args:
            num_entities: Number of entities to score

        Returns:
            List of default score dicts
        """
        return [
            {
                "entity_index": i,
                "relevance": 1.0,
                "accuracy": 1.0,
                "completeness": 1.0,
                "consistency": 1.0,
                "reasoning": "Reflection failed, using default scores"
            }
            for i in range(num_entities)
        ]


class BatchExtractionReflector(ExtractionReflector):
    """
    Batch reflector for efficient reflection on multiple extractions.

    Reflects on multiple extraction results in sequence, maintaining context
    across batches when extraction_context is provided.
    """

    def reflect_batch(
        self,
        text_entity_pairs: List[tuple],
        dimension_schemas: Optional[Dict] = None
    ) -> List[Dict[str, List[Dict]]]:
        """
        Reflect on multiple extraction results in batch.

        Args:
            text_entity_pairs: List of (text, extraction_result) tuples
            dimension_schemas: Optional dimension schemas

        Returns:
            List of reflected extraction results
        """
        logger.info(f"Running batch reflection on {len(text_entity_pairs)} extractions...")

        reflected_results = []

        for i, (text, extraction_result) in enumerate(text_entity_pairs):
            logger.debug(f"Batch reflection {i+1}/{len(text_entity_pairs)}...")
            reflected = self.reflect_on_extractions(text, extraction_result, dimension_schemas)
            reflected_results.append(reflected)

        logger.info(f"✓ Batch reflection complete")
        return reflected_results
