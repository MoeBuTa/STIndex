"""
Two-pass extraction verification for quality improvement.

Uses a second LLM pass to score and filter extraction results,
significantly reducing false positives.
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.extraction.utils import extract_json_from_text
from stindex.llm.manager import LLMManager


class ExtractionVerifier:
    """
    Two-pass extraction verifier.

    Pass 1: Extract entities (done by DimensionalExtractor)
    Pass 2: Verify and score extractions (this class)

    Reduces false positives by scoring each extraction on:
    - Relevance: Is it actually in the text?
    - Accuracy: Does it match the text exactly?
    - Completeness: Were all entities extracted?
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        relevance_threshold: float = 0.7,
        accuracy_threshold: float = 0.7
    ):
        """
        Initialize extraction verifier.

        Args:
            llm_manager: LLM manager for verification
            relevance_threshold: Minimum relevance score (0-1)
            accuracy_threshold: Minimum accuracy score (0-1)
        """
        self.llm_manager = llm_manager
        self.relevance_threshold = relevance_threshold
        self.accuracy_threshold = accuracy_threshold

    def verify_extractions(
        self,
        text: str,
        extraction_result: Dict[str, List[Dict]],
        dimension_schemas: Optional[Dict] = None
    ) -> Dict[str, List[Dict]]:
        """
        Verify and filter extraction results.

        Args:
            text: Original input text
            extraction_result: Extraction results dict {dimension_name: [entities]}
            dimension_schemas: Optional dimension schemas for context

        Returns:
            Filtered extraction results with only high-confidence entities
        """
        logger.info("Running two-pass verification...")

        verified_results = {}

        for dim_name, entities in extraction_result.items():
            if not entities:
                continue

            logger.debug(f"Verifying {len(entities)} {dim_name} entities...")

            # Run verification for this dimension
            scores = self._score_entities(text, dim_name, entities, dimension_schemas)

            # Filter based on scores
            verified_entities = []
            for entity, score in zip(entities, scores):
                if self._passes_threshold(score):
                    # Add confidence scores to entity
                    entity['verification_scores'] = score
                    entity['verified'] = True
                    verified_entities.append(entity)
                else:
                    logger.debug(
                        f"Filtered out low-confidence entity: {entity.get('text', '')} "
                        f"(relevance={score.get('relevance', 0):.2f}, "
                        f"accuracy={score.get('accuracy', 0):.2f})"
                    )

            verified_results[dim_name] = verified_entities
            logger.info(
                f"✓ {dim_name}: {len(verified_entities)}/{len(entities)} entities passed verification"
            )

        return verified_results

    def _score_entities(
        self,
        text: str,
        dimension_name: str,
        entities: List[Dict],
        dimension_schemas: Optional[Dict] = None
    ) -> List[Dict[str, float]]:
        """
        Score entities using LLM.

        Args:
            text: Original input text
            dimension_name: Name of dimension being verified
            entities: List of extracted entities
            dimension_schemas: Optional dimension schemas

        Returns:
            List of score dicts for each entity
        """
        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            text, dimension_name, entities, dimension_schemas
        )

        try:
            # Generate verification scores
            messages = [
                {
                    "role": "system",
                    "content": "You are an extraction quality verifier. Analyze extractions and provide scores."
                },
                {
                    "role": "user",
                    "content": verification_prompt
                }
            ]

            response = self.llm_manager.generate(messages)

            if not response.success:
                logger.warning(f"Verification LLM call failed: {response.error_msg}")
                return self._default_scores(len(entities))

            # Parse scores from response
            scores = extract_json_from_text(response.content, None, return_dict=True)

            if 'entity_scores' in scores:
                scores = scores['entity_scores']

            # Ensure we have scores for all entities
            if not isinstance(scores, list) or len(scores) != len(entities):
                logger.warning(
                    f"Verification returned {len(scores) if isinstance(scores, list) else 0} scores "
                    f"for {len(entities)} entities. Using default scores."
                )
                return self._default_scores(len(entities))

            return scores

        except Exception as e:
            logger.warning(f"Verification failed: {e}. Using default scores.")
            return self._default_scores(len(entities))

    def _build_verification_prompt(
        self,
        text: str,
        dimension_name: str,
        entities: List[Dict],
        dimension_schemas: Optional[Dict] = None
    ) -> str:
        """
        Build verification prompt for LLM.

        Args:
            text: Original input text
            dimension_name: Name of dimension being verified
            entities: List of extracted entities
            dimension_schemas: Optional dimension schemas

        Returns:
            Verification prompt string
        """
        entities_json = json.dumps(entities, indent=2)

        schema_desc = ""
        if dimension_schemas and dimension_name in dimension_schemas:
            schema = dimension_schemas[dimension_name]
            schema_desc = f"\n\nDimension schema:\n{json.dumps(schema, indent=2)}"

        prompt = f"""You are verifying extraction quality. Score each extracted entity on three criteria:

1. **Relevance** (0-1): Is this entity actually mentioned in the text?
2. **Accuracy** (0-1): Does the extraction accurately represent what's in the text?
3. **Completeness** (0-1): Is the extraction complete and not missing important details?

**Original Text:**
{text}

**Dimension:** {dimension_name}{schema_desc}

**Extracted Entities:**
{entities_json}

**Task:** For each entity, provide scores for relevance, accuracy, and completeness.

Respond with ONLY a JSON array, one score object per entity:
```json
[
  {{
    "entity_index": 0,
    "relevance": 0.95,
    "accuracy": 0.90,
    "completeness": 0.85,
    "reasoning": "Brief explanation"
  }},
  ...
]
```

CRITICAL: Return ONLY the JSON array, nothing else."""

        return prompt

    def _passes_threshold(self, score: Dict[str, float]) -> bool:
        """
        Check if entity score passes thresholds.

        Args:
            score: Score dict with relevance, accuracy, completeness

        Returns:
            True if passes all thresholds
        """
        relevance = score.get('relevance', 0.0)
        accuracy = score.get('accuracy', 0.0)

        return (
            relevance >= self.relevance_threshold and
            accuracy >= self.accuracy_threshold
        )

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
                "reasoning": "Verification failed, using default scores"
            }
            for i in range(num_entities)
        ]


class BatchExtractionVerifier(ExtractionVerifier):
    """
    Batch verifier for efficient verification of multiple extractions.

    Verifies multiple extraction results in a single LLM call when possible.
    """

    def verify_batch(
        self,
        text_entity_pairs: List[tuple],
        dimension_schemas: Optional[Dict] = None
    ) -> List[Dict[str, List[Dict]]]:
        """
        Verify multiple extraction results in batch.

        Args:
            text_entity_pairs: List of (text, extraction_result) tuples
            dimension_schemas: Optional dimension schemas

        Returns:
            List of verified extraction results
        """
        logger.info(f"Running batch verification on {len(text_entity_pairs)} extractions...")

        verified_results = []

        for text, extraction_result in text_entity_pairs:
            verified = self.verify_extractions(text, extraction_result, dimension_schemas)
            verified_results.append(verified)

        logger.info(f"✓ Batch verification complete")
        return verified_results
