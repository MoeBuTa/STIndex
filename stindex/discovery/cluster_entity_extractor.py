"""
Cluster entity extractor for schema discovery.

Extracts entities from all questions in a cluster using dimensional schemas,
returning HierarchicalEntity Pydantic models.
"""

from typing import Dict, List, Set
import json
from loguru import logger

from stindex.llm.manager import LLMManager
from stindex.llm.prompts.entity_extraction_with_discovery import ClusterEntityPrompt
from stindex.discovery.cot_logger import CoTLogger
from stindex.discovery.models import DiscoveredDimensionSchema, HierarchicalEntity


class ClusterEntityExtractor:
    """
    Extract entities from one cluster using discovered dimensional schemas.

    Returns HierarchicalEntity Pydantic models with validated hierarchy values.
    """

    def __init__(
        self,
        global_dimensions: Dict[str, DiscoveredDimensionSchema],
        llm_config: Dict,
        batch_size: int = 50,
        output_dir: str = None,
        allow_new_dimensions: bool = True,
        max_retries: int = 3,
        cot_logger: 'CoTLogger' = None
    ):
        """
        Initialize cluster entity extractor.

        Args:
            global_dimensions: Dimensional schemas (either discovered globally or per-cluster)
                Example: {'symptom': DiscoveredDimensionSchema(...), ...}
            llm_config: LLM configuration dict (provider, model, etc.)
            batch_size: Number of questions to process per LLM call (default: 50)
            output_dir: Output directory for CoT logging (optional, deprecated if cot_logger provided)
            allow_new_dimensions: Allow discovering new dimensions during extraction (default: True)
            max_retries: Maximum retry attempts for JSON parsing errors (default: 3)
            cot_logger: Shared CoT logger instance (optional, preferred over output_dir)
        """
        self.global_dimensions = global_dimensions
        self.llm_manager = LLMManager(llm_config)
        self.batch_size = batch_size
        # Use provided cot_logger, or create new one from output_dir
        self.cot_logger = cot_logger if cot_logger else (CoTLogger(output_dir) if output_dir else None)
        self.allow_new_dimensions = allow_new_dimensions
        self.max_retries = max_retries

        # Entity tracking per dimension (list of HierarchicalEntity)
        self.entity_lists: Dict[str, List[HierarchicalEntity]] = {
            dim_name: [] for dim_name in global_dimensions.keys()
        }

        logger.info(f"Initialized extractor with {len(global_dimensions)} dimensions (batch_size={batch_size})")
        for dim_name in sorted(global_dimensions.keys()):
            hierarchy = ' → '.join(global_dimensions[dim_name].hierarchy)
            logger.debug(f"  • {dim_name}: {hierarchy}")

    def extract_from_cluster(
        self,
        cluster_questions: List[str],
        cluster_id: int
    ) -> Dict[str, List[HierarchicalEntity]]:
        """
        Extract entities from all questions in cluster.

        Args:
            cluster_questions: Questions in this cluster
            cluster_id: Cluster identifier

        Returns:
            Dict mapping dimension name → list of HierarchicalEntity objects
        """
        logger.info(f"Extracting entities from Cluster {cluster_id} ({len(cluster_questions)} questions, batch_size={self.batch_size})")

        # Convert dimensions to DimensionConfig format for prompt
        dimension_configs = self._convert_to_dimension_configs(self.global_dimensions)

        # Create case-insensitive mapping for dimension names
        # Handle both underscores and spaces (e.g., "Developmental_Biology" → "Developmental Biology")
        dim_name_mapping = {}
        for dim_name in self.global_dimensions.keys():
            # Map both lowercase and with underscores replaced
            normalized_key = dim_name.lower().replace(' ', '_')
            dim_name_mapping[normalized_key] = dim_name
            # Also map with spaces
            dim_name_mapping[dim_name.lower()] = dim_name

        # Create extraction prompt (reused for all batches)
        prompt = ClusterEntityPrompt(
            dimensions=dimension_configs,
            extraction_context=None,  # Context added per batch
            allow_new_dimensions=self.allow_new_dimensions,  # Allow discovering new dimensions
            cluster_id=cluster_id
        )

        # Process questions in batches
        total_batches = (len(cluster_questions) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(cluster_questions))
            batch_questions = cluster_questions[start_idx:end_idx]

            # Build context: dimension names + current entity lists
            context_str = self._build_context()

            # Build messages for this batch
            messages = self._build_batch_messages(
                prompt=prompt,
                batch_questions=batch_questions,
                context_str=context_str,
                batch_idx=batch_idx,
                total_batches=total_batches
            )

            # Get LLM response with retry on JSON parsing errors
            result = None
            last_error = None

            for attempt in range(self.max_retries):
                try:
                    response = self.llm_manager.generate(messages)
                    result = prompt.parse_response_with_discovery(response.content)
                    # Success - break out of retry loop
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        logger.warning(f"  ⚠ Cluster {cluster_id} Batch {batch_idx+1}: JSON parsing error (attempt {attempt+1}/{self.max_retries}), retrying...")
                    else:
                        logger.error(f"  ✗ Cluster {cluster_id} Batch {batch_idx+1}: Failed after {self.max_retries} attempts: {e}")
                        # Continue with empty result
                        result = None
                        break

            if result is None:
                # All retries failed - skip this batch
                logger.warning(f"  ⚠ Cluster {cluster_id} Batch {batch_idx+1}: Skipping batch due to parsing errors")
                continue

            # Extract entity-first format
            entity_first = result['entities']
            reasoning = result['reasoning']
            new_dimensions = result.get('new_dimensions', {})

            # Log CoT
            if self.cot_logger:
                self.cot_logger.log_cluster_batch(
                    cluster_id=cluster_id,
                    batch_idx=batch_idx,
                    reasoning=reasoning,
                    raw_response=result['raw_response'],
                    n_entities=len(entity_first)
                )

            # Handle newly discovered dimensions
            if new_dimensions and self.allow_new_dimensions:
                for dim_name, dim_schema in new_dimensions.items():
                    if dim_name not in self.global_dimensions:
                        logger.info(f"  📝 Cluster {cluster_id} Batch {batch_idx+1}: Discovered new dimension '{dim_name}'")
                        logger.info(f"     Hierarchy: {' → '.join(dim_schema.get('hierarchy', []))}")

                        # Add to global dimensions - create DiscoveredDimensionSchema
                        self.global_dimensions[dim_name] = DiscoveredDimensionSchema(
                            hierarchy=dim_schema.get('hierarchy', []),
                            description=dim_schema.get('description', ''),
                            examples=dim_schema.get('example_entities', [])
                        )

                        # Initialize tracking for new dimension
                        self.entity_lists[dim_name] = []

                        # Update dimension name mapping
                        normalized_key = dim_name.lower().replace(' ', '_')
                        dim_name_mapping[normalized_key] = dim_name
                        dim_name_mapping[dim_name.lower()] = dim_name

            # Convert entity-first to dimension-first for aggregation
            dimension_first = self._convert_to_dimension_first(entity_first)

            # Update entity lists (preserve full entity objects with hierarchy)
            for dim_name, entities in dimension_first.items():
                # Normalize dimension name (case-insensitive, handle underscores)
                normalized_key = dim_name.lower().replace(' ', '_')
                normalized_dim = dim_name_mapping.get(normalized_key)
                if not normalized_dim:
                    normalized_key = dim_name.lower()
                    normalized_dim = dim_name_mapping.get(normalized_key, dim_name)

                if normalized_dim not in self.entity_lists:
                    logger.warning(f"Unexpected dimension '{dim_name}' in extraction result (Batch {batch_idx+1})")
                    self.entity_lists[normalized_dim] = []

                for entity in entities:
                    if not isinstance(entity, dict):
                        continue

                    entity_text = entity.get('text', '')
                    if entity_text and entity_text.strip():
                        # Extract hierarchy values (all fields except 'text' and 'dimension')
                        # Convert all values to strings to handle LLM returning integers/other types
                        hierarchy_values = {
                            k: str(v) if v is not None else None
                            for k, v in entity.items()
                            if k not in ['text', 'dimension']
                        }

                        # Create HierarchicalEntity object
                        entity_obj = HierarchicalEntity(
                            text=entity_text,
                            dimension=normalized_dim,
                            hierarchy_values=hierarchy_values
                        )

                        # Deduplicate using matches_entity()
                        if not any(e.matches_entity(entity_obj, similarity_threshold=0.85)
                                   for e in self.entity_lists[normalized_dim]):
                            self.entity_lists[normalized_dim].append(entity_obj)

            # Log progress
            total_entities = sum(len(entities) for entities in self.entity_lists.values())
            logger.info(f"  Progress: Batch {batch_idx+1}/{total_batches} ({end_idx}/{len(cluster_questions)} questions), {total_entities} unique entities")

        # Sort entities by text field for consistent output
        entities_lists = {
            dim: sorted(entities, key=lambda x: x.text.lower())
            for dim, entities in self.entity_lists.items()
        }

        entity_counts = {
            dim: len(entities)
            for dim, entities in entities_lists.items()
        }

        logger.info(f"✓ Extraction complete for Cluster {cluster_id}")
        for dim_name in sorted(entity_counts.keys()):
            logger.info(f"  • {dim_name}: {entity_counts[dim_name]} unique entities")

        # Log final cluster schema
        logger.info(f"  Final cluster schema: {len(self.global_dimensions)} dimensions")

        return {
            'cluster_id': cluster_id,
            'n_questions': len(cluster_questions),
            'entities': entities_lists,
            'entity_counts': entity_counts,
            'final_cluster_schema': self.global_dimensions  # Include evolved schema
        }

    def _build_context(self) -> str:
        """
        Build simple context string showing dimension names + entity lists.

        Returns:
            Context string like:
            "# Discovered Dimensions
             - symptom: fever, cough, headache ... (53 total)
             - disease: pneumonia, influenza, diabetes ... (27 total)
             ..."
        """
        if not any(self.entity_lists.values()):
            # No entities yet - return just dimension names
            return "# Discovered Dimensions\n" + "\n".join([
                f"- {dim_name}: {self.global_dimensions[dim_name].description}"
                for dim_name in sorted(self.entity_lists.keys())
            ])

        lines = ["# Discovered Dimensions"]
        for dim_name in sorted(self.entity_lists.keys()):
            entities = self.entity_lists[dim_name]

            if not entities:
                # No entities yet for this dimension
                desc = self.global_dimensions[dim_name].description
                lines.append(f"- {dim_name}: {desc}")
            else:
                # Extract entity text from entity objects and show first 5 + count
                entity_texts = [e.text for e in entities]
                display = ", ".join(sorted(entity_texts[:5], key=str.lower))
                if len(entity_texts) > 5:
                    display += f" ... ({len(entity_texts)} total)"
                else:
                    display += f" ({len(entity_texts)} total)"
                lines.append(f"- {dim_name}: {display}")

        return "\n".join(lines)

    def _build_batch_messages(
        self,
        prompt: ClusterEntityPrompt,
        batch_questions: List[str],
        context_str: str,
        batch_idx: int,
        total_batches: int
    ) -> List[Dict]:
        """
        Build messages for a batch of questions.

        Args:
            prompt: Extraction prompt
            batch_questions: List of questions in this batch
            context_str: Context string with discovered dimensions
            batch_idx: Current batch index
            total_batches: Total number of batches

        Returns:
            List of message dicts for LLM
        """
        # Format questions as numbered list
        questions_text = "\n".join([
            f"{i+1}. {q}" for i, q in enumerate(batch_questions)
        ])

        # Build user message with context + batch instructions + questions
        user_message = f"""{context_str}

# Batch {batch_idx+1}/{total_batches}

Extract entities from the following {len(batch_questions)} questions. For each question, identify entities across ALL dimensions.

QUESTIONS:
{questions_text}

OUTPUT FORMAT:
Return a single JSON object with all entities from ALL questions combined. Do not separate by question - just extract all unique entities you find across all questions in this batch.
"""

        # Use the prompt's system message
        return [
            {"role": "system", "content": prompt.system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def _convert_to_dimension_configs(self, dimensions: Dict[str, DiscoveredDimensionSchema]) -> Dict:
        """
        Convert global dimensions to DimensionConfig objects for prompt.

        Args:
            dimensions: Global dimensional schemas

        Returns:
            Dict of dimension_name → DimensionConfig
        """
        # Import here to avoid circular dependency
        from stindex.extraction.dimension_loader import DimensionConfig

        configs = {}

        for dim_name, dim_schema in dimensions.items():
            # Create a DimensionConfig object with hierarchy support
            hierarchy = dim_schema.hierarchy

            # Generate field definitions from hierarchy levels
            fields = []
            for level in hierarchy:
                fields.append({
                    'name': level,
                    'type': 'string',
                    'description': f'{level.replace("_", " ").title()} level'
                })

            # Prepare examples in the correct format (list of dicts)
            raw_examples = dim_schema.examples
            formatted_examples = []
            if raw_examples:
                for ex in raw_examples:
                    if isinstance(ex, str):
                        # Convert string examples to dict format
                        formatted_examples.append({
                            'text': f'Example text containing {ex}',
                            hierarchy[0]: ex  # Use first hierarchy level as key
                        })
                    elif isinstance(ex, dict):
                        formatted_examples.append(ex)

            # Create DimensionConfig object with hierarchy
            configs[dim_name] = DimensionConfig(
                name=dim_name,
                enabled=True,
                extraction_type='hierarchical_categorical',
                hierarchy=hierarchy,
                schema_type='HierarchicalCategoricalMention',
                fields=fields,
                description=dim_schema.description,
                examples=formatted_examples
            )

        return configs

    def _convert_to_dimension_first(self, entity_first: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Convert entity-first format to dimension-first for aggregation.

        Entity-first format (from LLM):
        {
            "liver": {"dimension": "Anatomy", "specific_structure": "liver", "body_region": "abdomen"},
            "aspirin": {"dimension": "Pharmacology", "specific_drug": "aspirin", "drug_class": "NSAID"}
        }

        Dimension-first format (for aggregation):
        {
            "Anatomy": [
                {"text": "liver", "specific_structure": "liver", "body_region": "abdomen"}
            ],
            "Pharmacology": [
                {"text": "aspirin", "specific_drug": "aspirin", "drug_class": "NSAID"}
            ]
        }

        Args:
            entity_first: Entity-first dict from LLM

        Returns:
            Dimension-first dict for aggregation
        """
        dimension_first = {}

        for entity_name, entity_data in entity_first.items():
            dimension = entity_data.get('dimension')

            if not dimension:
                logger.warning(f"Entity '{entity_name}' missing 'dimension' field, skipping")
                continue

            # Create entity object with 'text' field + hierarchy fields
            entity_obj = {
                'text': entity_name,
                **{k: v for k, v in entity_data.items() if k != 'dimension'}
            }

            # Add to dimension list
            if dimension not in dimension_first:
                dimension_first[dimension] = []
            dimension_first[dimension].append(entity_obj)

        return dimension_first
