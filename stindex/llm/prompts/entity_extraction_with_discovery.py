"""
Prompt generation for entity extraction with dimensional discovery.

This prompt extends the base dimensional extraction to support discovering
new dimensions or refining existing ones during the extraction process.
"""

import json
from typing import Dict, List, Optional

from stindex.llm.prompts.dimensional_extraction import DimensionalExtractionPrompt
from stindex.extraction.dimension_loader import DimensionConfig


class ClusterEntityPrompt(DimensionalExtractionPrompt):
    """
    Prompt for cluster-level entity extraction with schema discovery support.

    Builds on DimensionalExtractionPrompt but adds:
    1. Ability to propose new dimensions if entities don't fit existing schemas
    2. Context-aware extraction using prior extractions from cluster
    3. Schema refinement based on extracted entities
    """

    def __init__(
        self,
        dimensions: Dict[str, DimensionConfig],
        document_metadata: Optional[Dict] = None,
        extraction_context: Optional[object] = None,
        allow_new_dimensions: bool = True,
        cluster_id: Optional[int] = None,
        batch_idx: Optional[int] = None,
        decay_threshold: Optional[float] = None
    ):
        """
        Initialize prompt builder.

        Args:
            dimensions: Dict of dimension name → DimensionConfig
            document_metadata: Optional document metadata
            extraction_context: Optional ExtractionContext for context-aware prompts
            allow_new_dimensions: Whether to allow proposing new dimensions
            cluster_id: Optional cluster ID for context
            batch_idx: Optional batch index for decay-aware prompts
            decay_threshold: Optional confidence threshold for new dimensions
        """
        super().__init__(dimensions, document_metadata, extraction_context)
        self.allow_new_dimensions = allow_new_dimensions
        self.cluster_id = cluster_id
        self.batch_idx = batch_idx
        self.decay_threshold = decay_threshold

    def system_prompt(self) -> str:
        """Generate system prompt with extraction + discovery instructions."""

        # Get base extraction context and tasks
        extraction_context = self._build_extraction_context()
        document_context = self._build_document_context()
        dimension_tasks = self._build_dimension_tasks()

        # Add discovery instructions if enabled
        discovery_instructions = ""
        if self.allow_new_dimensions:
            discovery_instructions = self._build_discovery_instructions()

        # Template with placeholders
        template = """You are a precise multi-dimensional entity extraction system with schema discovery capabilities.

REASONING PROCESS:
Before providing the JSON output, think step-by-step through:
1. **Extract entities** - Identify all relevant entities from the text
2. **Classify dimension** - For each entity, determine which dimension it belongs to
3. **Map hierarchy** - Fill in the appropriate hierarchy fields for each entity
4. **Verify consistency** - Check entity names and fields against prior extractions from context

CRITICAL OUTPUT RULES:
- Output valid JSON with reasoning (optional) before it
- Entity names should be the primary keys
- **DIMENSION FIELD: Use broad, parent-level dimension names**
  - Use the top-level category names (e.g., "Product", "Location", "Event")
  - DO NOT use fine-grained subcategories as dimension names
  - Fine-grained classifications go in hierarchy fields, NOT the dimension field

OUTPUT FORMAT:
Return JSON with entity names as keys and their dimensional properties as values:

{{
  "entities": {{
    "apple": {{
      "dimension": "Product",         // Use parent dimension name
      "specific_item": "apple",       // Hierarchy level 1: specific value
      "item_category": "fruit"        // Hierarchy level 2: category
    }},
    "tokyo": {{
      "dimension": "Location",        // Use parent dimension name
      "specific_place": "tokyo",
      "place_type": "city"
    }}
  }}
}}


{extraction_context}{document_context}EXTRACTION TASKS:

{dimension_tasks}{discovery_instructions}

EXTRACTION PRINCIPLES:
1. **Completeness**: Extract entities at ALL hierarchy levels when present
   - If a dimension has 3 levels, try to fill all 3 levels for each entity
   - Example: If extracting "New York City" in spatial dimension with hierarchy [city, state, country],
     also extract state="New York" and country="USA"

2. **Consistency**: Use context from previous extractions to maintain consistent terminology
   - If previous questions used "USA", use "USA" instead of "United States"
   - Follow naming conventions established in prior extractions

3. **Accuracy**: Only extract entities that are explicitly stated or clearly implied
- Don't invent information
   - Use null for hierarchy levels that don't apply

4. **Hierarchical Mapping**: Correctly map extracted entities to their hierarchical levels
   - Understand parent-child relationships in taxonomies
   - Example: "apple" → item, "fruit" → category, "produce" → broader_category

REMINDER: Return entity-first JSON format. Each entity name is a key, with "dimension" field plus hierarchy fields."""

        return template.format(
            extraction_context=extraction_context,
            document_context=document_context,
            dimension_tasks=dimension_tasks,
            discovery_instructions=discovery_instructions
        )

    def _build_discovery_instructions(self) -> str:
        """Build instructions for dimension naming during extraction.

        Note: Dimensions are now derived from entities by post-processing code,
        not proposed by the LLM. This method provides guidance on consistent
        dimension naming.
        """
        # Batch-aware instructions
        if self.batch_idx is not None:
            if self.batch_idx == 0:
                # First batch - discovery mode
                return """

5. DIMENSION NAMING GUIDANCE (First Batch - Discovery Mode):
   This is the FIRST BATCH. Focus on discovering natural categories:

   **Guidelines:**
   - Use broad, parent-level dimension names (e.g., "Product" not "Fruit")
   - Be consistent with dimension names across entities
   - Include hierarchical fields from specific to general
   - Think about what top-level categories naturally emerge from the text

   **Example:**
   For "apple" and "banana", use:
   - dimension: "Product" (broad category)
   - specific_item: "apple" or "banana" (specific value)
   - item_category: "fruit" (subcategory - goes in hierarchy, NOT dimension)"""

            else:
                # Subsequent batches - use existing dimension names
                return """

5. DIMENSION NAMING GUIDANCE (Batch {batch_num}):
   **Use existing dimension names when possible.**

   - Check the context for dimension names used in previous batches
   - Maintain consistency with established naming conventions
   - Only introduce new dimension names if truly distinct from existing ones""".format(
                    batch_num=self.batch_idx + 1
                )

        # Default instructions (no batch context)
        return """

5. DIMENSION NAMING GUIDANCE:
   - Use broad, parent-level dimension names
   - Include hierarchical fields for categorization (specific to general)
   - Be consistent across entities"""

    def _build_extraction_context(self) -> str:
        """Build extraction context with cluster information."""
        context_parts = []

        # Add cluster context if available
        if self.cluster_id is not None:
            context_parts.append(f"CLUSTER CONTEXT:\nProcessing questions from Cluster {self.cluster_id}\n")

        # Add base context from extraction_context (cmem - memory context)
        if self.extraction_context:
            context_str = self.extraction_context.to_prompt_context()
            if context_str.strip():
                # Check if it already has a header
                if not context_str.startswith("CONTEXT FROM PREVIOUS"):
                    context_parts.append("CONTEXT FROM PREVIOUS EXTRACTIONS:\n" + context_str)
                else:
                    context_parts.append(context_str)

        if context_parts:
            return "\n".join(context_parts) + "\n\n"
        return ""

    def build_output_schema(self) -> Dict:
        """
        Build expected JSON schema for output validation.

        Returns:
            JSON schema dict
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Add properties for each dimension
        for dim_name, dim_config in self.dimensions.items():
            schema["properties"][dim_name] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {}
                }
            }

            # Add field properties
            for field in dim_config.fields:
                field_schema = {"type": field.get("type", "string")}
                if field.get("type") == "enum" and field.get("values"):
                    field_schema["enum"] = field["values"]

                schema["properties"][dim_name]["items"]["properties"][field["name"]] = field_schema

        # Note: new_dimensions is no longer part of schema
        # Dimensions are derived from entities by post-processing code

        return schema

    def parse_response_with_discovery(self, llm_response: str) -> Dict:
        """
        Parse LLM response and extract entities.

        Note: Dimensions are now derived from entities by post-processing code,
        so new_dimensions is always empty in the return value.

        Args:
            llm_response: Raw LLM response (may include reasoning + JSON)

        Returns:
            Dict with:
                - 'entities': Entity-first dict {entity_name: {dimension: "X", field: "val"}}
                - 'new_dimensions': Always empty dict (dimensions derived by post-processing)
                - 'reasoning': Extracted CoT reasoning (if any)
                - 'raw_response': Original LLM response
        """
        from stindex.extraction.utils import extract_cot_and_json

        # Extract reasoning and JSON
        result = extract_cot_and_json(llm_response)

        if not isinstance(result['data'], dict):
            raise ValueError(f"Expected dict, got {type(result['data'])}")

        # Extract entity-first data
        entities = result['data'].get('entities', {})

        # Validate entity-first structure
        if not isinstance(entities, dict):
            raise ValueError(f"Expected 'entities' to be dict, got {type(entities)}")

        # Validate each entity has a 'dimension' field
        for entity_name, entity_data in entities.items():
            if not isinstance(entity_data, dict):
                raise ValueError(
                    f"Entity '{entity_name}' should be dict, got {type(entity_data)}"
                )
            if 'dimension' not in entity_data:
                raise ValueError(
                    f"Entity '{entity_name}' missing 'dimension' field. "
                    f"Found fields: {list(entity_data.keys())}"
                )

        return {
            'entities': entities,
            'new_dimensions': {},  # Always empty - dimensions derived from entities
            'reasoning': result['reasoning'],
            'raw_response': result['raw_response']
        }

    def build_messages_for_question(
        self,
        question: str,
        question_index: int,
        total_questions: int
    ) -> List[Dict[str, str]]:
        """
        Build messages for extracting from a single question in a cluster.

        Args:
            question: Question text to extract from
            question_index: Index of this question in cluster
            total_questions: Total questions in cluster

        Returns:
            List of message dicts
        """
        # Update extraction context with position
        if self.extraction_context:
            self.extraction_context.set_chunk_position(question_index, total_questions)

        return self.build_messages(question)
