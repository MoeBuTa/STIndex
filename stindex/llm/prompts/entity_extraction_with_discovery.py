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
- Use dimension names EXACTLY as shown below (with spaces, not underscores)
- Entity names should be the primary keys

OUTPUT FORMAT:
Return JSON with entity names as keys and their dimensional properties as values:

{{
  "entities": {{
    "entity_name_1": {{
      "dimension": "Dimension Name",
      "field1": "value1",
      "field2": "value2"
    }},
    "entity_name_2": {{
      "dimension": "Another Dimension",
      "field_a": "value_a",
      "field_b": "value_b"
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
        """Build instructions for schema discovery during extraction."""

        # Batch-aware instructions
        if self.batch_idx is not None and self.decay_threshold is not None:
            batch_num = self.batch_idx + 1

            if self.batch_idx == 0:
                # First batch - discovery mode
                template = """

5. NEW DIMENSION DISCOVERY (Primary Task for First Batch):
   This is the FIRST BATCH - your primary task is to discover the dimensional schema.

   **DISCOVERY MODE:**
   - Analyze the questions to identify natural dimensions
   - Extract entities AND propose dimensions simultaneously
   - Think about what types of information are being asked about
   - Consider hierarchical relationships (e.g., specific → category → broader_category)

   **Requirements for new dimensions:**
   - Must include a "confidence" field (float 0.0-1.0) indicating your confidence in this dimension
   - Confidence threshold for acceptance: {threshold:.1f}
   - Provide a clear hierarchical structure (2-4 levels recommended)
   - Include example entities from the questions
   - Explain why this dimension is useful

   New dimension format:
   "new_dimensions": {{
       "proposed_dimension_name": {{
           "confidence": 0.95,  // REQUIRED: Your confidence in proposing this dimension
           "hierarchy": ["level1", "level2", "level3"],
           "description": "What this dimension captures",
           "example_entities": ["entity1", "entity2", "entity3"],
           "rationale": "Why this dimension is needed and how it helps organize the data"
       }}
   }}

   **Guidance:**
   - In this first batch, err on the side of discovering more dimensions
   - Any dimension with confidence ≥ {threshold:.1f} will be accepted
   - Focus on dimensions that appear consistently across questions"""

                return template.format(threshold=self.decay_threshold)

            else:
                # Subsequent batches - refinement mode with decay
                if batch_num <= 2:
                    stage = "Early Refinement"
                    guidance = "You can still propose new dimensions relatively easily"
                elif batch_num <= 5:
                    stage = "Medium Refinement"
                    guidance = "New dimensions should be well-justified and clearly distinct"
                else:
                    stage = "Late Refinement"
                    guidance = "New dimensions should be rare and exceptionally important"

                template = """

5. NEW DIMENSION DISCOVERY (Refinement Mode - Batch {batch_num}, {stage}):
   **REFINEMENT MODE:**
   - Primary task: Extract entities using existing dimensions
   - Secondary task: Propose new dimensions only if highly confident

   **Requirements for new dimensions:**
   - Must include "confidence" field (float 0.0-1.0)
   - Confidence threshold for acceptance: {threshold:.1f} ({guidance})
   - Must be clearly distinct from existing dimensions
   - Must appear consistently across multiple questions in this batch
   - Should fill a gap that existing dimensions don't cover

   New dimension format:
   "new_dimensions": {{
       "proposed_dimension_name": {{
           "confidence": 0.95,  // REQUIRED: Must be ≥ {threshold:.1f} to be accepted
           "hierarchy": ["level1", "level2"],
           "description": "What this dimension captures",
           "example_entities": ["entity1", "entity2"],
           "rationale": "Why existing dimensions are insufficient"
       }}
   }}

   **Note:** Most batches should NOT propose new dimensions. Only propose if you find a clear gap in the schema."""

                return template.format(
                    batch_num=batch_num,
                    stage=stage,
                    threshold=self.decay_threshold,
                    guidance=guidance
                )

        # Default instructions (no batch context)
        template = """

5. NEW DIMENSION DISCOVERY (Optional):
   If you encounter important entities that don't fit existing dimensions:
   - Add a "new_dimensions" field to your JSON output
   - Propose new dimension(s) with hierarchical structure
   - Include a "confidence" field (0.0-1.0) for each proposed dimension
   - This should be RARE - try to fit entities into existing dimensions first
   - Only propose dimensions for information types that appear frequently/are clearly important

   New dimension format:
   "new_dimensions": {{
       "proposed_dimension_name": {{
           "confidence": 0.90,  // REQUIRED: Your confidence in proposing this dimension
           "hierarchy": ["level1", "level2", "level3"],
           "description": "What this dimension captures",
           "example_entities": ["entity1", "entity2", "entity3"],
           "rationale": "Why this dimension is needed"
       }}
   }}"""

        return template

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

        Returns:ex
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

        # Add new_dimensions property if discovery is enabled
        if self.allow_new_dimensions:
            schema["properties"]["new_dimensions"] = {
                "type": "object",
                "description": "Optional: Proposed new dimensions for entities that don't fit existing schemas"
            }

        return schema

    def parse_response_with_discovery(self, llm_response: str) -> Dict:
        """
        Parse LLM response and extract both entities and newly discovered dimensions.

        Args:
            llm_response: Raw LLM response (may include reasoning + JSON)

        Returns:
            Dict with:
                - 'entities': Entity-first dict {entity_name: {dimension: "X", field: "val"}}
                - 'new_dimensions': Newly discovered dimensions (if any)
                - 'reasoning': Extracted CoT reasoning (if any)
                - 'raw_response': Original LLM response
        """
        from stindex.extraction.utils import extract_cot_and_json

        # Extract reasoning and JSON
        result = extract_cot_and_json(llm_response)

        if not isinstance(result['data'], dict):
            raise ValueError(f"Expected dict, got {type(result['data'])}")

        # Extract entity-first data and new dimensions
        entities = result['data'].get('entities', {})
        new_dimensions = result['data'].get('new_dimensions', {})

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
            'new_dimensions': new_dimensions,
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
