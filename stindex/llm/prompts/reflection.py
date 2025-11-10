"""
Reflection prompt generation for two-pass extraction quality control.

Generates context-aware prompts for LLM-based scoring and filtering of extraction results.
Uses context information (prior extractions, document metadata) to enhance reasoning.
"""

import json
from typing import Dict, List, Optional

from stindex.extraction.context_manager import ExtractionContext


class ReflectionPrompt:
    """
    Context-aware prompt builder for two-pass extraction reflection.

    Creates prompts that ask an LLM to reflect on extraction quality using:
    - Relevance: Is the entity actually in the text?
    - Accuracy: Does the extraction match the text exactly?
    - Completeness: Is the extraction complete?
    - Consistency: Does it align with prior extractions and context?

    Enhanced with context information:
    - Prior temporal/spatial references from same document
    - Document metadata (publication date, source location)
    - Chunk position within document
    """

    def __init__(
        self,
        text: str,
        dimension_name: str,
        entities: List[Dict],
        dimension_schemas: Optional[Dict] = None,
        extraction_context: Optional[ExtractionContext] = None
    ):
        """
        Initialize reflection prompt builder.

        Args:
            text: Original input text
            dimension_name: Name of dimension being reflected on
            entities: List of extracted entities to reflect on
            dimension_schemas: Optional dimension schemas for context
            extraction_context: Optional ExtractionContext for context-aware reflection
        """
        self.text = text
        self.dimension_name = dimension_name
        self.entities = entities
        self.dimension_schemas = dimension_schemas or {}
        self.extraction_context = extraction_context

    def system_prompt(self) -> str:
        """Generate system prompt for reflection task."""
        return (
            "You are an extraction quality reflector. Analyze extractions in context "
            "and provide scores with detailed reasoning. Consider document context, "
            "prior extractions, and consistency with known information."
        )

    def user_prompt(self) -> str:
        """
        Generate user prompt with context-aware reflection task.

        Returns:
            Reflection prompt string asking LLM to score entities with context
        """
        entities_json = json.dumps(self.entities, indent=2)

        # Build schema description
        schema_desc = ""
        if self.dimension_name in self.dimension_schemas:
            schema = self.dimension_schemas[self.dimension_name]
            schema_desc = f"\n\nDimension schema:\n{json.dumps(schema, indent=2)}"

        # Build context section (if available)
        context_section = self._build_context_section()

        prompt = f"""You are reflecting on extraction quality to filter false positives and improve accuracy.

Score each extracted entity on four criteria:

1. **Relevance** (0-1): Is this entity actually mentioned in the text?
2. **Accuracy** (0-1): Does the extraction accurately represent what's in the text?
3. **Completeness** (0-1): Is the extraction complete and not missing important details?
4. **Consistency** (0-1): Does it align with document context and prior extractions?

**Original Text:**
{self.text}

**Dimension:** {self.dimension_name}{schema_desc}
{context_section}
**Extracted Entities:**
{entities_json}

**Task:** For each entity, provide scores and explain your reasoning considering:
- Text evidence: What specific phrases support this extraction?
- Context alignment: Does it fit with the document's publication date, location, and prior references?
- Consistency: Is it consistent with other extractions from this document?
- Potential issues: Are there any red flags (hallucination, misinterpretation, missing info)?

Respond with ONLY a JSON array, one score object per entity:
```json
[
  {{
    "entity_index": 0,
    "relevance": 0.95,
    "accuracy": 0.90,
    "completeness": 0.85,
    "consistency": 0.90,
    "reasoning": "Detailed explanation considering text evidence and context"
  }},
  ...
]
```

CRITICAL: Return ONLY the JSON array, nothing else."""

        return prompt

    def _build_context_section(self) -> str:
        """
        Build context information section for prompt.

        Returns:
            Formatted context section with document metadata and prior extractions
        """
        if not self.extraction_context:
            return ""

        sections = []

        # Document context
        if self.extraction_context.document_metadata:
            sections.append("\n**Document Context:**")

            pub_date = self.extraction_context.document_metadata.get('publication_date')
            if pub_date:
                sections.append(f"- Publication Date: {pub_date}")

            source_loc = self.extraction_context.document_metadata.get('source_location')
            if source_loc:
                sections.append(f"- Source Location: {source_loc}")

            doc_title = self.extraction_context.document_metadata.get('document_title')
            if doc_title:
                sections.append(f"- Document Title: {doc_title}")

            # Chunk position
            if self.extraction_context.total_chunks > 0:
                sections.append(
                    f"- Position: Chunk {self.extraction_context.current_chunk_index + 1} "
                    f"of {self.extraction_context.total_chunks}"
                )

            if self.extraction_context.section_hierarchy:
                sections.append(f"- Section: {self.extraction_context.section_hierarchy}")

        # Prior temporal references (for temporal dimension)
        if self.dimension_name == "temporal" and self.extraction_context.prior_temporal_refs:
            sections.append("\n**Prior Temporal References from this Document:**")
            sections.append("Use these to assess consistency of relative temporal expressions:")
            for ref in self.extraction_context.prior_temporal_refs[-5:]:  # Last 5
                sections.append(f"  - {ref['text']} → {ref['normalized']}")

        # Prior spatial references (for spatial dimension)
        if self.dimension_name == "spatial" and self.extraction_context.prior_spatial_refs:
            sections.append("\n**Prior Spatial References from this Document:**")
            sections.append("Use these to assess spatial consistency and disambiguation:")
            for ref in self.extraction_context.prior_spatial_refs[-5:]:  # Last 5
                parent = ref.get('parent_region', '')
                parent_str = f" in {parent}" if parent else ""
                sections.append(f"  - {ref['text']}{parent_str}")

        # Prior events (if available)
        if self.extraction_context.prior_events:
            sections.append("\n**Prior Events from this Document:**")
            for event in self.extraction_context.prior_events[-3:]:  # Last 3
                sections.append(f"  - {event.get('text', '')}")

        if sections:
            return "\n".join(sections) + "\n"
        else:
            return ""

    def build_messages(self) -> List[Dict[str, str]]:
        """
        Build message list for reflection.

        Returns:
            List of message dicts for LLM API
        """
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_prompt()}
        ]
