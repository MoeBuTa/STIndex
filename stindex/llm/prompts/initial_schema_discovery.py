"""
Prompt generation for initial schema discovery from question clusters.

This prompt is used AFTER clustering to discover high-level dimensional schemas
by sampling questions from each cluster and asking the LLM to propose 2-3 dimensions.
"""

import json
from typing import Dict, List, Optional


class ClusterSchemaPrompt:
    """
    Prompt for cluster-level schema discovery from question clusters.

    This is the first phase of schema discovery where we ask the LLM to propose
    dimensions based purely on what naturally emerges from sample questions.
    No constraints on dimension count - fully data-driven.
    """

    def __init__(
        self,
        predefined_dimensions: Optional[List[str]] = None,
        cluster_id: Optional[int] = None
    ):
        """
        Initialize prompt builder.

        Args:
            predefined_dimensions: List of predefined dimension names (e.g., ['temporal', 'spatial'])
            cluster_id: Optional cluster ID for context
        """
        self.predefined_dimensions = predefined_dimensions or ['temporal', 'spatial']
        self.cluster_id = cluster_id

    def system_prompt(self) -> str:
        """Generate system prompt for schema discovery."""

        template = """You are a dimensional schema design expert for information extraction.

Your task is to analyze questions from a dataset and propose domain-specific dimensional schemas that can be used to organize and retrieve information.

REASONING PROCESS:
Before providing the JSON output, think step-by-step through:
1. **Entity patterns** - What types of entities appear frequently across questions?
2. **Natural groupings** - Which entities belong together conceptually?
3. **Hierarchical structure** - What are the natural hierarchy levels (specific → general)?
4. **Information utility** - How will these dimensions help retrieve relevant documents?

Your response may include reasoning before the JSON output.
If your model supports <think> tags, use them for your reasoning.
Otherwise, write your reasoning as text before the JSON.

Key Principles:
1. **Natural Hierarchical Structure**: Each dimension should have hierarchy levels that naturally emerge from the data
   - Use as many or as few levels as the concepts require
   - Let the inherent granularity of the domain determine the depth
   - Simple taxonomies may need fewer levels, complex ones may need more
   - Examples:
     * Shallow: item → category
     * Medium: specific_term → subcategory → category
     * Deep: most_specific → intermediate_levels... → broad_domain
   - IMPORTANT: Hierarchy depth should be data-driven, not predetermined

2. **Information Retrieval Utility**: Schemas should help in retrieving relevant documents for question answering
   - Consider: What information would help answer these questions?
   - Think: How can we organize knowledge to support this type of reasoning?
   - Deeper hierarchies enable both precise filtering and broad generalization

3. **Domain Relevance**: Focus on categorizations that align with how knowledge is naturally organized
   - Identify meaningful relationships between concepts
   - Consider existing taxonomies and ontologies if relevant
   - Let the structure emerge from observed patterns in the data

4. **Avoid Redundancy**: Do NOT propose temporal or spatial dimensions (these are predefined)
   - Predefined dimensions: {predefined_dims}
   - Focus on domain-specific dimensions only

OUTPUT FORMAT:
After your reasoning, provide valid JSON with VARIABLE hierarchy depths:
{{
    "dimension_name_1": {{
        "hierarchy": ["level1", "level2", "level3", "level4"],
        "description": "Clear description of what this dimension captures and why it's useful for retrieval",
        "examples": ["example entity 1", "example entity 2", "example entity 3"]
    }},
    "dimension_name_2": {{
        "hierarchy": ["level1", "level2"],
        "description": "...",
        "examples": ["...", "...", "..."]
    }},
    "dimension_name_3": {{
        "hierarchy": ["level1", "level2", "level3", "level4", "level5"],
        "description": "...",
        "examples": ["...", "...", "..."]
    }}
}}

NOTE: The examples above show 4, 2, and 5 levels respectively - your dimensions should have varying depths based on domain complexity!"""

        return template.format(
            predefined_dims=", ".join(self.predefined_dimensions)
        )

    def user_prompt(self, sample_questions: List[str]) -> str:
        """
        Generate user prompt with sampled questions.

        Args:
            sample_questions: Sample questions from the cluster (typically 15-20)

        Returns:
            Formatted user prompt
        """
        # Format questions (truncate if too long)
        questions_formatted = []
        for i, q in enumerate(sample_questions, 1):
            # Truncate long questions
            q_text = q[:300] + "..." if len(q) > 300 else q
            questions_formatted.append(f"{i}. {q_text}")

        questions_str = "\n".join(questions_formatted)

        cluster_context = f"Cluster {self.cluster_id}: " if self.cluster_id is not None else ""

        template = """{cluster_context}Analyze these {n_samples} questions and propose domain-specific dimensional schemas.

Questions:
{questions}

Requirements:
1. Propose domain-specific dimensions based ONLY on what you observe in the data
   - Only propose dimensions that clearly and naturally emerge from the question patterns
   - Fewer, high-quality dimensions are better than many forced ones
   - Do NOT propose temporal or spatial dimensions (those are predefined)
   - Let the data determine how many dimensions exist - could be 2, could be 5, could be more
2. Each dimension should capture a distinct aspect of knowledge relevant to these questions
3. Each dimension must have a hierarchical structure with as many levels as naturally fit the data
   - Don't force a specific number of levels - let the hierarchy emerge naturally
   - Simple concepts may need only 2 levels, complex ones may need 5 or more
4. Provide clear descriptions explaining what each dimension captures and why it's useful
5. Include 3-5 concrete examples per dimension

Consider:
- What types of information appear frequently in these questions?
- How is knowledge naturally organized in this domain?
- What hierarchical relationships exist between concepts?
- What information would help retrieve relevant documents to answer these questions?
- Are there truly distinct dimensional aspects, or would some dimensions overlap?

Output ONLY valid JSON following the schema in the system prompt."""

        return template.format(
            cluster_context=cluster_context,
            n_samples=len(sample_questions),
            questions=questions_str
        )

    def build_messages(self, sample_questions: List[str]) -> List[Dict[str, str]]:
        """
        Build message list for schema discovery.

        Args:
            sample_questions: Sample questions from cluster

        Returns:
            List of message dicts for LLM
        """
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_prompt(sample_questions)}
        ]

    def parse_response(self, llm_response: str) -> Dict:
        """
        Parse LLM response into discovered schemas.

        Args:
            llm_response: Raw LLM response (may include reasoning + JSON)

        Returns:
            Dict with:
                - 'schemas': Dict of dimension_name → schema config
                - 'reasoning': Extracted CoT reasoning (if any)
                - 'raw_response': Original LLM response
        """
        # Extract reasoning and JSON using new utility
        from stindex.extraction.utils import extract_cot_and_json

        result = extract_cot_and_json(llm_response)
        discovered_schemas = result['data']

        # Validate structure
        if not isinstance(discovered_schemas, dict):
            raise ValueError(f"Expected dict, got {type(discovered_schemas)}")

        for dim_name, schema in discovered_schemas.items():
            required_fields = {'hierarchy', 'description', 'examples'}
            missing = required_fields - set(schema.keys())
            if missing:
                raise ValueError(f"Dimension '{dim_name}' missing fields: {missing}")

        return {
            'schemas': discovered_schemas,
            'reasoning': result['reasoning'],
            'raw_response': result['raw_response']
        }
