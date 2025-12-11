"""
RAG Prompt Templates.

Provides prompt templates for different RAG tasks:
- Simple Q&A
- Multi-hop reasoning
- Citation-aware generation
- Custom templates
"""

from typing import Optional


class RAGPromptTemplates:
    """
    Collection of prompt templates for RAG generation.
    """

    # Default system prompt
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Your answers should be:
1. Accurate - Only use information from the provided context
2. Concise - Give direct answers without unnecessary elaboration
3. Honest - If the context doesn't contain enough information, say so

If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question based on the provided context."
"""

    # Reasoning system prompt for multi-hop questions
    REASONING_SYSTEM_PROMPT = """You are a helpful assistant that answers complex questions by reasoning step-by-step.

For each question:
1. Identify the key entities and relationships mentioned
2. Find relevant information in the context for each entity
3. Connect the information to form a complete answer
4. State your final answer clearly

Your answers should be accurate and based only on the provided context."""

    # Q&A prompt template
    QA_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

    # Reasoning prompt template
    REASONING_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Let me work through this step by step:"""

    # Citation-aware prompt template
    CITATION_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Please answer the question using information from the context above. Reference the relevant document(s) using [Document X] format where applicable.

Answer:"""

    def __init__(self):
        """Initialize prompt templates."""
        pass

    def get_system_prompt(self, template: str = "default") -> str:
        """
        Get system prompt by template name.

        Args:
            template: Template name ('default', 'reasoning', 'citation')

        Returns:
            System prompt string
        """
        if template == "reasoning":
            return self.REASONING_SYSTEM_PROMPT
        else:
            return self.DEFAULT_SYSTEM_PROMPT

    def get_reasoning_system_prompt(self) -> str:
        """Get system prompt for reasoning tasks."""
        return self.REASONING_SYSTEM_PROMPT

    def format_qa_prompt(self, question: str, context: str) -> str:
        """
        Format a simple Q&A prompt.

        Args:
            question: The question to answer
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return self.QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    def format_reasoning_prompt(self, question: str, context: str) -> str:
        """
        Format a reasoning prompt for multi-hop questions.

        Args:
            question: Complex question requiring reasoning
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return self.REASONING_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    def format_citation_prompt(self, question: str, context: str) -> str:
        """
        Format a citation-aware prompt.

        Args:
            question: The question to answer
            context: Retrieved context with document markers

        Returns:
            Formatted prompt string
        """
        return self.CITATION_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

    def format_custom_prompt(
        self,
        template: str,
        question: str,
        context: str,
        **kwargs,
    ) -> str:
        """
        Format a custom prompt template.

        Args:
            template: Template string with {question} and {context} placeholders
            question: The question to answer
            context: Retrieved context
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        return template.format(
            question=question,
            context=context,
            **kwargs,
        )


# Pre-built prompts for specific tasks

HOTPOTQA_SYSTEM_PROMPT = """You are an expert at answering multi-hop questions.

Multi-hop questions require combining information from multiple sources to find the answer.

Guidelines:
1. Carefully read all provided documents
2. Identify which documents contain relevant information
3. Connect facts across documents to reason towards the answer
4. Provide a concise, direct answer

If the documents don't contain enough information to answer the question definitively, explain what additional information would be needed."""

COMPARISON_SYSTEM_PROMPT = """You are an expert at answering comparison questions.

Guidelines:
1. Identify the entities being compared
2. Find relevant attributes for each entity in the context
3. Compare the specific attributes mentioned in the question
4. Provide a clear, factual comparison

Base your answer only on the provided context."""

TEMPORAL_SYSTEM_PROMPT = """You are an expert at answering questions about events and their timing.

Guidelines:
1. Pay attention to dates, times, and temporal relationships
2. Identify the correct time period for the question
3. Use chronological reasoning when needed
4. Be precise about dates when the context provides them

Base your answer only on the provided context."""
