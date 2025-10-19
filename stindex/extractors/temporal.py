"""
Temporal entity extraction using LLMs (both local and API-based).
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as LangChainBaseModel
from pydantic import Field as LangChainField

from stindex.models.schemas import TemporalEntity, TemporalType
from stindex.utils.enhanced_time_normalizer import EnhancedTimeNormalizer
from stindex.prompts.templates import get_temporal_prompt


# LangChain-compatible schema for structured output
class TemporalMention(LangChainBaseModel):
    """Schema for LLM to extract temporal mentions."""

    text: str = LangChainField(description="The exact temporal expression found in the text")
    context: str = LangChainField(
        description="Surrounding context (up to 20 words) for disambiguation"
    )


class TemporalExtractionResult(LangChainBaseModel):
    """Collection of temporal mentions extracted by LLM."""

    temporal_mentions: List[TemporalMention] = LangChainField(
        default_factory=list, description="List of temporal expressions found"
    )


class TemporalExtractor:
    """Extract temporal entities from text using LLMs."""

    def __init__(
        self,
        llm_provider: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        model_path: Optional[str] = None,
        temperature: float = 0.0,
        reference_date: Optional[str] = None,
        device: str = "auto",
        use_few_shot: bool = True,
    ):
        """
        Initialize TemporalExtractor.

        Args:
            llm_provider: LLM provider (openai/anthropic/local), defaults to "local"
            model_name: Model name to use, defaults to "Qwen/Qwen3-8B"
            model_path: Local model path (for local provider)
            temperature: Temperature for LLM
            reference_date: Reference date for relative time resolution
            device: Device for local models (cuda/cpu/auto)
            use_few_shot: Use few-shot examples in prompts
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.use_few_shot = use_few_shot

        # Initialize LLM
        if llm_provider == "local":
            from stindex.llm.local_llm import LocalQwenLLM, LocalLLMWrapper

            self.llm = LocalQwenLLM(
                model_path=model_path,
                model_name=model_name,
                device=device,
                temperature=temperature,
            )
            self.llm_wrapper = LocalLLMWrapper(self.llm)

        elif llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
            self.llm_wrapper = self.llm

        elif llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=model_name, temperature=temperature)
            self.llm_wrapper = self.llm

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize enhanced time normalizer with context awareness
        self.time_normalizer = EnhancedTimeNormalizer(reference_date=reference_date)

        # Create extraction chain if using API models
        if llm_provider in ["openai", "anthropic"]:
            self.extraction_chain = self._create_extraction_chain()
        else:
            self.extraction_chain = None

    def _create_extraction_chain(self):
        """Create LangChain extraction chain with structured output (for API models)."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at extracting temporal information from text.
Your task is to identify ALL temporal expressions including:
- Absolute dates (March 15, 2022, January 1st, 2024-01-01)
- Absolute times (3:00 PM, 15:30, noon, midnight)
- Relative times (yesterday, last week, 3 days ago, next month)
- Durations (for 2 hours, 3 years, a week)
- Temporal intervals (from Monday to Friday, between 2020 and 2022)

Extract the EXACT text as it appears, along with surrounding context.
Be comprehensive - don't miss any temporal references.""",
                ),
                ("human", "Extract all temporal expressions from this text:\n\n{text}"),
            ]
        )

        # Use with_structured_output for reliable extraction
        structured_llm = self.llm.with_structured_output(TemporalExtractionResult)
        return prompt | structured_llm

    def extract(self, text: str) -> List[TemporalEntity]:
        """
        Extract temporal entities from text with context-aware normalization.

        Args:
            text: Input text

        Returns:
            List of TemporalEntity objects
        """
        if self.llm_provider == "local":
            return self._extract_local(text)
        else:
            return self._extract_api(text)

    def _extract_local(self, text: str) -> List[TemporalEntity]:
        """Extract using local LLM with enhanced context-aware normalization."""
        # Generate prompt with few-shot examples
        prompt = get_temporal_prompt(text, with_examples=self.use_few_shot)

        # Generate structured output
        result = self.llm.generate_structured(prompt)

        # Handle errors
        if "error" in result:
            print(f"Warning: {result.get('error')}")
            return []

        # Parse mentions
        mentions = result.get("temporal_mentions", [])

        # Extract mentions with context
        mentions_with_context = []
        for mention in mentions:
            if isinstance(mention, dict):
                mention_text = mention.get("text", "")
                mention_context = mention.get("context", "")
                if mention_text:
                    mentions_with_context.append((mention_text, mention_context))

        # Use batch normalization for context-aware year inference
        normalized_results = self.time_normalizer.normalize_batch(
            mentions_with_context,
            document_text=text  # Pass full document for year extraction
        )

        # Create TemporalEntity objects
        temporal_entities = []
        for i, mention in enumerate(mentions):
            if not isinstance(mention, dict):
                continue

            mention_text = mention.get("text", "")
            if not mention_text or i >= len(normalized_results):
                continue

            normalized, temporal_type = normalized_results[i]

            # Skip intervals that couldn't be normalized properly
            if temporal_type == TemporalType.INTERVAL and "/" not in normalized:
                # Could not normalize interval, skip it
                print(f"Warning: Could not normalize interval '{mention_text}', skipping")
                continue

            # Find character offsets
            start_char = text.find(mention_text)
            end_char = start_char + len(mention_text) if start_char != -1 else None

            # Handle intervals specially
            start_date, end_date = None, None
            if temporal_type == TemporalType.INTERVAL:
                start_date, end_date = self.time_normalizer.get_date_range(mention_text)

            # Create TemporalEntity
            entity = TemporalEntity(
                text=mention_text,
                normalized=normalized,
                temporal_type=temporal_type,
                confidence=0.90,  # High confidence from LLM
                start_char=start_char if start_char != -1 else None,
                end_char=end_char,
                start_date=start_date,
                end_date=end_date,
            )

            temporal_entities.append(entity)

        return temporal_entities

    def _extract_api(self, text: str) -> List[TemporalEntity]:
        """Extract using API-based LLM with enhanced normalization."""
        # Run LLM extraction
        result = self.extraction_chain.invoke({"text": text})

        # Extract mentions with context
        mentions_with_context = [
            (mention.text, mention.context)
            for mention in result.temporal_mentions
        ]

        # Use batch normalization for context-aware year inference
        normalized_results = self.time_normalizer.normalize_batch(
            mentions_with_context,
            document_text=text
        )

        # Create TemporalEntity objects
        temporal_entities = []
        for i, mention in enumerate(result.temporal_mentions):
            if i >= len(normalized_results):
                continue

            normalized, temporal_type = normalized_results[i]

            # Skip intervals that couldn't be normalized properly
            if temporal_type == TemporalType.INTERVAL and "/" not in normalized:
                print(f"Warning: Could not normalize interval '{mention.text}', skipping")
                continue

            # Find character offsets
            start_char = text.find(mention.text)
            end_char = start_char + len(mention.text) if start_char != -1 else None

            # Handle intervals specially
            start_date, end_date = None, None
            if temporal_type == TemporalType.INTERVAL:
                start_date, end_date = self.time_normalizer.get_date_range(mention.text)

            # Create TemporalEntity
            entity = TemporalEntity(
                text=mention.text,
                normalized=normalized,
                temporal_type=temporal_type,
                confidence=0.95,  # High confidence from LLM
                start_char=start_char if start_char != -1 else None,
                end_char=end_char,
                start_date=start_date,
                end_date=end_date,
            )

            temporal_entities.append(entity)

        return temporal_entities

    def extract_with_context(
        self, text: str, context_window: int = 50
    ) -> List[dict]:
        """
        Extract temporal entities with extended context.

        Args:
            text: Input text
            context_window: Number of characters before/after for context

        Returns:
            List of dictionaries with entity and context
        """
        entities = self.extract(text)

        results = []
        for entity in entities:
            if entity.start_char is not None and entity.end_char is not None:
                # Extract context
                context_start = max(0, entity.start_char - context_window)
                context_end = min(len(text), entity.end_char + context_window)
                context = text[context_start:context_end]

                results.append(
                    {
                        "entity": entity,
                        "context": context,
                        "pre_context": text[context_start : entity.start_char],
                        "post_context": text[entity.end_char : context_end],
                    }
                )
            else:
                results.append({"entity": entity, "context": None})

        return results
