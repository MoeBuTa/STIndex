"""
SpatioTemporal Extractor Agent implementing observe-reason-act pattern.

This agent:
1. OBSERVE: Preprocesses text and extracts document-level context
2. REASON: Uses single LLM call to extract both temporal and spatial entities
3. ACT: Postprocesses using tools (normalize_temporal, geocode_location)
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.agents.base import BaseAgent
from stindex.agents.prompts import get_combined_prompt
from stindex.agents.response.models import (
    ExtractionActionResponse,
    ExtractionObservation,
    ExtractionReasoning,
    SpatialMention,
    TemporalMention,
)
from stindex.models.schemas import SpatialEntity, TemporalEntity, TemporalType
from stindex.utils.enhanced_geocoder import EnhancedGeocoderService
from stindex.utils.enhanced_time_normalizer import EnhancedTimeNormalizer
from stindex.utils.preprocessing import ContextEnricher, TextPreprocessor
from stindex.utils.tools import ToolRegistry


class SpatioTemporalExtractorAgent(BaseAgent):
    """
    Agent for extracting spatiotemporal entities using observe-reason-act pattern.
    """

    def __init__(self, config: Dict[str, Any], llm: Optional[Any] = None):
        """
        Initialize extractor agent.

        Args:
            config: Configuration dictionary
            llm: Optional pre-initialized LLM
        """
        super().__init__(config, llm)

        # Initialize preprocessing
        self.preprocessor = TextPreprocessor()
        self.context_enricher = ContextEnricher()

        # Initialize postprocessing tools
        self.time_normalizer = EnhancedTimeNormalizer(
            reference_date=config.get("reference_date")
        )

        self.geocoder = EnhancedGeocoderService(
            user_agent=config.get("user_agent", "stindex"),
            enable_cache=config.get("enable_cache", True),
            rate_limit=config.get("rate_limit", 1.0),
        )

        # Initialize tool registry
        self.tool_registry = ToolRegistry(
            time_normalizer=self.time_normalizer,
            geocoder=self.geocoder,
            enable_temporal=config.get("enable_temporal", True),
            enable_spatial=config.get("enable_spatial", True),
        )

        # Config
        self.min_confidence = config.get("min_confidence", 0.5)

    def observe(self, environment: Dict[str, Any]) -> ExtractionObservation:
        """
        OBSERVE: Preprocess text and extract document-level context.

        Args:
            environment: Dictionary with key 'text'

        Returns:
            ExtractionObservation with preprocessed data
        """
        text = environment.get("text", "")

        # Preprocess text
        preprocessed = self.preprocessor.preprocess(text)

        return ExtractionObservation(
            original_text=text,
            cleaned_text=preprocessed["cleaned_text"],
            temporal_context=preprocessed["temporal_context"],
            spatial_context=preprocessed["spatial_context"],
            language=preprocessed["language"],
            char_count=preprocessed["metadata"]["char_count"],
            word_count=preprocessed["metadata"]["word_count"],
        )

    def reason(self, observations: ExtractionObservation) -> ExtractionReasoning:
        """
        REASON: Single LLM call to extract both temporal and spatial entities.

        Args:
            observations: ExtractionObservation from observe phase

        Returns:
            ExtractionReasoning with LLM output
        """
        try:
            # Check LLM provider
            llm_provider = self.config.get("llm_provider", "local")

            if llm_provider == "local":
                return self._reason_with_local_llm(observations)
            else:
                return self._reason_with_api_llm(observations)

        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            return ExtractionReasoning(
                temporal_mentions=[],
                spatial_mentions=[],
                raw_output=str(e),
                success=False,
                error=str(e),
            )

    def _reason_with_local_llm(
        self, observations: ExtractionObservation
    ) -> ExtractionReasoning:
        """Reason using local LLM (Qwen3-8B)."""
        # Generate combined prompt
        prompt = get_combined_prompt(observations.cleaned_text)

        # Generate structured output
        result = self.llm.generate_structured(prompt)

        # Handle errors
        if "error" in result:
            return ExtractionReasoning(
                temporal_mentions=[],
                spatial_mentions=[],
                raw_output=str(result),
                success=False,
                error=result.get("error"),
            )

        # Parse mentions
        temporal_mentions = [
            TemporalMention(
                text=m.get("text", ""),
                context=m.get("context", "")
            )
            for m in result.get("temporal_mentions", [])
            if isinstance(m, dict) and m.get("text")
        ]

        spatial_mentions = [
            SpatialMention(
                text=m.get("text", ""),
                context=m.get("context", ""),
                type=m.get("type", "other")
            )
            for m in result.get("spatial_mentions", [])
            if isinstance(m, dict) and m.get("text")
        ]

        return ExtractionReasoning(
            temporal_mentions=temporal_mentions,
            spatial_mentions=spatial_mentions,
            raw_output=str(result),
            success=True,
        )

    def _reason_with_api_llm(
        self, observations: ExtractionObservation
    ) -> ExtractionReasoning:
        """Reason using API LLM (OpenAI/Anthropic)."""
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel as LangChainBaseModel
        from pydantic import Field as LangChainField

        # Define schema for structured output
        class UnifiedExtractionResult(LangChainBaseModel):
            temporal_mentions: List[TemporalMention] = LangChainField(default_factory=list)
            spatial_mentions: List[SpatialMention] = LangChainField(default_factory=list)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting both temporal and spatial information from text."),
            ("human", get_combined_prompt(observations.cleaned_text))
        ])

        # Use structured output
        structured_llm = self.llm.with_structured_output(UnifiedExtractionResult)
        chain = prompt | structured_llm

        # Run extraction
        result = chain.invoke({"text": observations.cleaned_text})

        return ExtractionReasoning(
            temporal_mentions=result.temporal_mentions,
            spatial_mentions=result.spatial_mentions,
            raw_output=str(result),
            success=True,
        )

    def act(self, reasoning: ExtractionReasoning) -> ExtractionActionResponse:
        """
        ACT: Postprocess entities using tools (normalize_temporal, geocode_location).

        Args:
            reasoning: ExtractionReasoning from reason phase

        Returns:
            ExtractionActionResponse with normalized entities
        """
        start_time = time.time()

        if not reasoning.success:
            return ExtractionActionResponse(
                temporal_entities=[],
                spatial_entities=[],
                success=False,
                error=reasoning.error,
                processing_time=time.time() - start_time,
            )

        try:
            # Enrich with document context
            observations = self.get_state("observations")

            temporal_mentions_enriched = self.context_enricher.enrich_temporal_mentions(
                [m.dict() for m in reasoning.temporal_mentions],
                observations.temporal_context if observations else {},
            )

            spatial_mentions_enriched = self.context_enricher.enrich_spatial_mentions(
                [m.dict() for m in reasoning.spatial_mentions],
                observations.spatial_context if observations else {},
            )

            # Postprocess temporal entities
            temporal_entities = self._postprocess_temporal(
                temporal_mentions_enriched,
                observations.cleaned_text if observations else "",
                observations.temporal_context if observations else {},
            )

            # Postprocess spatial entities
            spatial_entities = self._postprocess_spatial(
                spatial_mentions_enriched,
                observations.cleaned_text if observations else "",
                observations.spatial_context if observations else {},
            )

            # Filter by confidence
            temporal_entities = [
                e for e in temporal_entities
                if e.confidence >= self.min_confidence
            ]
            spatial_entities = [
                e for e in spatial_entities
                if e.confidence >= self.min_confidence
            ]

            processing_time = time.time() - start_time

            return ExtractionActionResponse(
                temporal_entities=[e.dict() for e in temporal_entities],
                spatial_entities=[e.dict() for e in spatial_entities],
                success=True,
                processing_time=processing_time,
                metadata={
                    "temporal_count": len(temporal_entities),
                    "spatial_count": len(spatial_entities),
                    "config": self.config,
                },
            )

        except Exception as e:
            logger.error(f"Action phase failed: {str(e)}")
            return ExtractionActionResponse(
                temporal_entities=[],
                spatial_entities=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _postprocess_temporal(
        self,
        mentions: List[Dict],
        document_text: str,
        temporal_context: Dict,
    ) -> List[TemporalEntity]:
        """Postprocess temporal mentions using normalization tools."""
        temporal_entities = []

        # Prepare for batch normalization
        mentions_with_context = [
            (m["text"], m.get("context", ""))
            for m in mentions
        ]

        # Batch normalize
        normalized_results = self.time_normalizer.normalize_batch(
            mentions_with_context,
            document_text=document_text
        )

        # Create TemporalEntity objects
        for i, mention in enumerate(mentions):
            if i >= len(normalized_results):
                continue

            normalized, temporal_type = normalized_results[i]

            # Skip intervals that couldn't be normalized
            if temporal_type == TemporalType.INTERVAL and "/" not in normalized:
                continue

            mention_text = mention["text"]
            start_char = document_text.find(mention_text)
            end_char = start_char + len(mention_text) if start_char != -1 else None

            # Handle intervals
            start_date, end_date = None, None
            if temporal_type == TemporalType.INTERVAL:
                start_date, end_date = self.time_normalizer.get_date_range(mention_text)

            entity = TemporalEntity(
                text=mention_text,
                normalized=normalized,
                temporal_type=temporal_type,
                confidence=0.90,
                start_char=start_char if start_char != -1 else None,
                end_char=end_char,
                start_date=start_date,
                end_date=end_date,
            )

            temporal_entities.append(entity)

        return temporal_entities

    def _postprocess_spatial(
        self,
        mentions: List[Dict],
        document_text: str,
        spatial_context: Dict,
    ) -> List[SpatialEntity]:
        """Postprocess spatial mentions using geocoding tools."""
        spatial_entities = []

        for mention in mentions:
            location_name = mention["text"]
            context = mention.get("context", "")
            location_type = mention.get("type", "other")

            # Extract parent region
            parent_region = None
            if spatial_context.get("regions"):
                parent_region = spatial_context["regions"][0]

            # Geocode using tool
            geocode_result = self.tool_registry.call_tool(
                "geocode_location",
                location_name=location_name,
                context=context,
                parent_region=parent_region
            )

            if geocode_result.get("success"):
                lat = geocode_result["latitude"]
                lon = geocode_result["longitude"]

                start_char = document_text.find(location_name)
                end_char = start_char + len(location_name) if start_char != -1 else None

                entity = SpatialEntity(
                    text=location_name,
                    latitude=lat,
                    longitude=lon,
                    location_type=location_type,
                    confidence=0.90,
                    start_char=start_char if start_char != -1 else None,
                    end_char=end_char,
                    address=None,
                    country=None,
                    admin_area=None,
                    locality=None,
                )

                spatial_entities.append(entity)

        return spatial_entities

    def run(self, environment: Dict[str, Any]) -> ExtractionActionResponse:
        """
        Run full observe-reason-act cycle.

        Args:
            environment: Dictionary with 'text' key

        Returns:
            ExtractionActionResponse
        """
        # Store observations in state for act phase
        observations = self.observe(environment)
        self.update_state("observations", observations)

        reasoning = self.reason(observations)
        return self.act(reasoning)
