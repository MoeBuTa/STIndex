"""
Unified spatiotemporal extraction using LLM with agentic tool calling.

This extractor:
1. Uses a SINGLE LLM call to extract both temporal and spatial entities
2. Can call tools (normalize_temporal, geocode_location) for postprocessing
3. Includes preprocessing for context extraction
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel as LangChainBaseModel
from pydantic import Field as LangChainField

from stindex.models.schemas import TemporalEntity, SpatialEntity, TemporalType
from stindex.utils.enhanced_time_normalizer import EnhancedTimeNormalizer
from stindex.utils.enhanced_geocoder import EnhancedGeocoderService
from stindex.utils.preprocessing import TextPreprocessor, ContextEnricher
from stindex.utils.tools import ToolRegistry
from stindex.prompts.templates import get_combined_prompt


# Structured output schemas for LLM
class TemporalMention(LangChainBaseModel):
    """Temporal mention extracted by LLM."""
    text: str = LangChainField(description="The exact temporal expression found")
    context: str = LangChainField(description="Surrounding context (up to 20 words)")


class SpatialMention(LangChainBaseModel):
    """Spatial mention extracted by LLM."""
    text: str = LangChainField(description="The exact location name found")
    context: str = LangChainField(description="Surrounding context (up to 20 words)")
    type: str = LangChainField(
        description="Location type: country, city, region, landmark, address, feature, or other"
    )


class UnifiedExtractionResult(LangChainBaseModel):
    """Result from unified LLM extraction."""
    temporal_mentions: List[TemporalMention] = LangChainField(
        default_factory=list,
        description="All temporal expressions found"
    )
    spatial_mentions: List[SpatialMention] = LangChainField(
        default_factory=list,
        description="All location mentions found"
    )


class UnifiedSpatioTemporalExtractor:
    """
    Unified extractor for both temporal and spatial entities.

    Uses a SINGLE LLM call to extract both types of entities,
    then applies specialized postprocessing using tools.
    """

    def __init__(
        self,
        llm_provider: str = "local",
        model_name: str = "Qwen/Qwen3-8B",
        model_path: Optional[str] = None,
        temperature: float = 0.0,
        reference_date: Optional[str] = None,
        device: str = "auto",
        geocoder_provider: str = "nominatim",
        user_agent: str = "stindex",
        rate_limit: float = 1.0,
        enable_cache: bool = True,
        enable_preprocessing: bool = True,
        enable_tool_calling: bool = True,
    ):
        """
        Initialize UnifiedSpatioTemporalExtractor.

        Args:
            llm_provider: LLM provider (openai/anthropic/local)
            model_name: Model name
            model_path: Local model path (for local provider)
            temperature: Temperature for LLM
            reference_date: Reference date for relative time resolution
            device: Device for local models (cuda/cpu/auto)
            geocoder_provider: Geocoding provider
            user_agent: User agent for geocoding
            rate_limit: Rate limit for geocoding
            enable_cache: Enable geocoding cache
            enable_preprocessing: Enable text preprocessing
            enable_tool_calling: Enable LLM tool calling (for API models)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.enable_preprocessing = enable_preprocessing
        self.enable_tool_calling = enable_tool_calling

        # Initialize LLM
        if llm_provider == "local":
            from stindex.llm.local_llm import LocalQwenLLM

            self.llm = LocalQwenLLM(
                model_path=model_path,
                model_name=model_name,
                device=device,
                temperature=temperature,
            )

        elif llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        elif llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=model_name, temperature=temperature)

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Initialize postprocessing tools
        self.time_normalizer = EnhancedTimeNormalizer(reference_date=reference_date)

        self.geocoder = EnhancedGeocoderService(
            user_agent=user_agent,
            enable_cache=enable_cache,
            rate_limit=rate_limit,
        )

        # Initialize preprocessing
        if enable_preprocessing:
            self.preprocessor = TextPreprocessor()
            self.context_enricher = ContextEnricher()
        else:
            self.preprocessor = None
            self.context_enricher = None

        # Initialize tool registry
        self.tool_registry = ToolRegistry(
            time_normalizer=self.time_normalizer,
            geocoder=self.geocoder,
            enable_temporal=True,
            enable_spatial=True,
        )

    def extract(
        self,
        text: str,
        min_confidence: float = 0.5
    ) -> Dict[str, List]:
        """
        Extract both temporal and spatial entities in a SINGLE LLM call.

        Args:
            text: Input text
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with temporal_entities and spatial_entities lists
        """
        # Step 1: Preprocessing
        if self.enable_preprocessing and self.preprocessor:
            preprocessed = self.preprocessor.preprocess(text)
            cleaned_text = preprocessed["cleaned_text"]
            temporal_context = preprocessed["temporal_context"]
            spatial_context = preprocessed["spatial_context"]
        else:
            cleaned_text = text
            temporal_context = {}
            spatial_context = {}

        # Step 2: Unified LLM extraction (SINGLE CALL)
        extraction_result = self._extract_with_llm(cleaned_text)

        # Step 3: Enrich with document context
        if self.context_enricher:
            temporal_mentions = self.context_enricher.enrich_temporal_mentions(
                extraction_result["temporal_mentions"],
                temporal_context
            )
            spatial_mentions = self.context_enricher.enrich_spatial_mentions(
                extraction_result["spatial_mentions"],
                spatial_context
            )
        else:
            temporal_mentions = extraction_result["temporal_mentions"]
            spatial_mentions = extraction_result["spatial_mentions"]

        # Step 4: Postprocessing with tools
        temporal_entities = self._postprocess_temporal(
            temporal_mentions,
            cleaned_text,
            temporal_context
        )

        spatial_entities = self._postprocess_spatial(
            spatial_mentions,
            cleaned_text,
            spatial_context
        )

        # Filter by confidence
        temporal_entities = [e for e in temporal_entities if e.confidence >= min_confidence]
        spatial_entities = [e for e in spatial_entities if e.confidence >= min_confidence]

        return {
            "temporal_entities": temporal_entities,
            "spatial_entities": spatial_entities
        }

    def _extract_with_llm(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract both temporal and spatial entities with a SINGLE LLM call.

        Args:
            text: Input text

        Returns:
            Dictionary with temporal_mentions and spatial_mentions
        """
        if self.llm_provider == "local":
            return self._extract_with_local_llm(text)
        else:
            return self._extract_with_api_llm(text)

    def _extract_with_local_llm(self, text: str) -> Dict[str, List[Dict]]:
        """Extract using local LLM (Qwen3-8B)."""
        # Generate combined prompt
        prompt = get_combined_prompt(text)

        # Generate structured output
        result = self.llm.generate_structured(prompt)

        # Handle errors
        if "error" in result:
            print(f"Warning: {result.get('error')}")
            return {"temporal_mentions": [], "spatial_mentions": []}

        # Convert to dict format
        temporal_mentions = [
            {"text": m.get("text", ""), "context": m.get("context", "")}
            for m in result.get("temporal_mentions", [])
            if isinstance(m, dict)
        ]

        spatial_mentions = [
            {
                "text": m.get("text", ""),
                "context": m.get("context", ""),
                "type": m.get("type", "other")
            }
            for m in result.get("spatial_mentions", [])
            if isinstance(m, dict)
        ]

        return {
            "temporal_mentions": temporal_mentions,
            "spatial_mentions": spatial_mentions
        }

    def _extract_with_api_llm(self, text: str) -> Dict[str, List[Dict]]:
        """Extract using API-based LLM (OpenAI/Anthropic)."""
        from langchain_core.prompts import ChatPromptTemplate

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting both temporal and spatial information from text.

Extract ALL temporal expressions (dates, times, durations, intervals) and ALL spatial mentions (locations, places, addresses).

Output both in a single JSON response."""),
            ("human", get_combined_prompt(text))
        ])

        # Use structured output
        structured_llm = self.llm.with_structured_output(UnifiedExtractionResult)
        chain = prompt | structured_llm

        # Run extraction
        result = chain.invoke({"text": text})

        # Convert to dict format
        temporal_mentions = [
            {"text": m.text, "context": m.context}
            for m in result.temporal_mentions
        ]

        spatial_mentions = [
            {"text": m.text, "context": m.context, "type": m.type}
            for m in result.spatial_mentions
        ]

        return {
            "temporal_mentions": temporal_mentions,
            "spatial_mentions": spatial_mentions
        }

    def _postprocess_temporal(
        self,
        mentions: List[Dict],
        document_text: str,
        temporal_context: Dict
    ) -> List[TemporalEntity]:
        """
        Postprocess temporal mentions using normalization tools.

        Args:
            mentions: Temporal mentions from LLM
            document_text: Full document text
            temporal_context: Document-level temporal context

        Returns:
            List of TemporalEntity objects
        """
        temporal_entities = []

        # Prepare mentions with context for batch normalization
        mentions_with_context = [
            (m["text"], m.get("context", ""))
            for m in mentions
        ]

        # Batch normalize (context-aware year inference)
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

            # Find character offsets
            mention_text = mention["text"]
            start_char = document_text.find(mention_text)
            end_char = start_char + len(mention_text) if start_char != -1 else None

            # Handle intervals
            start_date, end_date = None, None
            if temporal_type == TemporalType.INTERVAL:
                start_date, end_date = self.time_normalizer.get_date_range(mention_text)

            # Create entity
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
        spatial_context: Dict
    ) -> List[SpatialEntity]:
        """
        Postprocess spatial mentions using geocoding tools.

        Args:
            mentions: Spatial mentions from LLM
            document_text: Full document text
            spatial_context: Document-level spatial context

        Returns:
            List of SpatialEntity objects
        """
        spatial_entities = []

        for mention in mentions:
            location_name = mention["text"]
            context = mention.get("context", "")
            location_type = mention.get("type", "other")

            # Extract parent region from context or document context
            parent_region = None
            if spatial_context.get("regions"):
                # Use first region as hint
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

                # Find character offsets
                start_char = document_text.find(location_name)
                end_char = start_char + len(location_name) if start_char != -1 else None

                # Create entity
                entity = SpatialEntity(
                    text=location_name,
                    latitude=lat,
                    longitude=lon,
                    location_type=location_type,
                    confidence=0.90,  # High confidence from LLM + successful geocoding
                    start_char=start_char if start_char != -1 else None,
                    end_char=end_char,
                    address=None,
                    country=None,
                    admin_area=None,
                    locality=None,
                )

                spatial_entities.append(entity)

        return spatial_entities
