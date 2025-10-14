"""
Spatial entity extraction using spaCy NER and enhanced geocoding with disambiguation.
"""

from typing import List, Optional

import spacy
from spacy.tokens import Doc

from stindex.models.schemas import SpatialEntity
from stindex.utils.enhanced_geocoder import EnhancedGeocoderService


class SpatialExtractor:
    """Extract spatial entities from text using spaCy and geocoding."""

    # Location entity types in spaCy
    LOCATION_LABELS = {"GPE", "LOC", "FAC", "ORG"}  # Geopolitical entity  # Location  # Facility

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        geocoder_provider: str = "nominatim",
        user_agent: str = "stindex",
        rate_limit: float = 1.0,
        enable_geocoding: bool = True,
        enable_cache: bool = True,
    ):
        """
        Initialize SpatialExtractor with enhanced geocoding.

        Args:
            spacy_model: spaCy model name
            geocoder_provider: Geocoding provider
            user_agent: User agent for geocoding
            rate_limit: Minimum seconds between geocoding requests
            enable_geocoding: Whether to geocode extracted locations
            enable_cache: Whether to enable geocoding cache
        """
        self.spacy_model = spacy_model
        self.enable_geocoding = enable_geocoding

        # Load spaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model: {spacy_model}")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

        # Initialize enhanced geocoder with disambiguation and caching
        if enable_geocoding:
            self.geocoder = EnhancedGeocoderService(
                user_agent=user_agent,
                enable_cache=enable_cache,
                rate_limit=rate_limit,
            )
        else:
            self.geocoder = None

    def extract(self, text: str, min_confidence: float = 0.5) -> List[SpatialEntity]:
        """
        Extract spatial entities from text with context-aware disambiguation.

        Args:
            text: Input text
            min_confidence: Minimum confidence threshold

        Returns:
            List of SpatialEntity objects
        """
        # Process text with spaCy
        doc = self.nlp(text)

        spatial_entities = []

        # Extract location entities
        for ent in doc.ents:
            if ent.label_ in self.LOCATION_LABELS:
                # Geocode with context-aware disambiguation if enabled
                if self.enable_geocoding and self.geocoder:
                    # Get context for disambiguation
                    context = self.get_location_context(text, ent.text, window=100)

                    # Extract parent region from context for disambiguation
                    coords = self.geocoder.get_coordinates(
                        ent.text,
                        context=context,
                        parent_region=None  # Will be extracted from context
                    )

                    if coords:
                        lat, lon = coords

                        entity = SpatialEntity(
                            text=ent.text,
                            latitude=lat,
                            longitude=lon,
                            location_type=ent.label_,
                            confidence=0.9,  # High confidence from NER + successful geocoding
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                            address=None,  # Could be enhanced
                            country=None,
                            admin_area=None,
                            locality=None,
                        )
                        spatial_entities.append(entity)
                    else:
                        # NER found it but geocoding failed
                        pass
                else:
                    # No geocoding - skip entities without coordinates
                    pass

        return spatial_entities

    def extract_with_llm_fallback(
        self, text: str, llm_extractor=None
    ) -> List[SpatialEntity]:
        """
        Extract spatial entities with LLM fallback for missed locations.

        Args:
            text: Input text
            llm_extractor: Optional LLM-based extractor for fallback

        Returns:
            List of SpatialEntity objects
        """
        # First, try spaCy extraction
        entities = self.extract(text)

        # If LLM fallback provided and few entities found, try LLM
        if llm_extractor and len(entities) < 2:
            # TODO: Implement LLM-based spatial extraction
            pass

        return entities

    def extract_locations_only(self, text: str) -> List[str]:
        """
        Extract location names without geocoding.

        Args:
            text: Input text

        Returns:
            List of location names
        """
        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in self.LOCATION_LABELS]

    def disambiguate_location(
        self, location_name: str, context: str
    ) -> Optional[SpatialEntity]:
        """
        Disambiguate location using context.

        Args:
            location_name: Location name to disambiguate
            context: Surrounding context

        Returns:
            SpatialEntity if successful
        """
        if not self.geocoder:
            return None

        # Try geocoding with context
        location_details = self.geocoder.get_location_details(f"{location_name}, {context}")

        if location_details:
            return SpatialEntity(
                text=location_name,
                latitude=location_details["latitude"],
                longitude=location_details["longitude"],
                location_type="GPE",
                confidence=0.85,
                address=location_details.get("address"),
                country=location_details.get("country"),
                admin_area=location_details.get("state"),
                locality=location_details.get("city"),
            )

        return None

    def batch_extract(self, texts: List[str]) -> List[List[SpatialEntity]]:
        """
        Extract spatial entities from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of lists of SpatialEntity objects
        """
        results = []

        # Use spaCy's pipe for efficient batch processing
        for doc in self.nlp.pipe(texts):
            entities = []

            for ent in doc.ents:
                if ent.label_ in self.LOCATION_LABELS and self.enable_geocoding:
                    location_details = self.geocoder.get_location_details(ent.text)

                    if location_details:
                        entity = SpatialEntity(
                            text=ent.text,
                            latitude=location_details["latitude"],
                            longitude=location_details["longitude"],
                            location_type=ent.label_,
                            confidence=0.9,
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                            address=location_details.get("address"),
                            country=location_details.get("country"),
                            admin_area=location_details.get("state"),
                            locality=location_details.get("city"),
                        )
                        entities.append(entity)

            results.append(entities)

        return results

    def get_location_context(self, text: str, location_name: str, window: int = 50) -> str:
        """
        Get context around a location mention.

        Args:
            text: Input text
            location_name: Location to find
            window: Context window size in characters

        Returns:
            Context string
        """
        start = text.find(location_name)
        if start == -1:
            return ""

        context_start = max(0, start - window)
        context_end = min(len(text), start + len(location_name) + window)

        return text[context_start:context_end]
