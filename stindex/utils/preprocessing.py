"""
Preprocessing utilities for spatiotemporal extraction.

Prepares text and extracts document-level context before LLM extraction.
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TextPreprocessor:
    """Preprocess text before extraction."""

    def __init__(self, normalize_whitespace: bool = True, detect_language: bool = True):
        """
        Initialize TextPreprocessor.

        Args:
            normalize_whitespace: Whether to normalize whitespace
            detect_language: Whether to detect language
        """
        self.normalize_whitespace = normalize_whitespace
        self.detect_language = detect_language

    def preprocess(self, text: str) -> Dict[str, any]:
        """
        Preprocess text and extract document-level context.

        Args:
            text: Input text

        Returns:
            Dictionary with preprocessed text and context
        """
        # Clean text
        cleaned_text = self._clean_text(text)

        # Extract document-level temporal context
        temporal_context = self._extract_temporal_context(cleaned_text)

        # Extract document-level spatial context
        spatial_context = self._extract_spatial_context(cleaned_text)

        # Detect language
        language = self._detect_language(cleaned_text) if self.detect_language else "en"

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "language": language,
            "temporal_context": temporal_context,
            "spatial_context": spatial_context,
            "metadata": {
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
            }
        }

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            text = text.strip()

        return text

    def _extract_temporal_context(self, text: str) -> Dict[str, any]:
        """
        Extract document-level temporal context.

        This helps with year inference for incomplete dates like "March 15".

        Args:
            text: Input text

        Returns:
            Temporal context dictionary
        """
        context = {
            "mentioned_years": [],
            "year_positions": [],
            "has_explicit_years": False,
        }

        # Extract all 4-digit years (1900-2099)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        year_matches = list(re.finditer(year_pattern, text))

        if year_matches:
            years = [int(match.group(1)) for match in year_matches]
            positions = [match.start() for match in year_matches]

            context["mentioned_years"] = sorted(set(years), reverse=True)  # Most recent first
            context["year_positions"] = positions
            context["has_explicit_years"] = True

        # Detect relative temporal markers
        relative_markers = [
            "yesterday", "today", "tomorrow",
            "last week", "this week", "next week",
            "last month", "this month", "next month",
            "last year", "this year", "next year"
        ]

        context["has_relative_dates"] = any(
            marker in text.lower() for marker in relative_markers
        )

        return context

    def _extract_spatial_context(self, text: str) -> Dict[str, any]:
        """
        Extract document-level spatial context.

        This helps with location disambiguation (e.g., "Broome" + "Western Australia").

        Args:
            text: Input text

        Returns:
            Spatial context dictionary
        """
        context = {
            "regions": [],
            "countries": [],
            "has_country_mentions": False,
        }

        # Common country patterns
        country_keywords = [
            r'\bAustralia\b', r'\bUSA\b', r'\bUnited States\b',
            r'\bChina\b', r'\bFrance\b', r'\bJapan\b',
            r'\bGermany\b', r'\bUK\b', r'\bUnited Kingdom\b'
        ]

        # Common region patterns (Australian focus)
        region_keywords = [
            r'\bWestern Australia\b', r'\bNew South Wales\b', r'\bVictoria\b',
            r'\bQueensland\b', r'\bSouth Australia\b', r'\bNorthern Territory\b',
            r'\bTasmania\b'
        ]

        # Find countries
        for pattern in country_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                context["countries"].extend(matches)
                context["has_country_mentions"] = True

        # Find regions
        for pattern in region_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                context["regions"].extend(matches)

        # Remove duplicates
        context["countries"] = list(set(context["countries"]))
        context["regions"] = list(set(context["regions"]))

        return context

    def _detect_language(self, text: str) -> str:
        """
        Detect text language.

        Simple heuristic-based detection. For production, use langdetect or similar.

        Args:
            text: Input text

        Returns:
            Language code (en/zh/other)
        """
        # Check for Chinese characters
        chinese_char_pattern = r'[\u4e00-\u9fff]+'
        if re.search(chinese_char_pattern, text):
            return "zh"

        # Default to English
        return "en"


class ContextEnricher:
    """Enrich extraction results with additional context."""

    def __init__(self):
        """Initialize ContextEnricher."""
        pass

    def enrich_temporal_mentions(
        self,
        mentions: List[Dict],
        document_context: Dict
    ) -> List[Dict]:
        """
        Enrich temporal mentions with document context.

        Args:
            mentions: List of temporal mentions
            document_context: Document-level temporal context

        Returns:
            Enriched mentions
        """
        enriched = []

        for mention in mentions:
            enriched_mention = mention.copy()

            # Add document years for year inference
            if document_context.get("mentioned_years"):
                enriched_mention["document_years"] = document_context["mentioned_years"]

            # Add relative date context
            if document_context.get("has_relative_dates"):
                enriched_mention["has_relative_context"] = True

            enriched.append(enriched_mention)

        return enriched

    def enrich_spatial_mentions(
        self,
        mentions: List[Dict],
        document_context: Dict
    ) -> List[Dict]:
        """
        Enrich spatial mentions with document context for disambiguation.

        Args:
            mentions: List of spatial mentions
            document_context: Document-level spatial context

        Returns:
            Enriched mentions
        """
        enriched = []

        for mention in mentions:
            enriched_mention = mention.copy()

            # Add parent regions from document context
            if document_context.get("regions"):
                enriched_mention["document_regions"] = document_context["regions"]

            # Add countries from document context
            if document_context.get("countries"):
                enriched_mention["document_countries"] = document_context["countries"]

            enriched.append(enriched_mention)

        return enriched


def prepare_text_for_extraction(text: str) -> Dict[str, any]:
    """
    Convenience function to preprocess text.

    Args:
        text: Input text

    Returns:
        Preprocessed result
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)
