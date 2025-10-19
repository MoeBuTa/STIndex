"""
Enhanced time normalization with context-aware year inference.

Based on research from:
- Temporal coreference resolution (ACL 2024)
- SUTime/HeidelTime approaches
- Context-aware temporal normalization
"""

import re
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

import dateparser
import pendulum

from stindex.agents.response.schemas import TemporalType


class TemporalContext:
    """Maintains temporal context for year inference and coreference resolution."""

    def __init__(self):
        self.reference_years: List[int] = []  # Years mentioned in context
        self.reference_date: Optional[datetime] = None
        self.temporal_anchors: Dict[str, datetime] = {}  # Named temporal references

    def add_year(self, year: int):
        """Add a year to the context."""
        if year not in self.reference_years:
            self.reference_years.append(year)

    def get_most_recent_year(self) -> Optional[int]:
        """Get the most recently mentioned year."""
        return self.reference_years[-1] if self.reference_years else None

    def infer_year(self, month: int, day: Optional[int] = None) -> Optional[int]:
        """
        Infer year for incomplete dates based on context.

        Uses heuristics:
        1. If a year was mentioned in context, use it
        2. If reference date is set, use its year
        3. Otherwise, use current year

        Args:
            month: Month number (1-12)
            day: Optional day number

        Returns:
            Inferred year
        """
        # Strategy 1: Use most recent year from context
        if self.reference_years:
            return self.get_most_recent_year()

        # Strategy 2: Use reference date's year
        if self.reference_date:
            return self.reference_date.year

        # Strategy 3: Use current year
        return datetime.now().year


class EnhancedTimeNormalizer:
    """
    Enhanced time normalizer with context-aware year inference.

    Improvements over basic TimeNormalizer:
    1. Context-aware year inference for incomplete dates
    2. Temporal coreference tracking
    3. Better handling of relative dates in context
    4. Improved interval parsing
    """

    def __init__(self, reference_date: Optional[str] = None):
        """
        Initialize EnhancedTimeNormalizer.

        Args:
            reference_date: Reference date for resolving relative times (ISO format)
        """
        self.context = TemporalContext()
        self.reference_date = self._parse_reference_date(reference_date)

        if self.reference_date:
            self.context.reference_date = self.reference_date
            self.context.add_year(self.reference_date.year)

        # Dateparser settings
        self.dateparser_settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": self.reference_date if self.reference_date else datetime.now(),
            "RETURN_AS_TIMEZONE_AWARE": False,  # Avoid timezone issues with strftime
        }

    def _parse_reference_date(self, reference_date: Optional[str]) -> Optional[datetime]:
        """Parse reference date string."""
        if reference_date:
            try:
                return datetime.fromisoformat(reference_date)
            except ValueError:
                return None
        return None

    def normalize_batch(
        self,
        temporal_texts: List[Tuple[str, str]],
        document_text: Optional[str] = None
    ) -> List[Tuple[str, TemporalType]]:
        """
        Normalize multiple temporal expressions with shared context.

        This is the key method for handling temporal coreference - by processing
        all temporal expressions together, we can propagate year information.

        Args:
            temporal_texts: List of (text, context) tuples
            document_text: Full document text for additional context extraction

        Returns:
            List of (normalized, type) tuples
        """
        # Reset context for new document
        self.context = TemporalContext()
        if self.reference_date:
            self.context.reference_date = self.reference_date
            self.context.add_year(self.reference_date.year)

        # First pass: extract all absolute years from document
        if document_text:
            self._extract_contextual_years(document_text)

        # Also extract from all temporal texts
        for text, _ in temporal_texts:
            year = self._extract_year_from_text(text)
            if year:
                self.context.add_year(year)

        # Second pass: normalize each expression with context
        results = []
        for text, context in temporal_texts:
            normalized, temp_type = self.normalize(text, context)
            results.append((normalized, temp_type))

            # Update context after each normalization
            year = self._extract_year_from_text(normalized)
            if year:
                self.context.add_year(year)

        return results

    def _extract_contextual_years(self, text: str):
        """Extract all 4-digit years from text for context."""
        # Find 4-digit years (1900-2099)
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years = re.findall(year_pattern, text)
        for year in years:
            self.context.add_year(int(year))

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract year from text if present."""
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        match = re.search(year_pattern, text)
        if match:
            return int(match.group(1))
        return None

    def normalize(
        self, temporal_text: str, context: Optional[str] = None
    ) -> Tuple[str, TemporalType]:
        """
        Normalize temporal expression to ISO 8601 format with context awareness.

        Args:
            temporal_text: Temporal expression to normalize
            context: Surrounding context for disambiguation

        Returns:
            Tuple of (normalized_time, temporal_type)
        """
        # Try duration pattern first
        if self._is_duration(temporal_text):
            normalized = self._normalize_duration(temporal_text)
            return normalized, TemporalType.DURATION

        # Try interval pattern
        if self._is_interval(temporal_text):
            normalized = self._normalize_interval(temporal_text)
            return normalized, TemporalType.INTERVAL

        # Check if we need year inference (incomplete date like "March 17")
        needs_year_inference = self._needs_year_inference(temporal_text)

        # Parse using dateparser
        parsed = dateparser.parse(temporal_text, settings=self.dateparser_settings)

        if parsed is None:
            # Fallback: try pendulum
            try:
                parsed = pendulum.parse(temporal_text, strict=False)
            except Exception:
                # Return as-is if unable to parse
                return temporal_text, TemporalType.RELATIVE

        # Apply year inference if needed
        if needs_year_inference and parsed:
            parsed = self._apply_year_inference(temporal_text, parsed, context)

        # Determine type and format
        temporal_type = self._determine_type(temporal_text, parsed)
        normalized = self._format_datetime(parsed, temporal_type)

        return normalized, temporal_type

    def _needs_year_inference(self, text: str) -> bool:
        """
        Check if text needs year inference.

        Examples that need inference:
        - "March 17" (month + day, no year)
        - "March" (just month)
        - "Monday" (just day of week)
        """
        # Has explicit 4-digit year?
        if re.search(r'\b(19\d{2}|20\d{2})\b', text):
            return False

        # Has month name or number without year?
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b'
        if re.search(month_pattern, text.lower()):
            return True

        # Has day pattern without year?
        day_pattern = r'\b(\d{1,2})(st|nd|rd|th)?\b'
        if re.search(day_pattern, text.lower()) and not re.search(r'\d{4}', text):
            return True

        return False

    def _apply_year_inference(
        self,
        text: str,
        parsed: datetime,
        context: Optional[str]
    ) -> datetime:
        """
        Apply year inference based on context.

        Strategy (inspired by SUTime/HeidelTime):
        1. Extract year from surrounding context
        2. Use most recent year from document context
        3. Apply context year if original text lacks explicit year
        """
        # Extract year from context if available
        if context:
            year = self._extract_year_from_text(context)
            if year:
                self.context.add_year(year)

        # Get inferred year from context
        inferred_year = self.context.infer_year(parsed.month, parsed.day)

        # Apply inferred year if we have one from context
        # The _needs_year_inference() check already determined that
        # the original text lacks an explicit year, so we should
        # apply the context year unconditionally
        if inferred_year:
            parsed = parsed.replace(year=inferred_year)

        return parsed

    def _is_duration(self, text: str) -> bool:
        """Check if text represents a duration."""
        duration_patterns = [
            r"\d+\s*(year|month|week|day|hour|minute|second)s?",
            r"P\d+[YMWD]",  # ISO 8601 duration
            r"PT\d+[HMS]",  # ISO 8601 time duration
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in duration_patterns)

    def _normalize_duration(self, text: str) -> str:
        """Normalize duration to ISO 8601 format."""
        # Already in ISO format
        if text.startswith("P"):
            return text

        # Parse duration components
        text_lower = text.lower()
        duration_map = {
            "year": "Y",
            "month": "M",
            "week": "W",
            "day": "D",
            "hour": "H",
            "minute": "M",
            "second": "S",
        }

        result = "P"
        time_part = "T"
        has_time = False

        for unit, symbol in duration_map.items():
            pattern = rf"(\d+)\s*{unit}s?"
            match = re.search(pattern, text_lower)
            if match:
                value = match.group(1)
                if unit in ["hour", "minute", "second"]:
                    time_part += f"{value}{symbol}"
                    has_time = True
                else:
                    result += f"{value}{symbol}"

        if has_time:
            result += time_part

        return result if len(result) > 1 else "P0D"

    def _is_interval(self, text: str) -> bool:
        """Check if text represents a time interval."""
        interval_patterns = [
            r"from.+to",  # "from X to Y"
            r"between.+and",  # "between X and Y"
            r"\bsince\b",  # "since X"
            r"\buntil\b",  # "until X"
            r"\bthrough\b",  # "through X"
            r"\d{4}-\d{2}-\d{2}\s*-\s*\d{4}-\d{2}-\d{2}",  # Date range like "2022-01-01 - 2022-01-31"
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in interval_patterns)

    def _normalize_interval(self, text: str) -> str:
        """Normalize interval to ISO 8601 format with year inference."""
        # Split interval
        separators = [" to ", " - ", " until ", " through "]
        start_text, end_text = None, None

        for sep in separators:
            if sep in text.lower():
                parts = re.split(sep, text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    # Clean the parts
                    start_text = parts[0].lower().replace("from", "").replace("between", "").strip()
                    end_text = parts[1].lower().replace("and", "").strip()
                    break

        if start_text and end_text:
            # Check if dates need year inference
            start_needs_year = self._needs_year_inference(start_text)
            end_needs_year = self._needs_year_inference(end_text)

            # Parse dates
            start = dateparser.parse(start_text, settings=self.dateparser_settings)
            end = dateparser.parse(end_text, settings=self.dateparser_settings)

            if start and end:
                # Apply year inference if needed
                if start_needs_year:
                    inferred_year = self.context.infer_year(start.month, start.day)
                    if inferred_year:
                        start = start.replace(year=inferred_year)

                if end_needs_year:
                    # For end date, use start date's year if available
                    if start:
                        inferred_year = start.year
                    else:
                        inferred_year = self.context.infer_year(end.month, end.day)

                    if inferred_year:
                        end = end.replace(year=inferred_year)

                # Return ISO 8601 interval format
                return f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        return text

    def _determine_type(self, text: str, parsed: datetime) -> TemporalType:
        """Determine the type of temporal expression."""
        text_lower = text.lower()

        # Check for time indicators
        time_indicators = ["am", "pm", ":", "hour", "minute", "o'clock"]
        has_time = any(indicator in text_lower for indicator in time_indicators)

        # Check for date indicators
        date_indicators = ["year", "month", "day", "january", "february", "monday", "tuesday"]
        has_date = any(indicator in text_lower for indicator in date_indicators)

        # Check for relative expressions
        relative_indicators = ["ago", "last", "next", "yesterday", "tomorrow", "now"]
        is_relative = any(indicator in text_lower for indicator in relative_indicators)

        if is_relative:
            return TemporalType.RELATIVE
        elif has_date and has_time:
            return TemporalType.DATETIME
        elif has_time:
            return TemporalType.TIME
        else:
            return TemporalType.DATE

    def _format_datetime(self, dt: datetime, temporal_type: TemporalType) -> str:
        """Format datetime according to type."""
        if temporal_type == TemporalType.DATE:
            return dt.strftime("%Y-%m-%d")
        elif temporal_type == TemporalType.TIME:
            return dt.strftime("%H:%M:%S")
        elif temporal_type == TemporalType.DATETIME:
            return dt.isoformat()
        else:
            return dt.isoformat()

    def parse_relative_time(self, text: str) -> Optional[str]:
        """
        Parse relative time expressions.

        Args:
            text: Relative time expression like "3 days ago", "next week"

        Returns:
            ISO formatted datetime string
        """
        parsed = dateparser.parse(text, settings=self.dateparser_settings)
        if parsed:
            return parsed.isoformat()
        return None

    def get_date_range(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract date range from text.

        Args:
            text: Text potentially containing date range

        Returns:
            Tuple of (start_date, end_date) in ISO format
        """
        if self._is_interval(text):
            normalized = self._normalize_interval(text)
            if "/" in normalized:
                parts = normalized.split("/")
                return parts[0], parts[1]

        return None, None
