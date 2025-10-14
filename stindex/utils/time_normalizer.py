"""
Time normalization utilities using dateparser and pendulum.
"""

import re
from datetime import datetime
from typing import Optional, Tuple

import dateparser
import pendulum
from pendulum import DateTime

from stindex.models.schemas import TemporalType


class TimeNormalizer:
    """Normalize temporal expressions to ISO 8601 format."""

    def __init__(self, reference_date: Optional[str] = None):
        """
        Initialize TimeNormalizer.

        Args:
            reference_date: Reference date for resolving relative times (ISO format)
        """
        self.reference_date = self._parse_reference_date(reference_date)

        # Dateparser settings
        self.dateparser_settings = {
            "PREFER_DATES_FROM": "past",
            "RELATIVE_BASE": self.reference_date if self.reference_date else datetime.now(),
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "UTC",
        }

    def _parse_reference_date(self, reference_date: Optional[str]) -> Optional[datetime]:
        """Parse reference date string."""
        if reference_date:
            try:
                return datetime.fromisoformat(reference_date)
            except ValueError:
                return None
        return None

    def normalize(
        self, temporal_text: str, context: Optional[str] = None
    ) -> Tuple[str, TemporalType]:
        """
        Normalize temporal expression to ISO 8601 format.

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

        # Parse using dateparser
        parsed = dateparser.parse(temporal_text, settings=self.dateparser_settings)

        if parsed is None:
            # Fallback: try pendulum
            try:
                parsed = pendulum.parse(temporal_text, strict=False)
            except Exception:
                # Return as-is if unable to parse
                return temporal_text, TemporalType.RELATIVE

        # Determine type and format
        temporal_type = self._determine_type(temporal_text, parsed)
        normalized = self._format_datetime(parsed, temporal_type)

        return normalized, temporal_type

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
        interval_indicators = [
            "from.*to",
            "between.*and",
            "since",
            "until",
            "through",
            "-",  # Date range like "2022-01-01 - 2022-01-31"
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in interval_indicators)

    def _normalize_interval(self, text: str) -> str:
        """Normalize interval to ISO 8601 format."""
        # Split interval
        separators = [" to ", " - ", " until ", " through "]
        start_text, end_text = None, None

        for sep in separators:
            if sep in text.lower():
                parts = re.split(sep, text, flags=re.IGNORECASE)
                if len(parts) == 2:
                    start_text = parts[0].replace("from ", "").replace("between ", "")
                    end_text = parts[1].replace("and ", "")
                    break

        if start_text and end_text:
            start = dateparser.parse(start_text, settings=self.dateparser_settings)
            end = dateparser.parse(end_text, settings=self.dateparser_settings)

            if start and end:
                return f"{start.isoformat()}/{end.isoformat()}"

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
