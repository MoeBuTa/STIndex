"""
Relative temporal expression resolver.

Resolves relative temporal expressions to absolute ISO 8601 format
using document metadata as context.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from loguru import logger


class RelativeTemporalResolver:
    """
    Resolves relative temporal expressions to absolute dates/times.

    Examples:
    - "Monday" + document_date="2025-10-25" → "2025-10-27"
    - "next week" + document_date="2025-10-25" → "2025-11-01"
    - "11:00am to 7:00pm" + document_date="2025-10-27" → "2025-10-27T11:00:00/2025-10-27T19:00:00"
    """

    # Day name mappings
    WEEKDAYS = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }

    def __init__(self, timezone: str = "UTC"):
        """
        Initialize resolver.

        Args:
            timezone: Default timezone (e.g., "UTC", "Australia/Perth")
        """
        self.timezone = timezone

    def resolve(
        self,
        temporal_text: str,
        document_date: Optional[str] = None,
        temporal_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Resolve relative temporal expression.

        Args:
            temporal_text: Original temporal text
            document_date: ISO 8601 document date (anchor point)
            temporal_type: Type hint ("date", "time", "datetime", "interval")

        Returns:
            Tuple of (normalized_value, resolved_type)
        """
        temporal_text_lower = temporal_text.lower().strip()

        # If already ISO 8601, return as-is
        if self._is_iso8601(temporal_text):
            return temporal_text, temporal_type or "datetime"

        # Try resolving weekday
        weekday_match = self._extract_weekday(temporal_text_lower)
        if weekday_match and document_date:
            resolved_date = self._resolve_weekday(weekday_match, document_date)

            # Check if there's also a time component
            time_match = self._extract_time_range(temporal_text)
            if time_match:
                start_time, end_time = time_match
                # Build interval
                start_dt = f"{resolved_date}T{start_time}:00"
                end_dt = f"{resolved_date}T{end_time}:00"
                return f"{start_dt}/{end_dt}", "interval"
            else:
                return resolved_date, "date"

        # Try resolving time range (without date)
        time_match = self._extract_time_range(temporal_text)
        if time_match and document_date:
            start_time, end_time = time_match
            start_dt = f"{document_date}T{start_time}:00"
            end_dt = f"{document_date}T{end_time}:00"
            return f"{start_dt}/{end_dt}", "interval"

        # Try resolving relative expressions like "yesterday", "tomorrow", "next week"
        relative_date = self._resolve_relative_date(temporal_text_lower, document_date)
        if relative_date:
            return relative_date, "date"

        # Fall back: return original
        logger.warning(f"Could not resolve relative temporal: {temporal_text}")
        return temporal_text, temporal_type or "unknown"

    def _is_iso8601(self, text: str) -> bool:
        """Check if text is already in ISO 8601 format."""
        # Simple check for ISO 8601 patterns
        iso_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # YYYY-MM-DDTHH:MM:SS
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # Interval
            r'^P\d+[YMWD]',  # Duration
        ]

        for pattern in iso_patterns:
            if re.match(pattern, text):
                return True

        return False

    def _extract_weekday(self, text: str) -> Optional[str]:
        """Extract weekday name from text."""
        for day_name in self.WEEKDAYS.keys():
            if day_name in text:
                return day_name
        return None

    def _resolve_weekday(self, weekday_name: str, document_date: str) -> str:
        """
        Resolve weekday to absolute date.

        Args:
            weekday_name: Day name (e.g., "monday")
            document_date: ISO 8601 document date

        Returns:
            ISO 8601 date of the most recent occurrence of that weekday
        """
        target_weekday = self.WEEKDAYS[weekday_name]

        # Parse document date
        doc_dt = datetime.fromisoformat(document_date)
        current_weekday = doc_dt.weekday()

        # Calculate days difference
        days_diff = (target_weekday - current_weekday) % 7

        # If days_diff is 0, it means same day - use that date
        # Otherwise, use most recent past occurrence
        if days_diff == 0:
            resolved_dt = doc_dt
        else:
            # Go to most recent past occurrence (within last week)
            if days_diff > 0:
                days_diff -= 7
            resolved_dt = doc_dt + timedelta(days=days_diff)

        return resolved_dt.date().isoformat()

    def _extract_time_range(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract time range from text.

        Examples:
        - "11:00am to 7:00pm" → ("11:00", "19:00")
        - "from 11:00 am to 7:00 pm" → ("11:00", "19:00")
        """
        # Pattern: HH:MM am/pm to HH:MM am/pm
        pattern = r'(\d{1,2}):?(\d{2})?\s*(am|pm)?\s*(?:to|-)?\s*(\d{1,2}):?(\d{2})?\s*(am|pm)?'
        match = re.search(pattern, text.lower())

        if match:
            hour1, min1, period1, hour2, min2, period2 = match.groups()

            # Convert to 24-hour format
            start_time = self._to_24hour(int(hour1), int(min1 or 0), period1)
            end_time = self._to_24hour(int(hour2), int(min2 or 0), period2)

            return (start_time, end_time)

        return None

    def _to_24hour(self, hour: int, minute: int, period: Optional[str]) -> str:
        """Convert to 24-hour format."""
        if period:
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0

        return f"{hour:02d}:{minute:02d}"

    def _resolve_relative_date(
        self,
        text: str,
        document_date: Optional[str]
    ) -> Optional[str]:
        """
        Resolve relative date expressions.

        Examples:
        - "yesterday" → document_date - 1 day
        - "tomorrow" → document_date + 1 day
        - "next week" → document_date + 7 days
        - "last month" → document_date - 30 days
        """
        if not document_date:
            return None

        doc_dt = datetime.fromisoformat(document_date)

        # Define relative date mappings
        relative_mappings = {
            "yesterday": -1,
            "today": 0,
            "tomorrow": 1,
            "next week": 7,
            "last week": -7,
            "next month": 30,
            "last month": -30,
        }

        for phrase, days_offset in relative_mappings.items():
            if phrase in text:
                resolved_dt = doc_dt + timedelta(days=days_offset)
                return resolved_dt.date().isoformat()

        return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    resolver = RelativeTemporalResolver(timezone="Australia/Perth")

    # Test cases
    test_cases = [
        ("Monday 27/10/2025 from 11:00 am to 7:00 pm", "2025-10-25", None),
        ("Monday", "2025-10-25", "date"),
        ("11:00am to 7:00pm", "2025-10-27", "interval"),
        ("yesterday", "2025-10-27", "date"),
        ("tomorrow", "2025-10-27", "date"),
        ("2022-03-15", "2025-10-27", "date"),  # Already ISO 8601
    ]

    for text, doc_date, temporal_type in test_cases:
        normalized, resolved_type = resolver.resolve(text, doc_date, temporal_type)
        print(f"Input: '{text}'")
        print(f"Document date: {doc_date}")
        print(f"Output: {normalized} (type: {resolved_type})")
        print()
