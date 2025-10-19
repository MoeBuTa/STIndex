"""
Unit tests for time normalization module.
"""

import pytest
from stindex.utils.time_normalizer import TimeNormalizer
from stindex.agents.response.schemas import TemporalType


class TestTimeNormalizer:
    """Test cases for TimeNormalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.normalizer = TimeNormalizer()

    def test_absolute_date(self):
        """Test normalization of absolute dates."""
        normalized, temp_type = self.normalizer.normalize("March 15, 2022")
        assert normalized == "2022-03-15"
        assert temp_type == TemporalType.DATE

    def test_absolute_date_iso(self):
        """Test ISO format dates."""
        normalized, temp_type = self.normalizer.normalize("2024-01-15")
        assert normalized == "2024-01-15"
        assert temp_type == TemporalType.DATE

    def test_duration(self):
        """Test duration normalization."""
        normalized, temp_type = self.normalizer.normalize("3 days")
        assert temp_type == TemporalType.DURATION
        assert "P" in normalized or "D" in normalized

    def test_interval(self):
        """Test interval normalization."""
        normalized, temp_type = self.normalizer.normalize("from Monday to Friday")
        assert temp_type == TemporalType.INTERVAL

    def test_reference_date(self):
        """Test with reference date."""
        normalizer = TimeNormalizer(reference_date="2024-06-15")
        normalized, _ = normalizer.normalize("yesterday")
        assert normalized is not None

    def test_get_date_range(self):
        """Test date range extraction."""
        start, end = self.normalizer.get_date_range("from 2024-01-01 to 2024-12-31")
        assert start is not None
        assert end is not None
