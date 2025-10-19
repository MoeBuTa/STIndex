"""Tests for geocoder with spaCy-based parent region extraction."""

import pytest
from stindex.spatio.geocoder import GeocoderService


class TestGeocoderParentRegionExtraction:
    """Test spaCy NER-based parent region extraction."""

    @pytest.fixture
    def geocoder(self):
        """Create a GeocoderService instance."""
        return GeocoderService(enable_cache=False)

    def test_extract_parent_region_simple(self, geocoder):
        """Test extraction from simple 'City, Region' format."""
        context = "Cyclone hit Broome, Western Australia"
        result = geocoder._extract_parent_region(context)
        assert result == "Western Australia"

    def test_extract_parent_region_with_preposition(self, geocoder):
        """Test extraction when region mentioned with 'in' preposition."""
        context = "The storm moved from Perth to Broome in Western Australia"
        result = geocoder._extract_parent_region(context)
        assert result == "Western Australia"

    def test_extract_parent_region_country(self, geocoder):
        """Test extraction of country names."""
        test_cases = [
            ("Earthquake in Tokyo, Japan", "Japan"),
            ("Hurricane in Miami, Florida, USA", "USA"),
            ("Volcano eruption in Reykjavik, Iceland", "Iceland"),
        ]

        for context, expected in test_cases:
            result = geocoder._extract_parent_region(context)
            assert result == expected, f"Failed for context: {context}"

    def test_extract_parent_region_not_in_old_regex(self, geocoder):
        """Test that spaCy extracts regions not in old hardcoded regex list."""
        # These countries were NOT in the old regex pattern
        test_cases = [
            ("Floods in Bangkok, Thailand", "Thailand"),
            ("Earthquake in Santiago, Chile", "Chile"),
            ("Storm in Oslo, Norway", "Norway"),
            ("Typhoon in Manila, Philippines", "Philippines"),
        ]

        for context, expected in test_cases:
            result = geocoder._extract_parent_region(context)
            assert result == expected, f"Failed for context: {context}"

    def test_extract_parent_region_multiple_gpes(self, geocoder):
        """Test that it returns the last GPE when multiple are present."""
        context = "Wildfire near Sydney, New South Wales, Australia"
        result = geocoder._extract_parent_region(context)
        # Should return the last/broadest region
        assert result == "Australia"

    def test_extract_parent_region_no_region(self, geocoder):
        """Test extraction when no parent region is present."""
        context = "A storm occurred yesterday"
        result = geocoder._extract_parent_region(context)
        assert result is None

    def test_extract_parent_region_empty_context(self, geocoder):
        """Test extraction with empty context."""
        result = geocoder._extract_parent_region("")
        assert result is None


class TestGeocoderCache:
    """Test geocoder caching functionality."""

    @pytest.fixture
    def geocoder(self):
        """Create a GeocoderService with caching enabled."""
        return GeocoderService(enable_cache=True)

    def test_cache_initialization(self, geocoder):
        """Test that cache is properly initialized."""
        assert geocoder.enable_cache is True
        assert geocoder.cache is not None

    def test_cache_disabled(self):
        """Test geocoder with caching disabled."""
        geocoder = GeocoderService(enable_cache=False)
        assert geocoder.enable_cache is False
        assert geocoder.cache is None
