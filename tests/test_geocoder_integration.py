"""Integration tests for geocoder with spaCy parent region extraction in the full pipeline."""

import pytest
from unittest.mock import Mock, patch
from stindex.spatio.geocoder import GeocoderService


class TestGeocoderIntegration:
    """Test geocoder integration with extraction workflow."""

    @pytest.fixture
    def geocoder(self):
        """Create a GeocoderService instance."""
        return GeocoderService(enable_cache=False)

    def test_uses_llm_parent_region_when_provided(self, geocoder):
        """Test that LLM-provided parent_region is used first."""
        text = "Cyclone hit Broome, Western Australia"

        # Mock the geocoding API to verify the query
        with patch.object(geocoder, '_geocode_with_retry') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = -17.96
            mock_result.longitude = 122.24
            mock_result.address = "Broome, WA, Australia"
            mock_geocode.return_value = mock_result

            # LLM provides parent_region
            coords = geocoder.get_coordinates(
                location="Broome",
                context=text,
                parent_region="Western Australia"  # From LLM
            )

            # Should use LLM's parent_region
            assert coords == (-17.96, 122.24)
            # Verify it searched for "Broome, Western Australia"
            mock_geocode.assert_called_once_with("Broome, Western Australia")

    def test_fallback_to_spacy_when_llm_parent_region_missing(self, geocoder):
        """Test that spaCy extraction is used when LLM parent_region is None."""
        text = "Cyclone hit Broome, Western Australia"

        # Mock the geocoding API
        with patch.object(geocoder, '_geocode_with_retry') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = -17.96
            mock_result.longitude = 122.24
            mock_result.address = "Broome, WA, Australia"
            mock_geocode.return_value = mock_result

            # LLM failed to extract parent_region (None)
            coords = geocoder.get_coordinates(
                location="Broome",
                context=text,
                parent_region=None  # LLM missed it
            )

            # Should fallback to spaCy extraction from context
            assert coords == (-17.96, 122.24)
            # spaCy should extract "Western Australia" from context
            mock_geocode.assert_called_once_with("Broome, Western Australia")

    def test_fallback_with_country_only(self, geocoder):
        """Test spaCy fallback with country names."""
        text = "Earthquake in Tokyo, Japan"

        with patch.object(geocoder, '_geocode_with_retry') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = 35.68
            mock_result.longitude = 139.65
            mock_result.address = "Tokyo, Japan"
            mock_geocode.return_value = mock_result

            coords = geocoder.get_coordinates(
                location="Tokyo",
                context=text,
                parent_region=None
            )

            assert coords == (35.68, 139.65)
            # Should extract "Japan" from context
            mock_geocode.assert_called_once_with("Tokyo, Japan")

    def test_handles_no_context_gracefully(self, geocoder):
        """Test that geocoder works even without context."""
        with patch.object(geocoder, '_geocode_with_retry') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = -17.96
            mock_result.longitude = 122.24
            mock_result.address = "Broome, WA, Australia"
            mock_geocode.return_value = mock_result

            coords = geocoder.get_coordinates(
                location="Broome",
                context=None,
                parent_region=None
            )

            # Should just search for the location name alone
            assert coords == (-17.96, 122.24)
            mock_geocode.assert_called_once_with("Broome")

    def test_preposition_based_extraction(self, geocoder):
        """Test spaCy extraction with preposition-based patterns."""
        text = "The storm moved from Perth to Broome in Western Australia"

        with patch.object(geocoder, '_geocode_with_retry') as mock_geocode:
            mock_result = Mock()
            mock_result.latitude = -17.96
            mock_result.longitude = 122.24
            mock_result.address = "Broome, WA, Australia"
            mock_geocode.return_value = mock_result

            coords = geocoder.get_coordinates(
                location="Broome",
                context=text,
                parent_region=None
            )

            # Should extract "Western Australia" using preposition pattern
            assert coords == (-17.96, 122.24)
            mock_geocode.assert_called_once_with("Broome, Western Australia")
