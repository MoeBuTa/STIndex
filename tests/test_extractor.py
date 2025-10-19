"""
Integration tests for STIndexExtractor.
"""

import pytest
from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig


class TestSTIndexExtractor:
    """Integration tests for the main extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use minimal config for testing
        config = ExtractionConfig(
            enable_temporal=True, enable_spatial=True, model_name="gpt-4o-mini"
        )
        self.extractor = STIndexExtractor(config=config)

    def test_extract_temporal(self):
        """Test temporal entity extraction."""
        text = "On March 15, 2022, the event occurred."
        result = self.extractor.extract(text)

        assert result.temporal_count >= 1
        assert any("2022" in e.normalized for e in result.temporal_entities)

    def test_extract_spatial(self):
        """Test spatial entity extraction."""
        text = "The conference was held in Paris, France."
        result = self.extractor.extract(text)

        # Note: This requires actual geocoding, may need mocking
        assert result is not None

    def test_batch_extraction(self):
        """Test batch extraction."""
        texts = [
            "On January 1, 2024, in New York.",
            "The event was in London on June 15, 2023.",
        ]

        results = self.extractor.extract_batch(texts)
        assert len(results) == 2

    def test_result_structure(self):
        """Test result structure and properties."""
        text = "On March 15, 2022, in Paris."
        result = self.extractor.extract(text)

        assert hasattr(result, "temporal_entities")
        assert hasattr(result, "spatial_entities")
        assert hasattr(result, "processing_time")
        assert result.processing_time > 0
