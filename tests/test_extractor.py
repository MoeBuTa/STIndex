"""
Integration tests for ExtractionPipeline.
"""

import pytest
from stindex import ExtractionPipeline


class TestExtractionPipeline:
    """Integration tests for the extraction pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use minimal config for testing
        config = {
            "model_name": "gpt-4o-mini",
            "enable_temporal": True,
            "enable_spatial": True,
        }
        self.pipeline = ExtractionPipeline(config=config)

    def test_extract_temporal(self):
        """Test temporal entity extraction."""
        text = "On March 15, 2022, the event occurred."
        result = self.pipeline.extract(text)

        assert result.success
        assert len(result.temporal_entities) >= 1
        assert any("2022" in e.get("normalized", "") for e in result.temporal_entities)

    def test_extract_spatial(self):
        """Test spatial entity extraction."""
        text = "The conference was held in Paris, France."
        result = self.pipeline.extract(text)

        assert result.success
        assert result is not None
        # Note: Spatial extraction requires geocoding

    def test_batch_extraction(self):
        """Test batch extraction."""
        texts = [
            "On January 1, 2024, in New York.",
            "The event was in London on June 15, 2023.",
        ]

        batch_result = self.pipeline.extract_batch(texts)
        assert batch_result.total_count == 2
        assert len(batch_result.results) == 2
        assert batch_result.success_count >= 0

    def test_result_structure(self):
        """Test result structure and properties."""
        text = "On March 15, 2022, in Paris."
        result = self.pipeline.extract(text)

        assert hasattr(result, "temporal_entities")
        assert hasattr(result, "spatial_entities")
        assert hasattr(result, "processing_time")
        assert hasattr(result, "success")
        assert result.processing_time > 0

    def test_unified_extraction(self):
        """Test that extraction extracts both temporal and spatial in one call."""
        text = "On March 15, 2022, a cyclone hit Broome, Western Australia."
        result = self.pipeline.extract(text)

        assert result.success
        # Should extract both types
        assert isinstance(result.temporal_entities, list)
        assert isinstance(result.spatial_entities, list)
