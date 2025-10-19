"""
Extraction pipeline for spatiotemporal entity extraction.

Uses SpatioTemporalExtractorAgent with observe-reason-act pattern.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.agents.extractor import SpatioTemporalExtractorAgent
from stindex.pipeline.models import BatchExtractionResult, ExtractionResult
from stindex.utils.config import get_default_config, merge_configs


class ExtractionPipeline:
    """
    Main extraction pipeline for STIndex.

    Uses SpatioTemporalExtractorAgent to extract entities from text.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize extraction pipeline.

        Args:
            config: Configuration dictionary (merges with defaults)
        """
        # Merge with default config
        default_config = get_default_config()
        self.config = merge_configs(default_config, config or {})

        # Initialize agent
        self.agent = SpatioTemporalExtractorAgent(self.config)

        logger.info("Extraction pipeline initialized")
        logger.debug(f"Config: {self.config}")

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract spatiotemporal entities from text.

        Args:
            text: Input text

        Returns:
            ExtractionResult
        """
        start_time = time.time()

        try:
            # Run agent
            response = self.agent.run({"text": text})

            # Convert to ExtractionResult
            return ExtractionResult(
                text=text,
                temporal_entities=response.temporal_entities,
                spatial_entities=response.spatial_entities,
                success=response.success,
                error=response.error,
                processing_time=time.time() - start_time,
                metadata=response.metadata,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return ExtractionResult(
                text=text,
                temporal_entities=[],
                spatial_entities=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def extract_batch(self, texts: List[str]) -> BatchExtractionResult:
        """
        Extract from multiple texts.

        Args:
            texts: List of input texts

        Returns:
            BatchExtractionResult
        """
        start_time = time.time()

        results = []
        success_count = 0
        failure_count = 0

        for text in texts:
            result = self.extract(text)
            results.append(result)

            if result.success:
                success_count += 1
            else:
                failure_count += 1

        return BatchExtractionResult(
            results=results,
            total_count=len(texts),
            success_count=success_count,
            failure_count=failure_count,
            total_processing_time=time.time() - start_time,
        )

    def extract_from_file(self, file_path: str) -> ExtractionResult:
        """
        Extract from text file.

        Args:
            file_path: Path to text file

        Returns:
            ExtractionResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return ExtractionResult(
                text="",
                success=False,
                error=f"File not found: {file_path}",
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            return self.extract(text)

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            return ExtractionResult(
                text="",
                success=False,
                error=str(e),
            )
