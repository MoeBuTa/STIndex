"""
MCP (Model Context Protocol) server for STIndex.

Exposes the core spatiotemporal extraction functionality as an MCP tool
for integration with LLM applications like Claude Desktop.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from stindex.core.extraction import STIndexExtractor
from stindex.utils.config import load_config_from_file

# Initialize MCP server
mcp = FastMCP("STIndex")

# Global extractor instances (one per config)
_extractors: Dict[str, STIndexExtractor] = {}


def _get_extractor(config_name: str = "extract") -> STIndexExtractor:
    """Get or create an extractor instance for the given config."""
    global _extractors
    if config_name not in _extractors:
        logger.info(f"Initializing STIndexExtractor with config: {config_name}")
        _extractors[config_name] = STIndexExtractor(config_path=config_name)
    return _extractors[config_name]


# ============================================================================
# Tools - LLM-invocable functions
# ============================================================================


@mcp.tool()
def extract_spatiotemporal(
    text: str,
    config: str = "extract",
    include_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Extract temporal and spatial entities from unstructured text.

    This tool performs comprehensive spatiotemporal extraction:
    - Temporal: Dates, times, durations, intervals (normalized to ISO 8601)
    - Spatial: Locations with geocoded coordinates (latitude/longitude)

    Args:
        text: The input text to extract spatiotemporal information from
        config: Config name to use (openai/anthropic/hf/extract). Default: extract
        include_metadata: Whether to include extraction metadata (timing, etc.)

    Returns:
        Dictionary containing:
        - success: Whether extraction succeeded
        - temporal_entities: List of temporal expressions with normalized values
        - spatial_entities: List of locations with coordinates
        - metadata: Optional timing and statistics (if include_metadata=True)
    """
    try:
        extractor = _get_extractor(config)
        result = extractor.extract(text)

        # Build response
        response = {
            "success": result.success,
            "temporal_entities": [
                {
                    "text": entity.text,
                    "normalized": entity.normalized,
                    "temporal_type": entity.temporal_type.value,
                    "confidence": entity.confidence,
                }
                for entity in result.temporal_entities
            ],
            "spatial_entities": [
                {
                    "text": entity.text,
                    "location_type": entity.location_type.value,
                    "latitude": entity.latitude,
                    "longitude": entity.longitude,
                    "address": entity.address,
                    "confidence": entity.confidence,
                }
                for entity in result.spatial_entities
            ],
        }

        if include_metadata:
            response["metadata"] = {
                "processing_time_seconds": result.processing_time,
                "temporal_count": len(result.temporal_entities),
                "spatial_count": len(result.spatial_entities),
            }

        return response

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "temporal_entities": [],
            "spatial_entities": [],
        }


# ============================================================================
# Resources - Read-only data access
# ============================================================================


@mcp.resource("config://stindex/{config_name}")
def get_config(config_name: str) -> str:
    """
    Get STIndex configuration as JSON.

    Available configs: extract, openai, anthropic, hf

    Args:
        config_name: Name of config file (without .yml extension)

    Returns:
        JSON string of configuration
    """
    try:
        config = load_config_from_file(config_name)
        return json.dumps(config, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to load config: {e}"})


@mcp.resource("config://stindex/providers")
def list_providers() -> str:
    """
    List available LLM providers and their configurations.

    Returns:
        JSON string with provider information
    """
    providers = {
        "available_providers": ["openai", "anthropic", "hf"],
        "provider_descriptions": {
            "openai": "OpenAI GPT models (GPT-4, GPT-4o, GPT-3.5)",
            "anthropic": "Anthropic Claude models (Claude 3.5 Sonnet)",
            "hf": "HuggingFace local models (single or multi-GPU)",
        },
        "config_files": {
            "extract": "cfg/extract.yml (default - uses configured provider)",
            "openai": "cfg/openai.yml",
            "anthropic": "cfg/anthropic.yml",
            "hf": "cfg/hf.yml (HuggingFace, auto-detects single or multi-GPU)",
        },
    }
    return json.dumps(providers, indent=2)


# ============================================================================
# Prompts - Reusable templates
# ============================================================================


@mcp.prompt()
def analyze_spatiotemporal(
    document_text: str,
    focus: str = "both",
) -> str:
    """
    Generate a prompt for analyzing spatiotemporal information in a document.

    Args:
        document_text: The document text to analyze
        focus: What to focus on - "temporal", "spatial", or "both"

    Returns:
        Formatted prompt for LLM analysis
    """
    focus_instructions = {
        "temporal": "Focus on extracting and analyzing all temporal expressions (dates, times, durations, intervals).",
        "spatial": "Focus on extracting and analyzing all spatial entities (locations, places, regions).",
        "both": "Extract and analyze both temporal and spatial information.",
    }

    instruction = focus_instructions.get(focus, focus_instructions["both"])

    return f"""Analyze the following document for spatiotemporal information.

{instruction}

Document:
{document_text}

Please extract all relevant information and provide:
1. A summary of the temporal context (when events occurred)
2. A summary of the spatial context (where events occurred)
3. Any notable patterns or relationships between time and place
"""


# ============================================================================
# Server entry point
# ============================================================================


def main():
    """Run the MCP server."""
    logger.info("Starting STIndex MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
