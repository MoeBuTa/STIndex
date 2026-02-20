"""
Configuration loading for STIndex.

Loads configuration from YAML files. LLM provider defaults are hardcoded here;
model name, temperature, max_tokens, and base_url can be overridden via CLI flags
or by passing values directly to DimensionalExtractor.
"""

from typing import Any, Dict
from pathlib import Path

from dotenv import load_dotenv
import yaml

from stindex.utils.constants import (
    CFG_EXTRACTION_INFERENCE_DIR,
    CFG_EXTRACTION_POSTPROCESS_DIR,
    CFG_PREPROCESS_DIR,
    DEFAULT_LLM_PROVIDER,
)

# Load environment variables from .env file
load_dotenv()

# Default LLM settings per provider.
# These are applied when no override is given via CLI or config.
PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "anthropic": {
        "model_name": "claude-sonnet-4-5-20250929",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "hf": {
        "model_name": "Qwen3-4B-Instruct-2507",
        "temperature": 0.0,
        "max_tokens": 4096,
        "base_url": "http://localhost:8001",
    },
    "hf_batch": {
        "model_name": "Qwen3-4B-Instruct-2507",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
}


def load_config_from_file(config_path: str = "extract") -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with provider defaults.

    Args:
        config_path: Path to config file (e.g., 'extract', 'cfg/extraction/inference/extract.yml')
                    Defaults to 'extract' (cfg/extraction/inference/extract.yml)

    Returns:
        Dictionary containing merged configuration (provider defaults + YAML overrides)
    """
    # Resolve config file path
    if not config_path.endswith(('.yml', '.yaml')):
        config_file = Path(CFG_EXTRACTION_INFERENCE_DIR) / f"{config_path}.yml"
    else:
        config_file = Path(config_path)

    try:
        with open(config_file, "r") as f:
            main_config = yaml.safe_load(f) or {}

        # Get LLM provider
        llm_provider = main_config.get("llm", {}).get("llm_provider") or DEFAULT_LLM_PROVIDER

        # Apply provider defaults, then let YAML values override
        provider_defaults = PROVIDER_DEFAULTS.get(llm_provider, {})
        merged_llm = {**provider_defaults, **main_config.get("llm", {})}
        merged_llm["llm_provider"] = llm_provider

        merged_config = {**main_config, "llm": merged_llm}
        return merged_config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file {config_file}: {e}")


def load_preprocess_config(config_name: str) -> Dict[str, Any]:
    """
    Load preprocessing configuration file.

    Args:
        config_name: Config name (e.g., 'chunking', 'parsing', 'scraping')
                    without .yml extension

    Returns:
        Dictionary containing configuration
    """
    config_file = Path(CFG_PREPROCESS_DIR) / f"{config_name}.yml"

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Preprocess config not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file {config_file}: {e}")


def load_postprocess_config(config_name: str) -> Dict[str, Any]:
    """
    Load postprocessing configuration file.

    Args:
        config_name: Config name (e.g., 'spatial', 'temporal', 'validation')
                    without .yml extension

    Returns:
        Dictionary containing configuration
    """
    config_file = Path(CFG_EXTRACTION_POSTPROCESS_DIR) / f"{config_name}.yml"

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Postprocess config not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file {config_file}: {e}")
