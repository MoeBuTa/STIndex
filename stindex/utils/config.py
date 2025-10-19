"""
Configuration loading utilities for STIndex.

Loads configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

from stindex.utils.constants import *

# Load environment variables
load_dotenv()

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
CFG_DIR = PROJECT_ROOT / "cfg"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = Path.home() / CACHE_DIR_NAME


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yml extension)

    Returns:
        Configuration dictionary
    """
    config_path = CFG_DIR / f"{config_name}.yml"

    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {config_path}")
            return config
        else:
            logger.warning(f"Config file not found at {config_path}. Using defaults.")
            return get_default_config()

    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "llm_provider": os.getenv(ENV_LLM_PROVIDER, LLM_PROVIDER_LOCAL),
        "model_name": os.getenv(ENV_MODEL_NAME, DEFAULT_LOCAL_MODEL),
        "model_path": os.getenv(ENV_MODEL_PATH),
        "device": os.getenv(ENV_DEVICE, DEVICE_AUTO),
        "temperature": float(os.getenv(ENV_TEMPERATURE, "0.0")),
        "reference_date": os.getenv(ENV_REFERENCE_DATE),
        "enable_temporal": os.getenv(ENV_ENABLE_TEMPORAL, "true").lower() == "true",
        "enable_spatial": os.getenv(ENV_ENABLE_SPATIAL, "true").lower() == "true",
        "enable_cache": os.getenv(ENV_ENABLE_CACHE, "true").lower() == "true",
        "use_agentic": os.getenv(ENV_USE_AGENTIC, "true").lower() == "true",
        "geocoder": os.getenv(ENV_GEOCODER, GEOCODER_NOMINATIM),
        "user_agent": os.getenv(ENV_USER_AGENT, DEFAULT_USER_AGENT),
        "rate_limit": float(os.getenv(ENV_RATE_LIMIT, str(DEFAULT_RATE_LIMIT))),
        "min_confidence": float(os.getenv(ENV_MIN_CONFIDENCE, str(DEFAULT_MIN_CONFIDENCE))),
    }


def get_env_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables only.

    Returns:
        Configuration dictionary from environment
    """
    return get_default_config()


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    merged = base.copy()
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


# API Keys (loaded from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def ensure_directories():
    """Ensure necessary directories exist."""
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / GEOCODE_CACHE_DIR).mkdir(parents=True, exist_ok=True)


# Ensure directories on import
ensure_directories()
