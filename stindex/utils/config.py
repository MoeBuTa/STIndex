"""
Configuration loading for STIndex.

Loads configuration from YAML files with provider switching.
"""

import os
from typing import Any, Dict
from pathlib import Path

from dotenv import load_dotenv
import yaml

from stindex.utils.constants import (
    CFG_DIR,
    DEFAULT_LLM_PROVIDER,
)

# Load environment variables from .env file
load_dotenv()


def load_config_from_file(config_path: str = "extract") -> Dict[str, Any]:
    """
    Load complete configuration from a config file with LLM provider switching.

    This function loads the main config (extract.yml by default) and merges it with
    the provider-specific config (huggingface.yml, openai.yml, or claude.yml).

    Args:
        config_path: Path to config file (e.g., 'extract', 'cfg/extract.yml')
                    Defaults to 'extract' (cfg/extract.yml)

    Returns:
        Dictionary containing merged configuration
    """
    # If config_path doesn't end with .yml, add it and look in CFG_DIR
    if not config_path.endswith(('.yml', '.yaml')):
        config_file = Path(CFG_DIR) / f"{config_path}.yml"
    else:
        config_file = Path(config_path)

    try:
        # Load main config file
        with open(config_file, "r") as f:
            main_config = yaml.safe_load(f) or {}

        # Get LLM provider from main config
        llm_provider = main_config.get("llm_provider", DEFAULT_LLM_PROVIDER)

        # Load provider-specific config
        provider_config_file = Path(CFG_DIR) / f"{llm_provider}.yml"

        if provider_config_file.exists():
            with open(provider_config_file, "r") as f:
                provider_config = yaml.safe_load(f) or {}

            # Merge configs with proper handling of nested llm section
            # 1. Start with main_config
            merged_config = {**main_config}

            # 2. Merge llm section from provider config (preserving top-level llm_provider)
            if "llm" in provider_config:
                merged_config["llm"] = {
                    **merged_config.get("llm", {}),
                    **provider_config["llm"]
                }

            # 3. Ensure top-level llm_provider is preserved from main_config
            if "llm_provider" in main_config:
                merged_config["llm_provider"] = main_config["llm_provider"]
                # Also set it in the llm section for backward compatibility
                if "llm" not in merged_config:
                    merged_config["llm"] = {}
                merged_config["llm"]["llm_provider"] = main_config["llm_provider"]
        else:
            # If no provider config exists, use main config
            merged_config = main_config

        return merged_config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file {config_file}: {e}")
