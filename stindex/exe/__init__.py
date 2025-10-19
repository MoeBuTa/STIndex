"""
Execution module for CLI commands.

This module contains the execution logic for CLI commands,
separated from the CLI interface definition.
"""

from .extract import execute_extract
from .utils import get_output_dir, save_result, display_json


__all__ = [
    "execute_extract",
    "get_output_dir",
    "save_result",
    "display_json",
]
