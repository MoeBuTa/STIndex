#!/usr/bin/env python
"""
STIndex Setup Script

This setup.py provides backwards compatibility with older build tools.
Modern installations should use pyproject.toml with pip >= 21.3.

All metadata is read from pyproject.toml (single source of truth).
"""

from setuptools import setup

# For modern pip (>=21.3), pyproject.toml is automatically used.
# This setup.py only provides backward compatibility.
# All configuration is defined in pyproject.toml.

setup()