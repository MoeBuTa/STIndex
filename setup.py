#!/usr/bin/env python
"""
STIndex Setup Script

This setup.py provides backwards compatibility with older build tools.
Modern installations should use pyproject.toml with pip >= 21.3.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    "langchain>=0.1.0",
    "langchain-openai>=0.0.5",
    "langchain-anthropic>=0.1.0",
    "pydantic>=2.0.0,<2.12",
    "spacy>=3.7.0",
    "dateparser>=1.2.0",
    "pendulum>=3.0.0",
    "geopy>=2.4.0",
    "pandas>=2.0.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
    "python-dotenv>=1.0.0",
    # Spacy model - automatically downloaded during installation
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'dev': [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
    ],
    'transformers': [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "accelerate>=0.25.0",
    ],
}

# Add 'all' option to install all extras
EXTRAS_REQUIRE['all'] = [dep for deps in EXTRAS_REQUIRE.values() for dep in deps]

setup(
    name="stindex",
    version="0.1.0",
    description="Spatiotemporal Index Extraction from Unstructured Text",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="STIndex Team",
    author_email="stindex@example.com",
    url="https://github.com/Ameame1/stindex",
    project_urls={
        "Documentation": "https://github.com/Ameame1/stindex#readme",
        "Source": "https://github.com/Ameame1/stindex",
        "Bug Tracker": "https://github.com/Ameame1/stindex/issues",
    },
    packages=find_packages(include=['stindex', 'stindex.*']),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'stindex=stindex.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp spatiotemporal information-extraction ner geocoding temporal-extraction",
    license="MIT",
    include_package_data=True,
    zip_safe=False,
)