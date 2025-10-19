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

# Read dependencies from requirements.txt
def read_requirements():
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, 'requirements.txt')
    requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

INSTALL_REQUIRES = read_requirements()



setup(
    name="stindex",
    version="0.1.0",
    description="Spatiotemporal Index Extraction from Unstructured Text",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="STIndex Team",
    author_email="stindex@example.com",
    url="https://github.com/MoeBuTa/STIndex",
    project_urls={
        "Documentation": "https://github.com/MoeBuTa/STIndex#readme",
        "Source": "https://github.com/MoeBuTa/STIndex",
        "Bug Tracker": "https://github.com/MoeBuTa/STIndex/issues",
    },
    packages=find_packages(include=['stindex', 'stindex.*']),
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
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