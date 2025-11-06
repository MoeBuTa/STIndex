"""
Setup script for case study dependencies.

Ensures all required data files are downloaded (NLTK, spaCy models, etc.).
"""

import sys
from pathlib import Path

from loguru import logger


def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        import nltk
        logger.info("Downloading NLTK data packages...")

        # Required for unstructured package
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

        logger.info("✓ NLTK data downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False


def download_spacy_model():
    """Download required spaCy model."""
    try:
        import spacy

        # Check if model already exists
        try:
            spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model 'en_core_web_sm' already installed")
            return True
        except OSError:
            logger.info("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True)
            logger.info("✓ spaCy model downloaded")
            return True
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False


def setup_case_study_dependencies():
    """Setup all case study dependencies."""
    logger.info("=" * 80)
    logger.info("Setting up case study dependencies")
    logger.info("=" * 80)

    success = True

    # Download NLTK data
    if not download_nltk_data():
        success = False

    # Download spaCy model
    if not download_spacy_model():
        success = False

    if success:
        logger.info("\n✓ All dependencies setup successfully!")
    else:
        logger.warning("\n⚠ Some dependencies failed to setup")

    return success


if __name__ == "__main__":
    setup_case_study_dependencies()
