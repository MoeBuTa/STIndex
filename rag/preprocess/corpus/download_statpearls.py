"""
Download StatPearls corpus from NCBI FTP.

Based on MedRAG's approach: https://github.com/Teddy-XiongGZ/MedRAG
StatPearls cannot be redistributed per privacy policy, so we download from NCBI directly.

Usage:
    python -m rag.preprocess.corpus.download_statpearls
    python -m rag.preprocess.corpus.download_statpearls --output data/original/medcorp/raw/statpearls
"""

import argparse
import subprocess
from pathlib import Path

from loguru import logger


def download_statpearls(output_dir: str = "data/original/medcorp/raw/statpearls") -> Path:
    """
    Download StatPearls from NCBI FTP and extract.

    Follows MedRAG process:
    1. wget from NCBI FTP
    2. tar extract

    Args:
        output_dir: Directory to save downloaded and extracted files

    Returns:
        Path to extracted StatPearls directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # NCBI FTP URL for StatPearls
    ncbi_url = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
    tar_file = output_path / "statpearls_NBK430685.tar.gz"

    logger.info("=" * 80)
    logger.info("Downloading StatPearls corpus from NCBI")
    logger.info("=" * 80)
    logger.info(f"Source: {ncbi_url}")
    logger.info(f"Destination: {output_path}")

    # Check if already downloaded
    if tar_file.exists():
        logger.warning(f"Archive already exists: {tar_file}")
        logger.info("Skipping download. Delete file to re-download.")
    else:
        # Download from NCBI
        logger.info(f"Downloading StatPearls archive (~6.2 GB)...")
        try:
            subprocess.run(
                ["wget", ncbi_url, "-P", str(output_path)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"✓ Downloaded: {tar_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download StatPearls: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

    # Extract archive
    extracted_marker = output_path / ".extracted"
    if extracted_marker.exists():
        logger.warning("Archive already extracted (found .extracted marker)")
        logger.info("Skipping extraction. Delete .extracted file to re-extract.")
    else:
        logger.info(f"Extracting archive...")
        try:
            subprocess.run(
                ["tar", "-xzvf", str(tar_file), "-C", str(output_path)],
                check=True,
                capture_output=True,
                text=True
            )
            # Create marker file
            extracted_marker.touch()
            logger.info(f"✓ Extracted to: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract StatPearls: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise

    # Count NXML files
    nxml_files = list(output_path.glob("**/*.nxml"))
    logger.info("=" * 80)
    logger.info(f"✓ StatPearls download complete")
    logger.info(f"  Location: {output_path}")
    logger.info(f"  NXML files found: {len(nxml_files)}")
    logger.info("=" * 80)
    logger.info("Next step: Run preprocessing")
    logger.info("  python -m rag.preprocess.corpus.preprocess_statpearls")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download StatPearls corpus from NCBI FTP"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/original/medcorp/raw/statpearls",
        help="Output directory for StatPearls files"
    )
    args = parser.parse_args()

    download_statpearls(args.output)


if __name__ == "__main__":
    main()
