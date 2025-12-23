"""
Preprocess StatPearls corpus to our JSONL format.

Adapted from MedRAG: https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/data/statpearls.py
Parses NCBI NXML files, chunks them, and outputs to our format.

Usage:
    python -m rag.preprocess.corpus.preprocess_statpearls
    python -m rag.preprocess.corpus.preprocess_statpearls \\
        --input data/original/medcorp/raw/statpearls \\
        --output data/original/medcorp/raw/statpearls_processed.jsonl
"""

import argparse
import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from loguru import logger
from tqdm import tqdm


def extract_text(element: ET.Element) -> str:
    """
    Recursively extract text from XML element and all children.

    Args:
        element: XML element

    Returns:
        Extracted text string
    """
    text = (element.text or "").strip()
    for child in element:
        child_text = extract_text(child)
        if child_text:
            text += (" " if text else "") + child_text
        if child.tail and child.tail.strip():
            text += (" " if text else "") + child.tail.strip()
    return text.strip()


def ends_with_ending_punctuation(text: str) -> bool:
    """Check if text ends with ending punctuation."""
    if not text:
        return False
    return text[-1] in ".!?;:"


def parse_statpearls_nxml(nxml_path: Path) -> List[Dict]:
    """
    Parse a StatPearls NXML file and chunk into snippets.

    Chunking logic from MedRAG:
    - Target ~1000 characters per chunk
    - Merge small paragraphs (<200 chars) with previous chunk
    - Split lists into individual items if needed

    Args:
        nxml_path: Path to NXML file

    Returns:
        List of chunk dictionaries
    """
    try:
        tree = ET.parse(nxml_path)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse {nxml_path.name}: {e}")
        return []

    root = tree.getroot()

    # Extract document title
    title_elem = root.find(".//title-group/article-title")
    if title_elem is None:
        title_elem = root.find(".//title")
    doc_title = extract_text(title_elem) if title_elem is not None else nxml_path.stem

    # Extract all sections
    sections = root.findall(".//sec")

    chunks = []
    file_id = nxml_path.stem

    for sec in sections:
        # Get section title
        sec_title_elem = sec.find("title")
        sec_title = extract_text(sec_title_elem) if sec_title_elem is not None else ""

        # Get subsection title if exists
        parent_sec = sec.find("..")
        subsec_title = ""
        if parent_sec is not None and parent_sec.tag == "sec":
            subsec_title_elem = parent_sec.find("title")
            if subsec_title_elem is not None:
                subsec_title = extract_text(subsec_title_elem)

        # Build hierarchical title
        title_parts = [doc_title]
        if subsec_title:
            title_parts.append(subsec_title)
        if sec_title and sec_title != subsec_title:
            title_parts.append(sec_title)
        hierarchical_title = " -- ".join(title_parts)

        # Extract paragraphs and lists
        current_chunk = ""

        for elem in sec:
            if elem.tag in ["p", "list"]:
                elem_text = extract_text(elem)

                if not elem_text:
                    continue

                # Chunking logic
                if len(elem_text) < 200 and len(current_chunk) + len(elem_text) < 1000:
                    # Merge small paragraphs
                    if current_chunk:
                        current_chunk += " " + elem_text
                    else:
                        current_chunk = elem_text
                else:
                    # Save current chunk if exists
                    if current_chunk:
                        chunks.append({
                            "id": f"{file_id}_{len(chunks)}",
                            "title": hierarchical_title,
                            "content": current_chunk
                        })

                    # Start new chunk
                    if len(elem_text) < 1000:
                        current_chunk = elem_text
                    else:
                        # Split long text
                        current_chunk = elem_text[:1000]
                        chunks.append({
                            "id": f"{file_id}_{len(chunks)}",
                            "title": hierarchical_title,
                            "content": current_chunk
                        })
                        current_chunk = elem_text[1000:]

        # Save final chunk
        if current_chunk:
            chunks.append({
                "id": f"{file_id}_{len(chunks)}",
                "title": hierarchical_title,
                "content": current_chunk
            })

    return chunks


def format_chunk_to_our_schema(chunk: Dict, nxml_path: Path) -> Dict:
    """
    Format StatPearls chunk to our JSONL schema.

    Output format:
    {
        "doc_id": "uuid",
        "title": "...",
        "contents": "title. content",
        "metadata": {
            "source_corpus": "statpearls",
            "original_id": "NBK_ID_chunk_num",
            "type": "medical_corpus"
        }
    }

    Args:
        chunk: Chunk dict from parse_statpearls_nxml
        nxml_path: Path to original NXML file

    Returns:
        Formatted chunk dictionary
    """
    title = chunk["title"]
    content = chunk["content"]

    # Combine title and content (MedRAG style)
    if title and content:
        if ends_with_ending_punctuation(title):
            contents = f"{title} {content}"
        else:
            contents = f"{title}. {content}"
    elif title:
        contents = title
    else:
        contents = content

    return {
        "doc_id": uuid.uuid4().hex,
        "title": title,
        "contents": contents,
        "metadata": {
            "source_corpus": "statpearls",
            "original_id": chunk["id"],
            "original_file": nxml_path.name,
            "type": "medical_corpus"
        }
    }


def preprocess_statpearls(
    input_dir: str = "data/original/medcorp/raw/statpearls",
    output_file: str = "data/original/medcorp/raw/statpearls_processed.jsonl"
) -> None:
    """
    Preprocess all StatPearls NXML files to our JSONL format.

    Args:
        input_dir: Directory containing extracted StatPearls NXML files
        output_file: Output JSONL file path
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Preprocessing StatPearls corpus")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Find all NXML files
    nxml_files = list(input_path.glob("**/*.nxml"))
    logger.info(f"Found {len(nxml_files)} NXML files")

    if not nxml_files:
        logger.error(f"No NXML files found in {input_path}")
        logger.error("Please run download_statpearls.py first")
        return

    # Process all files
    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for nxml_path in tqdm(nxml_files, desc="Processing NXML files"):
            chunks = parse_statpearls_nxml(nxml_path)

            for chunk in chunks:
                formatted = format_chunk_to_our_schema(chunk, nxml_path)
                fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("=" * 80)
    logger.info(f"✓ StatPearls preprocessing complete")
    logger.info(f"  Processed {len(nxml_files)} NXML files")
    logger.info(f"  Generated {total_chunks} chunks")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 80)
    logger.info("Next step: Update medcorp.py to load StatPearls")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess StatPearls NXML files to JSONL format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/original/medcorp/raw/statpearls",
        help="Input directory containing NXML files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/original/medcorp/raw/statpearls_processed.jsonl",
        help="Output JSONL file path"
    )
    args = parser.parse_args()

    preprocess_statpearls(args.input, args.output)


if __name__ == "__main__":
    main()
