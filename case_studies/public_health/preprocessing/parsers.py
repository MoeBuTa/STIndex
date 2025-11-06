"""
Document parsers using the unstructured package.

Parses HTML, PDFs, and other document formats into structured text
suitable for STIndex extraction.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

# Ensure NLTK data is available (required by unstructured)
def _ensure_nltk_data():
    """Download NLTK data if not already available."""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("✓ NLTK data downloaded")
    except Exception as e:
        logger.warning(f"Failed to setup NLTK data: {e}")

_ensure_nltk_data()

try:
    from unstructured.partition.html import partition_html
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Element, Title, NarrativeText, Table
    from unstructured.cleaners.core import clean_extra_whitespace
except ImportError:
    logger.error(
        "unstructured package not found. Install with: pip install unstructured[local-inference]"
    )
    raise


@dataclass
class ParsedDocument:
    """Container for parsed document data."""

    source_file: str
    document_type: str  # "health_alert", "news_article", etc.
    title: str
    content: str  # Full text content
    sections: List[Dict[str, Any]]  # Structured sections
    tables: List[Dict[str, Any]]  # Extracted tables
    metadata: Dict[str, Any]
    parsing_method: str  # "unstructured", "beautifulsoup", etc.


class HealthAlertParser:
    """Parser for health alert documents using unstructured."""

    def __init__(self, output_dir: str = "case_studies/public_health/data/processed"):
        """
        Initialize parser.

        Args:
            output_dir: Directory to save parsed documents
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_from_html_string(
        self,
        html_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Parse HTML content using unstructured.

        Args:
            html_content: Raw HTML string
            metadata: Optional metadata dict

        Returns:
            ParsedDocument object
        """
        metadata = metadata or {}

        # Use unstructured to partition HTML
        elements = partition_html(text=html_content)

        # Extract different element types
        title = ""
        sections = []
        tables = []
        full_text_parts = []

        for element in elements:
            # Clean whitespace
            element_text = clean_extra_whitespace(element.text)

            if isinstance(element, Title):
                if not title:  # Use first title as document title
                    title = element_text
                sections.append({
                    "type": "title",
                    "text": element_text,
                    "level": getattr(element, 'level', 1)
                })

            elif isinstance(element, Table):
                # Extract table structure
                table_data = self._extract_table_data(element)
                tables.append(table_data)
                sections.append({
                    "type": "table",
                    "data": table_data
                })

            elif isinstance(element, NarrativeText):
                sections.append({
                    "type": "text",
                    "text": element_text
                })

            else:
                # Other element types
                sections.append({
                    "type": "other",
                    "text": element_text,
                    "element_type": type(element).__name__
                })

            # Add to full text
            if element_text:
                full_text_parts.append(element_text)

        # Combine full text
        full_text = "\n\n".join(full_text_parts)

        return ParsedDocument(
            source_file=metadata.get("source_file", "unknown"),
            document_type=metadata.get("document_type", "health_alert"),
            title=title or metadata.get("title", "Untitled"),
            content=full_text,
            sections=sections,
            tables=tables,
            metadata=metadata,
            parsing_method="unstructured"
        )

    def parse_from_file(self, file_path: str) -> ParsedDocument:
        """
        Parse document from file (auto-detects format).

        Args:
            file_path: Path to document file

        Returns:
            ParsedDocument object
        """
        file_path = Path(file_path)

        logger.info(f"Parsing {file_path.name}...")

        # Use unstructured's auto-partition
        elements = partition(str(file_path))

        # Process elements similar to HTML parsing
        title = ""
        sections = []
        tables = []
        full_text_parts = []

        for element in elements:
            element_text = clean_extra_whitespace(element.text)

            if isinstance(element, Title):
                if not title:
                    title = element_text
                sections.append({"type": "title", "text": element_text})

            elif isinstance(element, Table):
                table_data = self._extract_table_data(element)
                tables.append(table_data)
                sections.append({"type": "table", "data": table_data})

            elif isinstance(element, NarrativeText):
                sections.append({"type": "text", "text": element_text})

            else:
                sections.append({
                    "type": "other",
                    "text": element_text,
                    "element_type": type(element).__name__
                })

            if element_text:
                full_text_parts.append(element_text)

        full_text = "\n\n".join(full_text_parts)

        return ParsedDocument(
            source_file=str(file_path),
            document_type="health_alert",
            title=title or file_path.stem,
            content=full_text,
            sections=sections,
            tables=tables,
            metadata={"file_path": str(file_path)},
            parsing_method="unstructured"
        )

    def _extract_table_data(self, table_element: Table) -> Dict[str, Any]:
        """
        Extract structured data from table element.

        Args:
            table_element: Table element from unstructured

        Returns:
            Dict with table data
        """
        # Basic table extraction
        # unstructured returns table text, we'd need more sophisticated parsing
        # for structured data (could use pandas or custom parsing)

        table_text = clean_extra_whitespace(table_element.text)

        return {
            "raw_text": table_text,
            "rows": self._parse_table_rows(table_text)
        }

    def _parse_table_rows(self, table_text: str) -> List[List[str]]:
        """
        Parse table text into rows.

        Args:
            table_text: Raw table text

        Returns:
            List of rows (each row is a list of cells)
        """
        # Simple row parsing (split by newlines)
        rows = []
        for line in table_text.split('\n'):
            if line.strip():
                # Basic cell splitting (would need enhancement for complex tables)
                cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                if cells:
                    rows.append(cells)
        return rows

    def parse_health_alert_json(self, json_file: str) -> List[ParsedDocument]:
        """
        Parse health alerts from scraped JSON file.

        Args:
            json_file: Path to JSON file from scrapers

        Returns:
            List of ParsedDocument objects
        """
        json_path = Path(json_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            alerts = json.load(f)

        parsed_docs = []

        for alert in alerts:
            # Parse the raw HTML content
            if alert.get("raw_html"):
                metadata = {
                    "source_file": str(json_path),
                    "title": alert.get("title"),
                    "url": alert.get("url"),
                    "source": alert.get("source"),
                    "publication_date": alert.get("publication_date"),
                    "document_type": "health_alert",
                    **alert.get("metadata", {})
                }

                parsed_doc = self.parse_from_html_string(
                    alert["raw_html"],
                    metadata=metadata
                )
                parsed_docs.append(parsed_doc)

        return parsed_docs

    def save_parsed_documents(self, documents: List[ParsedDocument], filename: str):
        """
        Save parsed documents to JSON file.

        Args:
            documents: List of ParsedDocument objects
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Convert to dicts
        docs_data = [asdict(doc) for doc in documents]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(documents)} parsed documents to {output_path}")


def parse_all_health_alerts(
    raw_dir: str = "case_studies/public_health/data/raw",
    output_dir: str = "case_studies/public_health/data/processed"
):
    """
    Parse all scraped health alert JSON files.

    Args:
        raw_dir: Directory with scraped JSON files
        output_dir: Directory to save parsed documents
    """
    logger.info("Parsing scraped health alerts...")

    parser = HealthAlertParser(output_dir)
    raw_path = Path(raw_dir)

    # Find all JSON files
    json_files = list(raw_path.glob("*.json"))

    for json_file in json_files:
        logger.info(f"Processing {json_file.name}...")

        parsed_docs = parser.parse_health_alert_json(str(json_file))

        # Save with same name but in processed dir
        output_name = f"parsed_{json_file.name}"
        parser.save_parsed_documents(parsed_docs, output_name)

    logger.info("✓ Parsing complete!")


if __name__ == "__main__":
    # Parse all scraped alerts
    parse_all_health_alerts()
