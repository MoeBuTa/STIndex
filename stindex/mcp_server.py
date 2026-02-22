"""
STIndex MCP server — exposes the STIndex extraction pipeline over SSE/HTTP.

Transport: SSE (default, suitable for web-hosted MCP clients)
           stdio / streamable-http also supported via --transport flag.

Usage:
    stindex-mcp --port 8008           # SSE on 0.0.0.0:8008
    python -m stindex.mcp_server --transport stdio
"""

import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_response(
    pipeline_results: List[Dict[str, Any]],
    input_type: str,
    document_id: str,
    title: str,
) -> Dict[str, Any]:
    """Convert a list of pipeline result dicts into the ExtractionResponse schema.

    ExtractionResponse schema:
    {
        "success": true,
        "document_id": "...",
        "title": "...",
        "input_type": "text|file|url|content",
        "total_chunks": N,
        "chunks": [
            {
                "chunk_index": 0,
                "total_chunks": N,
                "text_preview": "First 200 chars…",
                "entities": { "temporal": [...], "spatial": [...], ... },
                "processing_time": 1.23
            }
        ],
        "summary": {
            "dimension_counts": { "temporal": 1, "spatial": 1 },
            "total_processing_time": 1.23
        }
    }
    """
    chunks: List[Dict[str, Any]] = []
    total_time = 0.0
    dimension_counts: Dict[str, int] = {}

    for r in pipeline_results:
        extraction = r.get("extraction", {})
        entities: Dict[str, Any] = extraction.get("entities") or {}
        processing_time = float(extraction.get("processing_time") or 0.0)
        total_time += processing_time

        for dim, ents in entities.items():
            dimension_counts[dim] = dimension_counts.get(dim, 0) + len(ents or [])

        chunk_params = r.get("chunk_params") or {}
        chunks.append({
            "chunk_index": r.get("chunk_index", 0),
            "total_chunks": chunk_params.get("total_chunks", len(pipeline_results)),
            "text_preview": (r.get("text") or "")[:200],
            "entities": entities,
            "processing_time": round(processing_time, 3),
        })

    return {
        "success": True,
        "document_id": document_id,
        "title": title,
        "input_type": input_type,
        "total_chunks": len(chunks),
        "chunks": chunks,
        "summary": {
            "dimension_counts": dimension_counts,
            "total_processing_time": round(total_time, 3),
        },
    }


def _run_pipeline_for_doc(
    doc: Any,
    input_type: str,
    config: str,
    context_aware: bool,
    document_id: Optional[str],
    title: Optional[str],
) -> Dict[str, Any]:
    """Run STIndexPipeline for one InputDocument and return an ExtractionResponse."""
    from stindex.pipeline import STIndexPipeline

    pipeline = STIndexPipeline(
        extractor_config=config,
        enable_context_aware=context_aware,
        save_intermediate=False,
    )
    pipeline_result = pipeline.run_pipeline([doc], save_results=False, analyze=False)
    results: List[Dict[str, Any]] = pipeline_result.get("results") or []

    doc_id = document_id or doc.document_id
    doc_title = title or doc.title or doc_id

    return _build_response(results, input_type=input_type, document_id=doc_id, title=doc_title)


def _response_to_pipeline_format(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct a pipeline result list from an ExtractionResponse (for analysis)."""
    pipeline_results: List[Dict[str, Any]] = []
    doc_id = response.get("document_id", "doc")
    title = response.get("title", "")

    for chunk in response.get("chunks") or []:
        pipeline_results.append({
            "doc_id": f"{doc_id}_chunk_{chunk.get('chunk_index', 0)}",
            "chunk_index": chunk.get("chunk_index", 0),
            "document_id": doc_id,
            "document_title": title,
            "text": chunk.get("text_preview", ""),
            "chunk_params": {"total_chunks": chunk.get("total_chunks", 1)},
            "extraction": {
                "success": True,
                "entities": chunk.get("entities") or {},
                "processing_time": chunk.get("processing_time", 0.0),
                "extraction_config": {},
            },
        })

    return pipeline_results


def _serialize(obj: Any) -> Any:
    """Make an object JSON-serializable via a round-trip through json.dumps."""
    return json.loads(json.dumps(obj, default=str))


# ---------------------------------------------------------------------------
# MCP server builder
# ---------------------------------------------------------------------------

def _build_mcp(host: str = "0.0.0.0", port: int = 8008) -> FastMCP:
    """Create and configure a FastMCP server with all STIndex tools registered."""
    mcp = FastMCP("STIndex", host=host, port=port)

    # ------------------------------------------------------------------
    # Tool 1: extract_text
    # ------------------------------------------------------------------
    @mcp.tool()
    def extract_text(
        text: str,
        config: str = "extract",
        context_aware: bool = True,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        publication_date: Optional[str] = None,
        source_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract temporal, spatial, and custom entities from plain text or HTML.

        Args:
            text: Raw text (or raw HTML string) to process.
            config: Config name — extract / openai / anthropic / hf.
            context_aware: Maintain context across document chunks.
            document_id: Optional document identifier override.
            title: Optional document title.
            publication_date: ISO 8601 date hint for relative temporal resolution.
            source_location: Geographic context hint for spatial disambiguation.

        Returns:
            ExtractionResponse JSON dict.
        """
        try:
            from stindex.extraction.dimensional_extraction import DimensionalExtractor

            metadata: Dict[str, Any] = {}
            if publication_date:
                metadata["publication_date"] = publication_date
            if source_location:
                metadata["source_location"] = source_location

            extractor = DimensionalExtractor(config_path=config)
            result = extractor.extract(text, document_metadata=metadata)

            doc_id = document_id or f"text_{abs(hash(text[:100])) % 10000:04d}"
            doc_title = title or (text[:50].strip() + ("..." if len(text) > 50 else ""))

            # Wrap single result in the pipeline list format _build_response expects
            pipeline_results: List[Dict[str, Any]] = [{
                "chunk_index": 0,
                "chunk_params": {"total_chunks": 1},
                "text": text,
                "extraction": result.model_dump(),
            }]

            return _build_response(
                pipeline_results,
                input_type="text",
                document_id=doc_id,
                title=doc_title,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}

    # ------------------------------------------------------------------
    # Tool 2: extract_file
    # ------------------------------------------------------------------
    @mcp.tool()
    def extract_file(
        file_path: str,
        config: str = "extract",
        context_aware: bool = True,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract entities from a local file (PDF, DOCX, TXT, HTML).

        Uses the unstructured library for parsing via STIndexPipeline.

        Args:
            file_path: Absolute path to the local file.
            config: Config name — extract / openai / anthropic / hf.
            context_aware: Maintain context across document chunks.
            document_id: Optional document identifier override.

        Returns:
            ExtractionResponse JSON dict.
        """
        try:
            from stindex.preprocess import InputDocument

            doc = InputDocument.from_file(file_path, document_id=document_id)
            return _run_pipeline_for_doc(
                doc,
                input_type="file",
                config=config,
                context_aware=context_aware,
                document_id=document_id,
                title=None,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}

    # ------------------------------------------------------------------
    # Tool 3: extract_url
    # ------------------------------------------------------------------
    @mcp.tool()
    def extract_url(
        url: str,
        config: str = "extract",
        context_aware: bool = True,
    ) -> Dict[str, Any]:
        """Scrape a web URL and extract entities from the page content.

        Args:
            url: Web URL to scrape and process.
            config: Config name — extract / openai / anthropic / hf.
            context_aware: Maintain context across document chunks.

        Returns:
            ExtractionResponse JSON dict.
        """
        try:
            from stindex.preprocess import InputDocument

            doc = InputDocument.from_url(url)
            return _run_pipeline_for_doc(
                doc,
                input_type="url",
                config=config,
                context_aware=context_aware,
                document_id=None,
                title=None,
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}

    # ------------------------------------------------------------------
    # Tool 4: extract_content
    # ------------------------------------------------------------------
    @mcp.tool()
    def extract_content(
        content_base64: str,
        filename: str,
        config: str = "extract",
        context_aware: bool = True,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract entities from base64-encoded file bytes.

        Useful for remote MCP clients that cannot share a local file path.
        Writes the bytes to a temporary file, runs extract_file, then cleans up.

        Args:
            content_base64: Base64-encoded bytes of the file.
            filename: Original filename (e.g. "report.pdf") — used to detect format.
            config: Config name — extract / openai / anthropic / hf.
            context_aware: Maintain context across document chunks.
            document_id: Optional document identifier override.

        Returns:
            ExtractionResponse JSON dict.
        """
        try:
            from stindex.preprocess import InputDocument

            content_bytes = base64.b64decode(content_base64)
            suffix = Path(filename).suffix or ".tmp"

            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                tmp.write(content_bytes)
                tmp.close()

                doc = InputDocument.from_file(
                    tmp.name,
                    document_id=document_id,
                    title=Path(filename).stem,
                )
                return _run_pipeline_for_doc(
                    doc,
                    input_type="content",
                    config=config,
                    context_aware=context_aware,
                    document_id=document_id,
                    title=Path(filename).stem,
                )
            finally:
                Path(tmp.name).unlink(missing_ok=True)

        except Exception as exc:
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}

    # ------------------------------------------------------------------
    # Tool 5: analyze
    # ------------------------------------------------------------------
    @mcp.tool()
    def analyze(
        results: Dict[str, Any],
        dimensions: Optional[List[str]] = None,
        clustering_mode: str = "spatiotemporal",
        export_geojson: bool = False,
    ) -> Dict[str, Any]:
        """Analyse extraction results: spatiotemporal clustering and dimension statistics.

        Args:
            results: Output dict from any extract_* tool (ExtractionResponse).
            dimensions: Subset of dimensions to analyse (None = all discovered).
            clustering_mode: 'temporal' | 'spatial' | 'spatiotemporal' | 'categorical' | 'multi'.
            export_geojson: Include GeoJSON content in response for map visualisation.

        Returns:
            Dict with 'clusters', 'dimension_analysis', 'exported_files',
            and optionally 'geojson_content' when export_geojson=True.
        """
        try:
            from stindex.pipeline import STIndexPipeline

            # Accept both ExtractionResponse dicts and raw pipeline result lists
            if isinstance(results, list):
                raw_results: List[Dict[str, Any]] = results
            elif isinstance(results, dict) and "chunks" in results:
                raw_results = _response_to_pipeline_format(results)
            else:
                raw_results = [results]

            pipeline = STIndexPipeline(save_intermediate=False)
            analysis = pipeline.run_analysis(
                results=raw_results,
                dimensions=dimensions,
                clustering_mode=clustering_mode,
                export_geojson=export_geojson,
            )

            serialized = _serialize(analysis)

            # If GeoJSON was requested, embed its content directly in the response
            if export_geojson:
                geojson_path = (analysis.get("exported_files") or {}).get("geojson")
                if geojson_path and Path(str(geojson_path)).exists():
                    with open(str(geojson_path)) as fh:
                        serialized["geojson_content"] = json.load(fh)

            return serialized

        except Exception as exc:
            return {"success": False, "error": str(exc), "error_type": type(exc).__name__}

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the STIndex MCP server."""
    p = argparse.ArgumentParser(description="STIndex MCP Server")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8008, help="Port to listen on (default: 8008)")
    p.add_argument(
        "--transport",
        default="sse",
        choices=["sse", "stdio", "streamable-http"],
        help="Transport protocol (default: sse)",
    )
    args = p.parse_args()

    mcp = _build_mcp(host=args.host, port=args.port)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
