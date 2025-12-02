"""
File-based warehouse storage for STIndex.

Provides simple, portable storage using JSON, Parquet, and GeoJSON formats.
No database server required - all data stored in files that can be queried
using DuckDB or pandas.

Storage structure:
    data/warehouse/
    ├── documents/              # Document-level data
    │   └── {doc_id}.json       # Per-document extraction results
    ├── chunks.parquet          # All chunks (main fact table)
    ├── events.geojson          # Spatial events for map visualization
    ├── indexes/                # Inverted indexes for fast filtering
    │   ├── temporal_index.json
    │   └── spatial_index.json
    └── metadata.json           # Schema version, stats, config
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from stindex.llm.response.dimension_models import MultiDimensionalResult
from stindex.warehouse.chunk_labeler import ChunkDimensionalLabels, DimensionalChunkLabeler


class FileBasedWarehouse:
    """
    Simple file-based warehouse for extraction results.

    Stores data in JSON/Parquet/GeoJSON formats that can be:
    - Queried with DuckDB (SQL interface)
    - Loaded into pandas/geopandas
    - Exported to other formats
    - Version controlled with git

    Example:
        warehouse = FileBasedWarehouse("data/warehouse")

        # Save extraction results
        warehouse.save_document(
            document_id="doc_001",
            extraction_results=results,
            document_metadata={"title": "Report", "url": "..."},
        )

        # Query with DuckDB
        df = warehouse.query("SELECT * FROM chunks WHERE temporal_year = 2022")

        # Export to GeoJSON
        warehouse.export_geojson("data/export/events.geojson")
    """

    def __init__(
        self,
        base_dir: str = "data/warehouse",
        chunk_labeler: Optional[DimensionalChunkLabeler] = None,
    ):
        """
        Initialize file-based warehouse.

        Args:
            base_dir: Base directory for warehouse files
            chunk_labeler: Optional chunk labeler for generating hierarchical labels
        """
        self.base_dir = Path(base_dir)
        self.chunk_labeler = chunk_labeler or DimensionalChunkLabeler()

        # Create directory structure
        self._init_directories()

        # Load or create metadata
        self.metadata = self._load_metadata()

        logger.info(f"FileBasedWarehouse initialized at {self.base_dir}")

    def _init_directories(self) -> None:
        """Create warehouse directory structure."""
        (self.base_dir / "documents").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "indexes").mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create warehouse metadata."""
        metadata_path = self.base_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Create default metadata
        metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": 0,
            "chunk_count": 0,
            "event_count": 0,
            "schema": {
                "chunks": self._get_chunk_schema(),
                "events": self._get_event_schema(),
            },
        }

        self._save_metadata(metadata)
        return metadata

    def _save_metadata(self, metadata: Optional[Dict] = None) -> None:
        """Save warehouse metadata."""
        if metadata is None:
            metadata = self.metadata

        metadata["updated_at"] = datetime.now().isoformat()
        metadata_path = self.base_dir / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _get_chunk_schema(self) -> Dict[str, str]:
        """Get schema for chunks table."""
        return {
            "chunk_id": "string",
            "document_id": "string",
            "chunk_index": "integer",
            "chunk_text": "string",
            "chunk_hash": "string",
            # Temporal dimensions
            "temporal_text": "string",
            "temporal_normalized": "string",
            "temporal_type": "string",
            "temporal_year": "integer",
            "temporal_quarter": "integer",
            "temporal_month": "integer",
            "temporal_labels": "array<string>",
            "temporal_path": "string",
            # Spatial dimensions
            "spatial_text": "string",
            "latitude": "float",
            "longitude": "float",
            "location_type": "string",
            "parent_region": "string",
            "spatial_labels": "array<string>",
            "spatial_path": "string",
            # Other dimensions (dynamic)
            "dimensions": "object",
            # Metadata
            "confidence_score": "float",
            "dimension_confidences": "object",
            "entity_counts": "object",
            "extraction_timestamp": "timestamp",
            "llm_provider": "string",
            "llm_model": "string",
        }

    def _get_event_schema(self) -> Dict[str, str]:
        """Get schema for events (GeoJSON features)."""
        return {
            "id": "integer",
            "chunk_id": "string",
            "document_id": "string",
            "timestamp": "string",
            "location": "string",
            "latitude": "float",
            "longitude": "float",
            "dimensions": "object",
        }

    # =========================================================================
    # SAVE OPERATIONS
    # =========================================================================

    def save_document(
        self,
        document_id: str,
        extraction_results: List[MultiDimensionalResult],
        document_metadata: Dict[str, Any],
        document_text: Optional[str] = None,
    ) -> int:
        """
        Save extraction results for a document.

        Args:
            document_id: Unique document identifier
            extraction_results: List of extraction results (one per chunk)
            document_metadata: Document metadata (title, url, etc.)
            document_text: Optional full document text

        Returns:
            Number of chunks saved
        """
        logger.info(f"Saving document {document_id} with {len(extraction_results)} chunks")

        # Generate document hash if not provided
        if not document_id:
            text_for_hash = document_text or str(document_metadata)
            document_id = hashlib.sha256(text_for_hash.encode()).hexdigest()[:16]

        # Process each chunk and collect data
        chunks_data = []
        events_data = []

        for chunk_idx, result in enumerate(extraction_results):
            # Generate labels
            labels = self.chunk_labeler.label_chunk(
                chunk_text=result.input_text,
                extraction_result=result,
                chunk_index=chunk_idx,
            )

            # Create chunk record
            chunk_record = self._create_chunk_record(
                document_id=document_id,
                chunk_index=chunk_idx,
                extraction_result=result,
                labels=labels,
            )
            chunks_data.append(chunk_record)

            # Create event record (for GeoJSON)
            event_record = self._create_event_record(
                event_id=len(events_data),
                document_id=document_id,
                chunk_id=chunk_record["chunk_id"],
                extraction_result=result,
            )
            if event_record:
                events_data.append(event_record)

        # Save document-level JSON
        doc_path = self.base_dir / "documents" / f"{document_id}.json"
        doc_data = {
            "document_id": document_id,
            "metadata": document_metadata,
            "chunk_count": len(chunks_data),
            "chunks": chunks_data,
            "created_at": datetime.now().isoformat(),
        }
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False, default=str)

        # Append to main chunks file
        self._append_chunks(chunks_data)

        # Append to events file
        if events_data:
            self._append_events(events_data)

        # Update indexes
        self._update_indexes(chunks_data)

        # Update metadata
        self.metadata["document_count"] += 1
        self.metadata["chunk_count"] += len(chunks_data)
        self.metadata["event_count"] += len(events_data)
        self._save_metadata()

        logger.success(f"Saved {len(chunks_data)} chunks, {len(events_data)} events for {document_id}")

        return len(chunks_data)

    def _create_chunk_record(
        self,
        document_id: str,
        chunk_index: int,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> Dict[str, Any]:
        """Create a chunk record for storage."""
        chunk_id = f"{document_id}_{chunk_index:04d}"

        # Extract temporal info
        temporal_entity = (
            extraction_result.temporal_entities[0]
            if extraction_result.temporal_entities
            else {}
        )
        temporal_normalized = temporal_entity.get("normalized", "")

        # Parse temporal components
        temporal_year = None
        temporal_quarter = None
        temporal_month = None

        if temporal_normalized and len(temporal_normalized) >= 4:
            try:
                temporal_year = int(temporal_normalized[:4])
                if len(temporal_normalized) >= 7:
                    month = int(temporal_normalized[5:7])
                    temporal_month = month
                    temporal_quarter = (month - 1) // 3 + 1
            except (ValueError, IndexError):
                pass

        # Extract spatial info
        spatial_entity = (
            extraction_result.spatial_entities[0]
            if extraction_result.spatial_entities
            else {}
        )

        # Build chunk record
        record = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_text": extraction_result.input_text,
            "chunk_hash": labels.chunk_hash,
            # Temporal
            "temporal_text": temporal_entity.get("text"),
            "temporal_normalized": temporal_normalized,
            "temporal_type": temporal_entity.get("normalization_type"),
            "temporal_year": temporal_year,
            "temporal_quarter": temporal_quarter,
            "temporal_month": temporal_month,
            "temporal_labels": labels.temporal_labels,
            "temporal_path": labels.temporal_path,
            # Spatial
            "spatial_text": spatial_entity.get("text"),
            "latitude": spatial_entity.get("latitude"),
            "longitude": spatial_entity.get("longitude"),
            "location_type": spatial_entity.get("location_type"),
            "parent_region": spatial_entity.get("parent_region"),
            "spatial_labels": labels.spatial_labels,
            "spatial_path": labels.spatial_path,
            # Other dimensions
            "dimensions": {
                k: v for k, v in extraction_result.entities.items()
                if k not in ("temporal", "spatial")
            },
            # Metadata
            "confidence_score": labels.confidence_score,
            "dimension_confidences": labels.dimension_confidences,
            "entity_counts": labels.entity_counts,
            "extraction_timestamp": datetime.now().isoformat(),
            "llm_provider": (
                extraction_result.extraction_config.get("llm_provider")
                if extraction_result.extraction_config else None
            ),
            "llm_model": (
                extraction_result.extraction_config.get("model_name")
                if extraction_result.extraction_config else None
            ),
        }

        return record

    def _create_event_record(
        self,
        event_id: int,
        document_id: str,
        chunk_id: str,
        extraction_result: MultiDimensionalResult,
    ) -> Optional[Dict[str, Any]]:
        """Create an event record for GeoJSON."""
        temporal_entity = (
            extraction_result.temporal_entities[0]
            if extraction_result.temporal_entities
            else {}
        )
        spatial_entity = (
            extraction_result.spatial_entities[0]
            if extraction_result.spatial_entities
            else {}
        )

        # Need at least temporal or spatial
        if not temporal_entity and not spatial_entity:
            return None

        lat = spatial_entity.get("latitude")
        lon = spatial_entity.get("longitude")

        # For GeoJSON, we need coordinates
        if lat is None or lon is None:
            return None

        return {
            "id": event_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "timestamp": temporal_entity.get("normalized"),
            "temporal_text": temporal_entity.get("text"),
            "location": spatial_entity.get("text"),
            "latitude": lat,
            "longitude": lon,
            "location_type": spatial_entity.get("location_type"),
            "dimensions": {
                k: v[0] if v else None
                for k, v in extraction_result.entities.items()
                if k not in ("temporal", "spatial") and v
            },
            "text_preview": extraction_result.input_text[:200] if extraction_result.input_text else None,
        }

    def _append_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Append chunks to the main chunks file (JSON Lines format)."""
        chunks_path = self.base_dir / "chunks.jsonl"

        with open(chunks_path, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False, default=str) + "\n")

    def _append_events(self, events: List[Dict[str, Any]]) -> None:
        """Append events to the GeoJSON file."""
        geojson_path = self.base_dir / "events.geojson"

        # Load existing GeoJSON or create new
        if geojson_path.exists():
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson = json.load(f)
        else:
            geojson = {"type": "FeatureCollection", "features": []}

        # Convert events to GeoJSON features
        for event in events:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [event["longitude"], event["latitude"]],
                },
                "properties": {
                    k: v for k, v in event.items()
                    if k not in ("latitude", "longitude")
                },
            }
            geojson["features"].append(feature)

        # Save updated GeoJSON
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False, default=str)

    def _update_indexes(self, chunks: List[Dict[str, Any]]) -> None:
        """Update inverted indexes for fast filtering."""
        # Temporal index: year/month/quarter -> chunk_ids
        temporal_index_path = self.base_dir / "indexes" / "temporal_index.json"
        temporal_index = self._load_index(temporal_index_path)

        # Spatial index: region/country -> chunk_ids
        spatial_index_path = self.base_dir / "indexes" / "spatial_index.json"
        spatial_index = self._load_index(spatial_index_path)

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]

            # Update temporal index
            if chunk.get("temporal_year"):
                year_key = str(chunk["temporal_year"])
                if year_key not in temporal_index:
                    temporal_index[year_key] = []
                temporal_index[year_key].append(chunk_id)

            if chunk.get("temporal_labels"):
                for label in chunk["temporal_labels"]:
                    if label not in temporal_index:
                        temporal_index[label] = []
                    if chunk_id not in temporal_index[label]:
                        temporal_index[label].append(chunk_id)

            # Update spatial index
            if chunk.get("spatial_labels"):
                for label in chunk["spatial_labels"]:
                    if label not in spatial_index:
                        spatial_index[label] = []
                    if chunk_id not in spatial_index[label]:
                        spatial_index[label].append(chunk_id)

        # Save indexes
        self._save_index(temporal_index_path, temporal_index)
        self._save_index(spatial_index_path, spatial_index)

    def _load_index(self, path: Path) -> Dict[str, List[str]]:
        """Load an inverted index."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self, path: Path, index: Dict[str, List[str]]) -> None:
        """Save an inverted index."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # LOAD OPERATIONS
    # =========================================================================

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks from the warehouse."""
        chunks_path = self.base_dir / "chunks.jsonl"

        if not chunks_path.exists():
            return []

        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        return chunks

    def load_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific document."""
        doc_path = self.base_dir / "documents" / f"{document_id}.json"

        if not doc_path.exists():
            return None

        with open(doc_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_events_geojson(self) -> Dict[str, Any]:
        """Load events as GeoJSON."""
        geojson_path = self.base_dir / "events.geojson"

        if not geojson_path.exists():
            return {"type": "FeatureCollection", "features": []}

        with open(geojson_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_chunks_by_temporal(
        self,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
        date_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get chunks filtered by temporal dimensions.

        Args:
            year: Filter by year (e.g., 2022)
            quarter: Filter by quarter (1-4)
            month: Filter by month (1-12)
            date_range: Tuple of (start_date, end_date) in ISO format

        Returns:
            List of matching chunks
        """
        # Try to use index first
        if year and not (quarter or month or date_range):
            index = self._load_index(self.base_dir / "indexes" / "temporal_index.json")
            chunk_ids = index.get(str(year), [])
            if chunk_ids:
                return self._load_chunks_by_ids(chunk_ids)

        # Fall back to full scan
        chunks = self.load_all_chunks()
        filtered = []

        for chunk in chunks:
            if year and chunk.get("temporal_year") != year:
                continue
            if quarter and chunk.get("temporal_quarter") != quarter:
                continue
            if month and chunk.get("temporal_month") != month:
                continue
            if date_range:
                normalized = chunk.get("temporal_normalized", "")
                if normalized < date_range[0] or normalized > date_range[1]:
                    continue
            filtered.append(chunk)

        return filtered

    def get_chunks_by_spatial(
        self,
        region: Optional[str] = None,
        bbox: Optional[tuple] = None,
        radius_km: Optional[float] = None,
        center: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get chunks filtered by spatial dimensions.

        Args:
            region: Filter by region name (uses spatial_labels)
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            radius_km: Radius in kilometers (requires center)
            center: Center point (longitude, latitude) for radius search

        Returns:
            List of matching chunks
        """
        # Try to use index first for region filter
        if region and not (bbox or radius_km):
            index = self._load_index(self.base_dir / "indexes" / "spatial_index.json")
            chunk_ids = index.get(region, [])
            if chunk_ids:
                return self._load_chunks_by_ids(chunk_ids)

        # Fall back to full scan
        chunks = self.load_all_chunks()
        filtered = []

        for chunk in chunks:
            lat = chunk.get("latitude")
            lon = chunk.get("longitude")

            if region:
                labels = chunk.get("spatial_labels", [])
                if region not in labels:
                    continue

            if bbox and lat is not None and lon is not None:
                min_lon, min_lat, max_lon, max_lat = bbox
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue

            if radius_km and center and lat is not None and lon is not None:
                dist = self._haversine_distance(center[1], center[0], lat, lon)
                if dist > radius_km:
                    continue

            filtered.append(chunk)

        return filtered

    def _load_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Load specific chunks by their IDs."""
        chunk_ids_set = set(chunk_ids)
        chunks = []

        for chunk in self.load_all_chunks():
            if chunk.get("chunk_id") in chunk_ids_set:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in kilometers."""
        from math import radians, sin, cos, sqrt, atan2

        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    # =========================================================================
    # EXPORT OPERATIONS
    # =========================================================================

    def export_to_parquet(self, output_path: Optional[str] = None) -> str:
        """
        Export chunks to Parquet format for efficient querying.

        Args:
            output_path: Output file path (default: base_dir/chunks.parquet)

        Returns:
            Path to exported file
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet export: pip install pandas pyarrow")

        output_path = output_path or str(self.base_dir / "chunks.parquet")
        chunks = self.load_all_chunks()

        if not chunks:
            logger.warning("No chunks to export")
            return output_path

        df = pd.DataFrame(chunks)
        df.to_parquet(output_path, index=False)

        logger.info(f"Exported {len(chunks)} chunks to {output_path}")
        return output_path

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """Export chunks to CSV format."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV export: pip install pandas")

        output_path = output_path or str(self.base_dir / "chunks.csv")
        chunks = self.load_all_chunks()

        if not chunks:
            logger.warning("No chunks to export")
            return output_path

        df = pd.DataFrame(chunks)

        # Convert list/dict columns to JSON strings for CSV
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if isinstance(x, (list, dict)) else x)

        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(chunks)} chunks to {output_path}")
        return output_path

    def export_geojson(self, output_path: Optional[str] = None) -> str:
        """Export events to GeoJSON format."""
        output_path = output_path or str(self.base_dir / "events.geojson")
        geojson = self.load_events_geojson()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Exported {len(geojson['features'])} events to {output_path}")
        return output_path

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get warehouse statistics."""
        chunks = self.load_all_chunks()

        # Temporal distribution
        years = {}
        for chunk in chunks:
            year = chunk.get("temporal_year")
            if year:
                years[year] = years.get(year, 0) + 1

        # Spatial distribution
        regions = {}
        for chunk in chunks:
            for label in chunk.get("spatial_labels", []):
                regions[label] = regions.get(label, 0) + 1

        return {
            "total_documents": self.metadata.get("document_count", 0),
            "total_chunks": len(chunks),
            "total_events": self.metadata.get("event_count", 0),
            "temporal_distribution": dict(sorted(years.items())),
            "spatial_distribution": dict(sorted(regions.items(), key=lambda x: -x[1])[:20]),
            "storage_path": str(self.base_dir),
        }

    def clear(self) -> None:
        """Clear all warehouse data (use with caution!)."""
        import shutil

        logger.warning(f"Clearing warehouse at {self.base_dir}")

        # Remove files
        for file in ["chunks.jsonl", "chunks.parquet", "chunks.csv", "events.geojson"]:
            path = self.base_dir / file
            if path.exists():
                path.unlink()

        # Clear directories
        for dir_name in ["documents", "indexes"]:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)

        # Reset metadata
        self.metadata = {
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "document_count": 0,
            "chunk_count": 0,
            "event_count": 0,
            "schema": {
                "chunks": self._get_chunk_schema(),
                "events": self._get_event_schema(),
            },
        }
        self._save_metadata()

        logger.info("Warehouse cleared")
