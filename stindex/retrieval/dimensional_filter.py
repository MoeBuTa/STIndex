"""
Dimensional Filter for STIndex.

Provides pre-filtering of chunks based on temporal and spatial dimensions
before or after vector similarity search.

Supports:
- Temporal filtering: year, quarter, month, date range
- Spatial filtering: region, country, bounding box, radius
- Combined filters with AND/OR logic
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger


@dataclass
class TemporalFilter:
    """Filter criteria for temporal dimension."""
    year: Optional[int] = None
    quarter: Optional[int] = None
    month: Optional[int] = None
    start_date: Optional[str] = None  # ISO format
    end_date: Optional[str] = None    # ISO format


@dataclass
class SpatialFilter:
    """Filter criteria for spatial dimension."""
    region: Optional[str] = None
    country: Optional[str] = None
    bbox: Optional[tuple] = None  # (min_lon, min_lat, max_lon, max_lat)
    center: Optional[tuple] = None  # (lon, lat) for radius search
    radius_km: Optional[float] = None


@dataclass
class DimensionalFilterResult:
    """Result of dimensional filtering."""
    chunk_ids: Set[str]
    applied_filters: Dict[str, Any]
    filter_stats: Dict[str, int] = field(default_factory=dict)


class DimensionalFilter:
    """
    Filter chunks by temporal and spatial dimensions using inverted indexes.

    Uses pre-built inverted indexes for fast filtering without full table scan.
    """

    def __init__(
        self,
        warehouse_path: str,
        temporal_index_path: Optional[str] = None,
        spatial_index_path: Optional[str] = None,
    ):
        """
        Initialize dimensional filter.

        Args:
            warehouse_path: Path to STIndex warehouse
            temporal_index_path: Path to temporal index (default: warehouse/indexes/temporal_index.json)
            spatial_index_path: Path to spatial index (default: warehouse/indexes/spatial_index.json)
        """
        self.warehouse_path = Path(warehouse_path)

        # Set index paths
        self.temporal_index_path = Path(temporal_index_path) if temporal_index_path else \
            self.warehouse_path / "indexes" / "temporal_index.json"
        self.spatial_index_path = Path(spatial_index_path) if spatial_index_path else \
            self.warehouse_path / "indexes" / "spatial_index.json"

        # Load indexes
        self.temporal_index = self._load_index(self.temporal_index_path)
        self.spatial_index = self._load_index(self.spatial_index_path)

        # Load enriched chunks for additional filtering
        self._chunks_cache = None

        logger.info(f"DimensionalFilter initialized with "
                   f"{len(self.temporal_index)} temporal keys, "
                   f"{len(self.spatial_index)} spatial keys")

    def _load_index(self, path: Path) -> Dict[str, List[str]]:
        """Load inverted index from JSON file."""
        if not path.exists():
            logger.warning(f"Index not found: {path}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load enriched chunks for complex filtering."""
        if self._chunks_cache is not None:
            return self._chunks_cache

        chunks_path = self.warehouse_path / "chunks_enriched.jsonl"
        if not chunks_path.exists():
            logger.warning(f"Enriched chunks not found: {chunks_path}")
            return []

        import jsonlines
        self._chunks_cache = []
        with jsonlines.open(chunks_path, "r") as reader:
            for chunk in reader:
                self._chunks_cache.append(chunk)

        return self._chunks_cache

    def filter_temporal(
        self,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Set[str]:
        """
        Filter chunks by temporal criteria.

        Args:
            year: Filter by year
            quarter: Filter by quarter (1-4)
            month: Filter by month (1-12)
            start_date: Start of date range (ISO format)
            end_date: End of date range (ISO format)

        Returns:
            Set of matching chunk IDs
        """
        result_ids = None

        # Year filter (use index)
        if year is not None:
            year_ids = set(self.temporal_index.get(str(year), []))
            result_ids = year_ids if result_ids is None else result_ids & year_ids

        # Quarter filter (use index)
        if quarter is not None:
            quarter_key = f"Q{quarter}"
            quarter_ids = set(self.temporal_index.get(quarter_key, []))
            result_ids = quarter_ids if result_ids is None else result_ids & quarter_ids

        # Month filter (need full scan if no specific index)
        if month is not None:
            # Try to find month in index keys
            month_key = f"{year or '*'}-{month:02d}" if year else f"*-{month:02d}"
            month_ids = set()
            for key, ids in self.temporal_index.items():
                if "-" in key:
                    try:
                        key_month = int(key.split("-")[1])
                        if key_month == month:
                            month_ids.update(ids)
                    except (ValueError, IndexError):
                        pass
            if month_ids:
                result_ids = month_ids if result_ids is None else result_ids & month_ids

        # Date range filter (need full scan)
        if start_date or end_date:
            chunks = self._load_chunks()
            range_ids = set()
            for chunk in chunks:
                normalized = chunk.get("temporal_normalized", "")
                if not normalized:
                    continue
                if start_date and normalized < start_date:
                    continue
                if end_date and normalized > end_date:
                    continue
                range_ids.add(chunk.get("chunk_id", ""))

            result_ids = range_ids if result_ids is None else result_ids & range_ids

        return result_ids or set()

    def filter_spatial(
        self,
        region: Optional[str] = None,
        country: Optional[str] = None,
        bbox: Optional[tuple] = None,
        center: Optional[tuple] = None,
        radius_km: Optional[float] = None,
    ) -> Set[str]:
        """
        Filter chunks by spatial criteria.

        Args:
            region: Filter by region/location name
            country: Filter by country
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            center: Center point for radius search (lon, lat)
            radius_km: Radius in kilometers

        Returns:
            Set of matching chunk IDs
        """
        result_ids = None

        # Region filter (use index)
        if region is not None:
            region_ids = set(self.spatial_index.get(region, []))
            result_ids = region_ids if result_ids is None else result_ids & region_ids

        # Country filter (use index)
        if country is not None:
            country_ids = set(self.spatial_index.get(country, []))
            result_ids = country_ids if result_ids is None else result_ids & country_ids

        # Bounding box filter (need full scan)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            chunks = self._load_chunks()
            bbox_ids = set()
            for chunk in chunks:
                lat = chunk.get("latitude")
                lon = chunk.get("longitude")
                if lat is None or lon is None:
                    continue
                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    bbox_ids.add(chunk.get("chunk_id", ""))

            result_ids = bbox_ids if result_ids is None else result_ids & bbox_ids

        # Radius filter (need full scan)
        if center is not None and radius_km is not None:
            center_lon, center_lat = center
            chunks = self._load_chunks()
            radius_ids = set()
            for chunk in chunks:
                lat = chunk.get("latitude")
                lon = chunk.get("longitude")
                if lat is None or lon is None:
                    continue
                dist = self._haversine_distance(center_lat, center_lon, lat, lon)
                if dist <= radius_km:
                    radius_ids.add(chunk.get("chunk_id", ""))

            result_ids = radius_ids if result_ids is None else result_ids & radius_ids

        return result_ids or set()

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

    def filter(
        self,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]] = None,
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]] = None,
        combine_mode: str = "AND",
    ) -> DimensionalFilterResult:
        """
        Apply combined temporal and spatial filters.

        Args:
            temporal_filter: Temporal filter criteria
            spatial_filter: Spatial filter criteria
            combine_mode: How to combine filters: 'AND' (intersection) or 'OR' (union)

        Returns:
            DimensionalFilterResult with matching chunk IDs
        """
        applied_filters = {}
        filter_stats = {}

        # Convert dict to dataclass if needed
        if isinstance(temporal_filter, dict):
            temporal_filter = TemporalFilter(**temporal_filter)
        if isinstance(spatial_filter, dict):
            spatial_filter = SpatialFilter(**spatial_filter)

        temporal_ids = None
        spatial_ids = None

        # Apply temporal filter
        if temporal_filter is not None:
            temporal_ids = self.filter_temporal(
                year=temporal_filter.year,
                quarter=temporal_filter.quarter,
                month=temporal_filter.month,
                start_date=temporal_filter.start_date,
                end_date=temporal_filter.end_date,
            )
            applied_filters["temporal"] = {
                k: v for k, v in temporal_filter.__dict__.items() if v is not None
            }
            filter_stats["temporal_matches"] = len(temporal_ids)

        # Apply spatial filter
        if spatial_filter is not None:
            spatial_ids = self.filter_spatial(
                region=spatial_filter.region,
                country=spatial_filter.country,
                bbox=spatial_filter.bbox,
                center=spatial_filter.center,
                radius_km=spatial_filter.radius_km,
            )
            applied_filters["spatial"] = {
                k: v for k, v in spatial_filter.__dict__.items() if v is not None
            }
            filter_stats["spatial_matches"] = len(spatial_ids)

        # Combine results
        if temporal_ids is not None and spatial_ids is not None:
            if combine_mode == "AND":
                result_ids = temporal_ids & spatial_ids
            else:  # OR
                result_ids = temporal_ids | spatial_ids
        elif temporal_ids is not None:
            result_ids = temporal_ids
        elif spatial_ids is not None:
            result_ids = spatial_ids
        else:
            result_ids = set()

        filter_stats["final_matches"] = len(result_ids)
        filter_stats["combine_mode"] = combine_mode

        return DimensionalFilterResult(
            chunk_ids=result_ids,
            applied_filters=applied_filters,
            filter_stats=filter_stats,
        )

    def get_available_temporal_keys(self) -> List[str]:
        """Get available temporal filter keys from index."""
        return sorted(self.temporal_index.keys())

    def get_available_spatial_keys(self) -> List[str]:
        """Get available spatial filter keys from index."""
        return sorted(self.spatial_index.keys())
