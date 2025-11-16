"""
OpenStreetMap context provider for spatial disambiguation.

Queries Overpass API for nearby geographic features to improve
location disambiguation accuracy (GeoLLM paper: 3.3x improvement).
"""

import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from geopy.distance import geodesic
from loguru import logger

from stindex.utils.constants import PROJECT_DIR


class OSMContextProvider:
    """
    Provide nearby location context from OpenStreetMap.

    Uses Overpass API to find nearby Points of Interest (POIs) that can help
    disambiguate location mentions. For example, finding "Broome" with nearby
    "Roebuck Bay" helps distinguish it from other places named Broome.

    Based on GeoLLM research (ICLR 2024) showing 3.3x improvement in spatial
    disambiguation when including nearby location information.
    """

    def __init__(
        self,
        overpass_url: str = "https://overpass-api.de/api/interpreter",
        timeout: int = 30,
        max_results: int = 10,
        rate_limit: float = 1.0,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize OSM context provider.

        Args:
            overpass_url: Overpass API endpoint URL
            timeout: Request timeout in seconds
            max_results: Maximum number of nearby locations to return
            rate_limit: Minimum seconds between API requests (default: 1.0)
            cache_enabled: Enable file-based caching (default: True)
            cache_dir: Cache directory path (default: data/cache/osm_context)
        """
        self.overpass_url = overpass_url
        self.timeout = timeout
        self.max_results = max_results

        # Rate limiting
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self._rate_limit_lock = threading.Lock()

        # Caching
        self.cache_enabled = cache_enabled
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(PROJECT_DIR) / "data/cache/osm_context"

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"OSM context cache enabled: {self.cache_dir}")

    def _get_cache_key(self, location: Tuple[float, float], radius_km: float) -> str:
        """
        Generate cache key for a location query.

        Args:
            location: (lat, lon) tuple
            radius_km: Search radius

        Returns:
            Cache key string
        """
        lat, lon = location
        # Round to 4 decimal places (~11m precision) for cache key
        return f"{lat:.4f}_{lon:.4f}_{radius_km}"

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """
        Load nearby locations from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached nearby locations or None if not found
        """
        if not self.cache_enabled:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"OSM context cache hit: {cache_key}")
                return data
            except Exception as e:
                logger.debug(f"Failed to load from cache: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, nearby_locations: List[Dict]):
        """
        Save nearby locations to cache.

        Args:
            cache_key: Cache key
            nearby_locations: List of nearby locations
        """
        if not self.cache_enabled:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(nearby_locations, f, indent=2)
            logger.debug(f"OSM context saved to cache: {cache_key}")
        except Exception as e:
            logger.debug(f"Failed to save to cache: {e}")

    def _wait_for_rate_limit(self):
        """Thread-safe rate limiting for API requests."""
        with self._rate_limit_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit:
                wait_time = self.rate_limit - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            self._last_request_time = time.time()

    def get_nearby_locations(
        self,
        location: Tuple[float, float],
        radius_km: float = 100
    ) -> List[Dict[str, any]]:
        """
        Get nearby POIs from OpenStreetMap using Overpass API.

        Args:
            location: (lat, lon) tuple
            radius_km: Search radius in kilometers

        Returns:
            List of nearby locations with names, distances, directions
            Example:
            [
                {
                    'name': 'Roebuck Bay',
                    'distance_km': 5.2,
                    'direction': 'SE',
                    'type': 'bay'
                },
                ...
            ]
        """
        lat, lon = location

        # Check cache first
        cache_key = self._get_cache_key(location, radius_km)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Apply rate limiting before API request
        self._wait_for_rate_limit()

        # Convert km to meters for Overpass API
        radius_m = int(radius_km * 1000)

        # Build Overpass QL query
        # Query for named features (nodes and ways) within radius
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          node(around:{radius_m},{lat},{lon})[name];
          way(around:{radius_m},{lat},{lon})[name];
        );
        out body {self.max_results * 2};
        """

        try:
            response = requests.post(
                self.overpass_url,
                data=query,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            logger.warning(f"Overpass API request failed: {e}")
            return []

        except Exception as e:
            logger.warning(f"Failed to parse Overpass API response: {e}")
            return []

        # Process elements
        elements = data.get('elements', [])
        nearby = []

        for element in elements:
            try:
                # Get coordinates
                if element.get('type') == 'node':
                    poi_lat = element.get('lat')
                    poi_lon = element.get('lon')
                elif element.get('type') == 'way':
                    # For ways, use center coordinates if available
                    center = element.get('center', {})
                    poi_lat = center.get('lat')
                    poi_lon = center.get('lon')
                else:
                    continue

                if not poi_lat or not poi_lon:
                    continue

                # Calculate distance
                distance = geodesic((lat, lon), (poi_lat, poi_lon)).km

                # Skip if too close (likely the same location)
                if distance < 0.1:
                    continue

                # Calculate bearing and convert to cardinal direction
                bearing = self._calculate_bearing(lat, lon, poi_lat, poi_lon)
                direction = self._bearing_to_direction(bearing)

                # Extract name and type
                tags = element.get('tags', {})
                name = tags.get('name')

                if not name:
                    continue

                # Determine feature type
                feature_type = self._determine_feature_type(tags)

                nearby.append({
                    'name': name,
                    'distance_km': round(distance, 1),
                    'direction': direction,
                    'type': feature_type,
                    'osm_type': element.get('type'),
                    'osm_id': element.get('id')
                })

            except Exception as e:
                logger.debug(f"Error processing OSM element: {e}")
                continue

        # Sort by distance and limit results
        nearby.sort(key=lambda x: x['distance_km'])
        nearby = nearby[:self.max_results]

        logger.debug(f"Found {len(nearby)} nearby locations within {radius_km}km")

        # Save to cache
        self._save_to_cache(cache_key, nearby)

        return nearby

    def _calculate_bearing(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate bearing between two points.

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2

        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad

        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - (
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        )

        bearing = math.atan2(x, y)
        bearing_degrees = (math.degrees(bearing) + 360) % 360

        return bearing_degrees

    def _bearing_to_direction(self, bearing: float) -> str:
        """
        Convert bearing to cardinal direction.

        Args:
            bearing: Bearing in degrees (0-360)

        Returns:
            Cardinal direction (N, NE, E, SE, S, SW, W, NW)
        """
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = round(bearing / 45) % 8
        return directions[index]

    def _determine_feature_type(self, tags: Dict[str, str]) -> str:
        """
        Determine feature type from OSM tags.

        Args:
            tags: OSM element tags

        Returns:
            Feature type string
        """
        # Priority order for determining type
        type_keys = [
            'place',           # city, town, village
            'natural',         # bay, beach, mountain
            'waterway',        # river, stream
            'landuse',         # residential, commercial
            'amenity',         # school, hospital
            'tourism',         # hotel, attraction
            'building',        # house, commercial
            'highway',         # primary, secondary
            'railway',         # station
        ]

        for key in type_keys:
            if key in tags:
                return tags[key]

        # Fall back to generic type
        return 'feature'

    def get_location_context_str(
        self,
        location: Tuple[float, float],
        radius_km: float = 100,
        max_display: int = 5
    ) -> str:
        """
        Get nearby locations as formatted string for LLM prompt.

        Args:
            location: (lat, lon) tuple
            radius_km: Search radius in kilometers
            max_display: Maximum number of locations to display

        Returns:
            Formatted string describing nearby locations
        """
        nearby = self.get_nearby_locations(location, radius_km)

        if not nearby:
            return ""

        lines = [f"Nearby geographic features (within {radius_km}km):"]
        for poi in nearby[:max_display]:
            lines.append(
                f"  - {poi['name']} ({poi['type']}): "
                f"{poi['distance_km']}km {poi['direction']}"
            )

        return "\n".join(lines)
