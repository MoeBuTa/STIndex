"""
OpenStreetMap context provider for spatial disambiguation.

Queries Overpass API for nearby geographic features to improve
location disambiguation accuracy (GeoLLM paper: 3.3x improvement).
"""

import math
from typing import Dict, List, Optional, Tuple

import requests
from geopy.distance import geodesic
from loguru import logger


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
        max_results: int = 10
    ):
        """
        Initialize OSM context provider.

        Args:
            overpass_url: Overpass API endpoint URL
            timeout: Request timeout in seconds
            max_results: Maximum number of nearby locations to return
        """
        self.overpass_url = overpass_url
        self.timeout = timeout
        self.max_results = max_results

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
