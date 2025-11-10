"""
Generic map generator for geocoded spatial dimensions.

Creates interactive Folium maps with:
- Point markers for geocoded entities
- Popups with entity details
- Color coding by categorical dimensions
- Timeline animation for temporal dimensions
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    import folium
    from folium.plugins import MarkerCluster, TimestampedGeoJson
    FOLIUM_AVAILABLE = True
except ImportError:
    logger.warning("folium not installed. Install with: pip install folium")
    FOLIUM_AVAILABLE = False


class MapGenerator:
    """Generate interactive maps from extraction results."""

    def __init__(self):
        """Initialize map generator."""
        if not FOLIUM_AVAILABLE:
            raise ImportError("folium is required for map generation")

    def generate_map(
        self,
        results: List[Dict[str, Any]],
        output_file: str,
        temporal_dim: str = "temporal",
        spatial_dim: str = "spatial",
        category_dim: Optional[str] = None,
        animated: bool = True
    ) -> str:
        """
        Generate interactive map from extraction results.

        Args:
            results: List of extraction results
            output_file: Path to save HTML map
            temporal_dim: Name of temporal dimension for animation
            spatial_dim: Name of spatial dimension for geocoding
            category_dim: Name of categorical dimension for color coding
            animated: Create animated timeline if True

        Returns:
            Path to generated HTML file
        """
        logger.info("Generating interactive map...")

        # Extract events with spatial and temporal data
        events = self._extract_events(
            results,
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            category_dim=category_dim
        )

        if not events:
            logger.warning("No events with valid coordinates to visualize")
            return None

        # Create map based on mode
        if animated and temporal_dim and any(e.get('datetime') for e in events):
            map_obj = self._create_animated_map(events, category_dim)
        else:
            map_obj = self._create_static_map(events, category_dim)

        # Save map
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        map_obj.save(str(output_path))

        logger.info(f"✓ Map saved to: {output_path}")
        return str(output_path)

    def _extract_events(
        self,
        results: List[Dict[str, Any]],
        temporal_dim: str,
        spatial_dim: str,
        category_dim: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract events with spatial and temporal data."""
        events = []

        for result in results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Get dimension entities
            spatial_entities = entities.get(spatial_dim, [])
            temporal_entities = entities.get(temporal_dim, []) if temporal_dim else []
            category_entities = entities.get(category_dim, []) if category_dim else []

            # Create events (link spatial + temporal + category)
            for spatial in spatial_entities:
                lat = spatial.get('latitude')
                lon = spatial.get('longitude')

                if not (lat and lon):
                    continue

                # Get temporal info if available
                temporal_text = ""
                temporal_normalized = ""
                event_datetime = None

                if temporal_entities:
                    temporal = temporal_entities[0]  # Use first temporal
                    temporal_text = temporal.get('text', '')
                    temporal_normalized = temporal.get('normalized', '')

                    # Parse datetime with better error handling
                    try:
                        time_str = temporal_normalized.split('/')[0] if '/' in temporal_normalized else temporal_normalized

                        # Handle different ISO 8601 formats
                        # Extract just the date if it's a datetime string
                        if 'T' in time_str:
                            # Full datetime - keep it for better precision
                            event_datetime = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        else:
                            # Date only - parse as date and set to noon for better timeline display
                            event_datetime = datetime.fromisoformat(f"{time_str}T12:00:00")

                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Failed to parse datetime '{temporal_normalized}': {e}")
                        # Try fallback parsing
                        try:
                            from dateutil import parser
                            event_datetime = parser.parse(time_str)
                        except:
                            logger.warning(f"Could not parse temporal value: {temporal_normalized}")
                            pass

                # Get category if available
                category_text = ""
                category_value = "unknown"

                if category_entities:
                    cat = category_entities[0]  # Use first category
                    category_text = cat.get('text', '')
                    category_value = cat.get('category', 'unknown')

                event = {
                    'document_title': result.get('document_title', 'Unknown'),
                    'source': result.get('source', 'Unknown'),
                    'chunk_id': result.get('chunk_id', ''),
                    # Spatial
                    'location': spatial.get('text', ''),
                    'latitude': lat,
                    'longitude': lon,
                    # Temporal
                    'temporal_text': temporal_text,
                    'temporal_normalized': temporal_normalized,
                    'datetime': event_datetime,
                    # Category
                    'category_text': category_text,
                    'category': category_value,
                    # All entities for popup
                    'all_entities': entities
                }

                events.append(event)

        logger.info(f"Extracted {len(events)} events with valid coordinates")
        return events

    def _create_static_map(
        self,
        events: List[Dict[str, Any]],
        category_dim: Optional[str]
    ) -> folium.Map:
        """Create static map with markers."""
        # Calculate map center
        avg_lat = sum(e['latitude'] for e in events) / len(events)
        avg_lon = sum(e['longitude'] for e in events) / len(events)

        # Create base map
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )

        # Get unique categories and assign colors
        if category_dim:
            categories = list(set(e['category'] for e in events))
            color_map = self._get_color_map(categories)
        else:
            color_map = {}

        # Add markers
        marker_cluster = MarkerCluster().add_to(m)

        for event in events:
            # Create popup content
            popup_html = self._create_popup_html(event)

            # Get color
            color = color_map.get(event['category'], 'gray') if category_dim else 'blue'

            # Add marker
            folium.Marker(
                location=[event['latitude'], event['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=event['location'],
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(marker_cluster)

        # Add legend if category dimension
        if category_dim and color_map:
            self._add_legend(m, color_map, category_dim)

        return m

    def _create_animated_map(
        self,
        events: List[Dict[str, Any]],
        category_dim: Optional[str]
    ) -> folium.Map:
        """Create animated map with timeline."""
        # Filter events with datetime
        valid_events = [e for e in events if e.get('datetime')]
        logger.info(f"Creating animated map with {len(valid_events)} timestamped events")

        if not valid_events:
            logger.warning("No events with timestamps, falling back to static map")
            return self._create_static_map(events, category_dim)

        # Calculate map center
        avg_lat = sum(e['latitude'] for e in valid_events) / len(valid_events)
        avg_lon = sum(e['longitude'] for e in valid_events) / len(valid_events)

        # Create base map
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )

        # Get color map
        if category_dim:
            categories = list(set(e['category'] for e in valid_events))
            color_map = self._get_color_map(categories)
        else:
            color_map = {}

        # Prepare GeoJSON features with validation
        features = []
        skipped_features = 0

        for event in sorted(valid_events, key=lambda e: e['datetime']):
            try:
                # Validate coordinates
                lat = float(event['latitude'])
                lon = float(event['longitude'])

                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                    skipped_features += 1
                    continue

                # Get color
                color = color_map.get(event['category'], 'gray') if category_dim else 'blue'

                # Map folium colors to hex for TimestampedGeoJson
                color_hex_map = {
                    'red': '#d73027', 'blue': '#4575b4', 'green': '#91cf60',
                    'purple': '#9970ab', 'orange': '#fc8d59', 'darkred': '#a50026',
                    'lightred': '#f46d43', 'beige': '#fee090', 'darkblue': '#313695',
                    'darkgreen': '#1a9850', 'cadetblue': '#abd9e9', 'darkpurple': '#762a83',
                    'pink': '#fbb4b9', 'lightblue': '#c6dbef', 'lightgreen': '#d9f0a3',
                    'gray': '#969696', 'black': '#252525', 'lightgray': '#cccccc'
                }
                color_code = color_hex_map.get(color, '#4575b4')

                popup_html = self._create_popup_html(event)

                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]  # GeoJSON uses [lon, lat]
                    },
                    'properties': {
                        'time': event['datetime'].isoformat(),
                        'popup': popup_html,
                        'tooltip': event['location'],
                        'style': {
                            'color': color_code,
                            'fillColor': color_code,
                            'fillOpacity': 0.6,
                            'weight': 2
                        },
                        'icon': 'circle',
                        'iconstyle': {
                            'fillColor': color_code,
                            'fillOpacity': 0.8,
                            'stroke': 'true',
                            'radius': 8
                        }
                    }
                }
                features.append(feature)

            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid event: {e}")
                skipped_features += 1
                continue

        if skipped_features > 0:
            logger.warning(f"Skipped {skipped_features} invalid features")

        if not features:
            logger.warning("No valid features for animated map, falling back to static")
            return self._create_static_map(events, category_dim)

        logger.info(f"Created {len(features)} valid features for animated map")

        # Create TimestampedGeoJson layer
        try:
            timestamped_geojson = TimestampedGeoJson(
                {
                    'type': 'FeatureCollection',
                    'features': features
                },
                period='P1D',  # 1 day period
                add_last_point=True,
                auto_play=False,
                loop=False,
                max_speed=2,
                loop_button=True,
                date_options='YYYY-MM-DD',
                time_slider_drag_update=True,
                duration='P1D'  # Add explicit duration
            )

            timestamped_geojson.add_to(m)
            logger.info("✓ TimestampedGeoJson layer added successfully")

        except Exception as e:
            logger.error(f"Failed to create TimestampedGeoJson: {e}")
            logger.warning("Falling back to static map")
            return self._create_static_map(events, category_dim)

        # Add legend
        if category_dim and color_map:
            self._add_legend(m, color_map, category_dim)

        return m

    def _create_popup_html(self, event: Dict[str, Any]) -> str:
        """Create HTML content for popup."""
        html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{event['location']}</h4>
            <table style="width: 100%; border-collapse: collapse;">
        """

        # Add basic info
        if event.get('temporal_text'):
            html += f"<tr><td><b>Time:</b></td><td>{event['temporal_text']}</td></tr>"

        if event.get('category_text'):
            html += f"<tr><td><b>Category:</b></td><td>{event['category_text']}</td></tr>"

        # Add all entities
        all_entities = event.get('all_entities', {})
        for dim_name, dim_entities in all_entities.items():
            if dim_name not in ['temporal', 'spatial'] and dim_entities:
                # Show first entity of each dimension
                entity = dim_entities[0]
                text = entity.get('text', '') or entity.get('category', '')
                if text:
                    html += f"<tr><td><b>{dim_name.title()}:</b></td><td>{text}</td></tr>"

        html += f"""
                <tr><td><b>Source:</b></td><td>{event.get('source', 'Unknown')}</td></tr>
            </table>
        </div>
        """

        return html

    def _get_color_map(self, categories: List[str]) -> Dict[str, str]:
        """Get color mapping for categories."""
        # Folium color options
        colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
            'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue',
            'lightgreen', 'gray', 'black', 'lightgray'
        ]

        color_map = {}
        for i, category in enumerate(sorted(categories)):
            color_map[category] = colors[i % len(colors)]

        return color_map

    def _add_legend(
        self,
        map_obj: folium.Map,
        color_map: Dict[str, str],
        category_dim: str
    ):
        """Add legend to map."""
        legend_html = f'''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 220px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
            <p style="margin: 0 0 5px 0;"><b>{category_dim.replace('_', ' ').title()}</b></p>
        '''

        for category, color in sorted(color_map.items()):
            legend_html += f'<p style="margin: 2px 0;"><span style="color: {color};">●</span> {category.replace("_", " ").title()}</p>'

        legend_html += '</div>'

        map_obj.get_root().html.add_child(folium.Element(legend_html))
