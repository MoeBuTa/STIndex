"""
Folium animated map visualization for health surveillance events.

Creates an interactive map with:
- Health events plotted by location
- Timeline animation showing events over time
- Color coding by disease type
- Popups with event details
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger


def load_extraction_results(results_file: str) -> List[Dict[str, Any]]:
    """Load extraction results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} extraction results")
    return results


def build_health_events(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build structured health events from extraction results.

    Links spatial, temporal, and categorical dimensions into complete events.
    """
    events = []

    for result in results:
        if not result.get('extraction', {}).get('success'):
            continue

        entities = result['extraction'].get('entities', {})

        # Get dimensions
        spatial_entities = entities.get('spatial', [])
        temporal_entities = entities.get('temporal', [])
        event_types = entities.get('event_type', [])
        venue_types = entities.get('venue_type', [])
        diseases = entities.get('disease', [])

        # Link dimensions (simple strategy: all combinations)
        # TODO: Improve with proximity-based linking
        for spatial in spatial_entities:
            for temporal in temporal_entities:
                event = {
                    'document_title': result.get('document_title', 'Unknown'),
                    'source': result.get('source', 'Unknown'),
                    # Spatial
                    'location': spatial.get('text', ''),
                    'latitude': spatial.get('latitude'),
                    'longitude': spatial.get('longitude'),
                    # Temporal
                    'time': temporal.get('normalized', ''),
                    'time_text': temporal.get('text', ''),
                    # Categorical
                    'event_type': event_types[0].get('category') if event_types else 'unknown',
                    'venue_type': venue_types[0].get('category') if venue_types else 'unknown',
                    'disease': diseases[0].get('category') if diseases else 'unknown',
                }

                # Parse time for sorting
                try:
                    # Handle intervals (take start time)
                    time_str = event['time'].split('/')[0] if '/' in event['time'] else event['time']
                    event['datetime'] = datetime.fromisoformat(time_str)
                except:
                    event['datetime'] = None

                events.append(event)

    logger.info(f"Built {len(events)} health events from extraction results")
    return events


def create_animated_map(
    events: List[Dict[str, Any]],
    output_file: str = "case_studies/public_health/data/results/health_events_map.html"
):
    """
    Create animated Folium map showing health events over time.

    Args:
        events: List of health events with spatial and temporal data
        output_file: Path to save HTML map
    """
    logger.info("Creating animated health surveillance map...")

    # Filter events with valid coordinates and times
    valid_events = [e for e in events if e['latitude'] and e['longitude'] and e['datetime']]
    logger.info(f"Using {len(valid_events)} events with valid coordinates and timestamps")

    if not valid_events:
        logger.error("No valid events to visualize!")
        return

    # Calculate map center
    avg_lat = sum(e['latitude'] for e in valid_events) / len(valid_events)
    avg_lon = sum(e['longitude'] for e in valid_events) / len(valid_events)

    # Create base map
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )

    # Disease color mapping
    disease_colors = {
        'measles': 'red',
        'influenza': 'blue',
        'covid19': 'purple',
        'pertussis': 'orange',
        'dengue': 'green',
        'unknown': 'gray',
    }

    # Event type icon mapping
    event_icons = {
        'exposure_site': 'exclamation-triangle',
        'case_report': 'user',
        'outbreak_alert': 'bell',
        'surveillance_update': 'chart-line',
        'vaccination_clinic': 'syringe',
        'unknown': 'question',
    }

    # Prepare features for TimestampedGeoJson
    features = []

    for event in sorted(valid_events, key=lambda e: e['datetime']):
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">{event['disease'].title()} Event</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Location:</b></td><td>{event['location']}</td></tr>
                <tr><td><b>Time:</b></td><td>{event['time_text']}</td></tr>
                <tr><td><b>Event Type:</b></td><td>{event['event_type'].replace('_', ' ').title()}</td></tr>
                <tr><td><b>Venue:</b></td><td>{event['venue_type'].replace('_', ' ').title()}</td></tr>
                <tr><td><b>Source:</b></td><td>{event['source']}</td></tr>
            </table>
        </div>
        """

        # Create GeoJSON feature
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [event['longitude'], event['latitude']]
            },
            'properties': {
                'time': event['datetime'].isoformat(),
                'popup': popup_html,
                'icon': event_icons.get(event['event_type'], 'question'),
                'iconstyle': {
                    'iconUrl': '',
                    'iconSize': [20, 20],
                    'fillColor': disease_colors.get(event['disease'], 'gray'),
                    'color': disease_colors.get(event['disease'], 'gray'),
                    'fillOpacity': 0.8,
                    'weight': 2
                },
                'style': {
                    'color': disease_colors.get(event['disease'], 'gray'),
                    'weight': 2,
                    'fillColor': disease_colors.get(event['disease'], 'gray'),
                    'fillOpacity': 0.6
                }
            }
        }
        features.append(feature)

    # Create TimestampedGeoJson layer
    timestamped_geojson = TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='P1D',  # Period of 1 day
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True
    )

    timestamped_geojson.add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 200px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
        <p style="margin: 0 0 5px 0;"><b>Disease Types</b></p>
        <p style="margin: 2px 0;"><span style="color: red;">●</span> Measles</p>
        <p style="margin: 2px 0;"><span style="color: blue;">●</span> Influenza</p>
        <p style="margin: 2px 0;"><span style="color: purple;">●</span> COVID-19</p>
        <p style="margin: 2px 0;"><span style="color: orange;">●</span> Pertussis</p>
        <p style="margin: 2px 0;"><span style="color: green;">●</span> Dengue</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    logger.info(f"✓ Map saved to: {output_path}")
    logger.info(f"  Open in browser to view animated timeline")


def create_visualization(
    results_file: str = "case_studies/public_health/data/results/batch_extraction_results.json",
    output_file: str = "case_studies/public_health/data/results/health_events_map.html"
):
    """
    Main function to create visualization from extraction results.

    Args:
        results_file: Path to extraction results JSON
        output_file: Path to save HTML map
    """
    logger.info("=" * 80)
    logger.info("Health Surveillance Map Visualization")
    logger.info("=" * 80)

    # Load results
    results = load_extraction_results(results_file)

    # Build events
    events = build_health_events(results)

    if not events:
        logger.error("No events to visualize!")
        return

    # Create map
    create_animated_map(events, output_file)

    logger.info("\n" + "=" * 80)
    logger.info("Visualization complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create animated map visualization from extraction results"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="case_studies/public_health/data/results/batch_extraction_results.json",
        help="Path to extraction results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="case_studies/public_health/data/results/health_events_map.html",
        help="Path to save HTML map file"
    )

    args = parser.parse_args()

    try:
        create_visualization(args.results, args.output)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
