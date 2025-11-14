#!/usr/bin/env python3
"""
Regenerate events.json from patched extraction_results.json.

This ensures events.json has the correct source information.
"""

import json
import sys
from pathlib import Path
from loguru import logger


def flatten_events(extraction_results):
    """
    Flatten extraction results into individual events for map/timeline.

    Based on stindex.analysis.export.AnalysisDataExporter._flatten_events
    """
    events = []
    event_id = 0

    for result in extraction_results:
        if not result.get('extraction', {}).get('success'):
            continue

        entities = result['extraction'].get('entities', {})

        # Extract temporal and spatial entities
        temporal_entities = entities.get('temporal', [])
        spatial_entities = entities.get('spatial', [])

        # Create events for each spatiotemporal combination
        if temporal_entities and spatial_entities:
            # Create cross-product of temporal and spatial
            for temporal in temporal_entities:
                for spatial in spatial_entities:
                    event = create_event(
                        event_id,
                        result,
                        temporal,
                        spatial,
                        entities
                    )
                    if event:
                        events.append(event)
                        event_id += 1

        elif temporal_entities:
            # Temporal only (no location)
            for temporal in temporal_entities:
                event = create_event(
                    event_id,
                    result,
                    temporal,
                    None,
                    entities
                )
                if event:
                    events.append(event)
                    event_id += 1

        elif spatial_entities:
            # Spatial only (no timestamp)
            for spatial in spatial_entities:
                event = create_event(
                    event_id,
                    result,
                    None,
                    spatial,
                    entities
                )
                if event:
                    events.append(event)
                    event_id += 1

    logger.info(f"  Flattened {len(events)} events from {len(extraction_results)} chunks")
    return events


def create_event(event_id, result, temporal, spatial, all_entities):
    """Create a single flattened event."""
    event = {
        'id': event_id,
        'chunk_id': result.get('chunk_id'),
        'document_id': result.get('document_id'),
        'document_title': result.get('document_title'),
        'source': result.get('source'),  # Now includes source!
        'text': result.get('text', '')[:200]  # Truncate for frontend
    }

    # Add temporal data
    if temporal:
        event['temporal'] = {
            'text': temporal.get('text'),
            'normalized': temporal.get('normalized'),
            'temporal_type': temporal.get('temporal_type')
        }

    # Add spatial data
    if spatial:
        lat = spatial.get('latitude')
        lon = spatial.get('longitude')

        if lat and lon:
            event['spatial'] = {
                'text': spatial.get('text'),
                'latitude': lat,
                'longitude': lon,
                'location_type': spatial.get('location_type'),
                'parent_region': spatial.get('parent_region')
            }

    # Add other dimensions
    event['dimensions'] = {}
    for dim_name, dim_entities in all_entities.items():
        if dim_name in ['temporal', 'spatial']:
            continue

        if dim_entities:
            # Store first entity of each dimension
            entity = dim_entities[0]
            event['dimensions'][dim_name] = {
                'text': entity.get('text'),
                'category': entity.get('category')
            }

    # Only return if has at least temporal or spatial
    if 'temporal' in event or 'spatial' in event:
        return event

    return None


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python regenerate_events.py <extraction_results.json> [output_events.json]")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else results_file.parent / "events.json"

    logger.info(f"Loading extraction results from: {results_file}")
    with open(results_file, 'r') as f:
        extraction_results = json.load(f)

    logger.info(f"✓ Loaded {len(extraction_results)} extraction results")

    logger.info("Flattening events...")
    events = flatten_events(extraction_results)

    logger.info(f"Saving events to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)

    logger.success(f"✓ Generated {len(events)} events")
    logger.success(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()