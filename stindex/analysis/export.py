"""
Export analysis results to static JSON files for frontend consumption.

Generates JSON files that can be loaded directly by frontend without backend API:
- extraction_results.json: All extraction results with all dimensions from config
- clusters.json: Spatiotemporal clusters
- story_arcs.json: Detected story arcs
- dimension_analysis.json: Statistical analysis for all dimensions
- metadata.json: Configuration and timestamps

Fixed dimensions: temporal, spatial
Custom dimensions: Auto-detected from extraction results (loaded from config YAML)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class AnalysisDataExporter:
    """
    Export analysis data to static JSON files for frontend.

    Generates a complete data package that can be loaded by a static frontend:
    - No backend API required
    - All data in JSON format
    - Optimized for web consumption (serializable, compact)

    Example:
        exporter = AnalysisDataExporter(output_dir="case_studies/public_health/frontend_data")

        exporter.export_all(
            extraction_results=results,
            clusters=clusters,
            story_arcs=story_arcs,
            dimension_analysis=analysis,
            metadata={'case_study': 'public_health', 'model': 'Qwen3-8B'}
        )

        # Generates:
        # frontend_data/
        # ├── extraction_results.json
        # ├── clusters.json
        # ├── story_arcs.json
        # ├── dimension_analysis.json
        # ├── events.json (flattened for map/timeline)
        # └── metadata.json
    """

    def __init__(self, output_dir: str):
        """
        Initialize exporter.

        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        extraction_results: List[Dict[str, Any]],
        clusters: Optional[Dict[str, Any]] = None,
        story_arcs: Optional[List[Dict[str, Any]]] = None,
        dimension_analysis: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export all analysis data to JSON files.

        Args:
            extraction_results: Extraction results from pipeline
            clusters: Cluster analysis results
            story_arcs: Story arc detection results
            dimension_analysis: Dimension analysis results
            metadata: Additional metadata (case study info, config, etc.)

        Returns:
            Dict mapping data type to file path
        """
        logger.info(f"Exporting analysis data to {self.output_dir}")

        exported_files = {}

        # 1. Export extraction results
        if extraction_results:
            filepath = self._export_json(
                'extraction_results.json',
                self._serialize_extraction_results(extraction_results)
            )
            exported_files['extraction_results'] = filepath

        # 2. Export clusters
        if clusters:
            filepath = self._export_json(
                'clusters.json',
                self._serialize_clusters(clusters)
            )
            exported_files['clusters'] = filepath

        # 3. Export story arcs
        if story_arcs:
            filepath = self._export_json(
                'story_arcs.json',
                self._serialize_story_arcs(story_arcs)
            )
            exported_files['story_arcs'] = filepath

        # 4. Export dimension analysis
        if dimension_analysis:
            filepath = self._export_json(
                'dimension_analysis.json',
                self._serialize_dimension_analysis(dimension_analysis)
            )
            exported_files['dimension_analysis'] = filepath

        # 5. Export flattened events (for map/timeline)
        if extraction_results:
            events = self._flatten_events(extraction_results)
            filepath = self._export_json('events.json', events)
            exported_files['events'] = filepath

        # 6. Export metadata
        metadata = metadata or {}
        metadata['export_timestamp'] = datetime.now().isoformat()
        metadata['file_count'] = len(exported_files)
        metadata['extraction_count'] = len(extraction_results) if extraction_results else 0

        filepath = self._export_json('metadata.json', metadata)
        exported_files['metadata'] = filepath

        logger.info(f"✓ Exported {len(exported_files)} files to {self.output_dir}")

        return exported_files

    def _export_json(self, filename: str, data: Any) -> str:
        """Export data to JSON file."""
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"  ✓ {filename}")
        return str(filepath)

    def _serialize_extraction_results(
        self,
        extraction_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Serialize extraction results for JSON export."""
        serialized = []

        for result in extraction_results:
            # Only keep essential fields
            serialized_result = {
                'document_id': result.get('document_id'),
                'document_title': result.get('document_title'),
                'chunk_id': result.get('chunk_id'),
                'source': result.get('source'),
                'text': result.get('text', '')[:500],  # Truncate long text
                'extraction': {
                    'success': result.get('extraction', {}).get('success', False),
                    'entities': result.get('extraction', {}).get('entities', {})
                }
            }
            serialized.append(serialized_result)

        return serialized

    def _serialize_clusters(
        self,
        clusters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Serialize cluster results for JSON export."""
        serialized = {
            'clusters': [],
            'burst_periods': clusters.get('burst_periods', []),
            'statistics': clusters.get('statistics', {})
        }

        # Serialize clusters (remove redundant event data)
        for cluster in clusters.get('clusters', []):
            serialized_cluster = {
                'cluster_id': cluster.get('cluster_id'),
                'cluster_type': cluster.get('cluster_type'),
                'size': cluster.get('size'),
                'centroid': cluster.get('centroid'),
                'category_value': cluster.get('category_value'),  # For categorical clusters
                'dimension': cluster.get('dimension'),  # For categorical clusters
                # Store minimal event references
                'event_ids': [
                    e['event'].get('chunk_id', '')
                    for e in cluster.get('events', [])
                    if 'event' in e
                ]
            }
            serialized['clusters'].append(serialized_cluster)

        return serialized

    def _serialize_story_arcs(
        self,
        story_arcs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Serialize story arcs for JSON export."""
        serialized = []

        for story in story_arcs:
            serialized_story = {
                'story_id': story.get('story_id'),
                'length': story.get('length'),
                'progression_type': story.get('progression_type'),
                'confidence': story.get('confidence'),
                'temporal_span': story.get('temporal_span'),
                'spatial_span': story.get('spatial_span'),
                'narrative_summary': story.get('narrative_summary'),
                'key_dimensions': story.get('key_dimensions'),
                # Store event references
                'event_ids': [
                    e.get('chunk_id', '')
                    for e in story.get('events', [])
                ]
            }
            serialized.append(serialized_story)

        return serialized

    def _serialize_dimension_analysis(
        self,
        dimension_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Serialize dimension analysis for JSON export."""
        # Already in JSON-friendly format, just clean up any numpy types
        return json.loads(json.dumps(dimension_analysis, default=str))

    def _flatten_events(
        self,
        extraction_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flatten extraction results into individual events for map/timeline.

        Each event has:
        - Unique ID
        - Timestamp (if available)
        - Location (if available)
        - All dimension entities
        - Source metadata
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
                        event = self._create_event(
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
                    event = self._create_event(
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
                    event = self._create_event(
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

    def _create_event(
        self,
        event_id: int,
        result: Dict[str, Any],
        temporal: Optional[Dict[str, Any]],
        spatial: Optional[Dict[str, Any]],
        all_entities: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a single flattened event."""
        event = {
            'id': event_id,
            'chunk_id': result.get('chunk_id'),
            'document_id': result.get('document_id'),
            'document_title': result.get('document_title'),
            'source': result.get('source'),
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

    def export_geojson(
        self,
        events: List[Dict[str, Any]],
        filename: str = 'events.geojson'
    ) -> str:
        """
        Export events as GeoJSON for map visualization.

        Args:
            events: Flattened events from _flatten_events()
            filename: Output filename

        Returns:
            File path
        """
        features = []

        for event in events:
            if 'spatial' not in event:
                continue

            spatial = event['spatial']
            lat = spatial.get('latitude')
            lon = spatial.get('longitude')

            if not (lat and lon):
                continue

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]  # GeoJSON uses [lon, lat]
                },
                'properties': {
                    'id': event['id'],
                    'location': spatial.get('text'),
                    'timestamp': event.get('temporal', {}).get('normalized'),
                    'document_title': event.get('document_title'),
                    'source': event.get('source'),
                    **event.get('dimensions', {})
                }
            }

            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        filepath = self._export_json(filename, geojson)
        return filepath
