"""
Dimension-agnostic spatiotemporal clustering and event burst detection.

Supports clustering on any combination of dimensions configured in YAML:
- Temporal clustering (time windows, bursts) - FIXED dimension
- Spatial clustering (DBSCAN, geographic proximity) - FIXED dimension
- Categorical clustering (group by any categorical dimension values)
- Multi-dimensional clustering (spatiotemporal + any custom dimensions from config)

All custom dimensions are loaded from config YAML - no hardcoding.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


class EventClusterAnalyzer:
    """
    Dimension-agnostic event clustering and burst detection.

    Features:
    - Temporal burst detection (identify spikes in activity)
    - Spatial clustering (DBSCAN with geodesic distance)
    - Spatiotemporal clustering (combined space-time proximity)
    - Categorical grouping (group by any categorical dimension values)
    - Multi-dimensional clustering (any combination of dimensions from config)

    Only fixed dimensions (temporal, spatial) are hardcoded.
    All other dimensions are auto-detected from extraction results.

    Example:
        analyzer = EventClusterAnalyzer(
            temporal_window='1D',          # 1-day time windows
            spatial_radius_km=50,          # 50km radius for spatial clusters
            min_cluster_size=2             # Minimum 2 events per cluster
        )

        # Works with any extraction results - dimensions auto-detected
        clusters = analyzer.detect_clusters(
            extraction_results=results,
            dimensions=None  # None = use all discovered dimensions
        )

        # Or specify which dimensions to use for clustering
        clusters = analyzer.detect_clusters(
            extraction_results=results,
            dimensions=['temporal', 'spatial']  # Add any custom dims from your config
        )
    """

    def __init__(
        self,
        temporal_window: str = '1D',      # Pandas time frequency string
        spatial_radius_km: float = 50,    # kilometers
        min_cluster_size: int = 2,
        eps_temporal_hours: float = 24,   # For DBSCAN temporal clustering
    ):
        """
        Initialize cluster analyzer.

        Args:
            temporal_window: Time window for burst detection (e.g., '1D', '6H', '1W')
            spatial_radius_km: Radius for spatial clustering (kilometers)
            min_cluster_size: Minimum events per cluster
            eps_temporal_hours: Maximum temporal distance for DBSCAN (hours)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering")

        self.temporal_window = temporal_window
        self.spatial_radius_km = spatial_radius_km
        self.min_cluster_size = min_cluster_size
        self.eps_temporal_hours = eps_temporal_hours

    def detect_clusters(
        self,
        extraction_results: List[Dict[str, Any]],
        dimensions: Optional[List[str]] = None,
        clustering_mode: str = 'spatiotemporal'
    ) -> Dict[str, Any]:
        """
        Detect clusters across multiple dimensions.

        Args:
            extraction_results: List of extraction results from pipeline
            dimensions: Dimensions to consider (None = all enabled dimensions)
            clustering_mode: 'temporal', 'spatial', 'spatiotemporal', 'categorical', 'multi'

        Returns:
            Dictionary with clustering results:
                - clusters: List of cluster objects
                - burst_periods: Temporal bursts
                - statistics: Cluster statistics
        """
        logger.info(f"Detecting {clustering_mode} clusters...")

        # Extract events from results
        events = self._extract_events(extraction_results, dimensions)

        if not events:
            logger.warning("No events found with required dimensions")
            return {'clusters': [], 'burst_periods': [], 'statistics': {}}

        logger.info(f"Extracted {len(events)} events for clustering")

        # Perform clustering based on mode
        if clustering_mode == 'temporal':
            clusters = self._cluster_temporal(events)
        elif clustering_mode == 'spatial':
            clusters = self._cluster_spatial(events)
        elif clustering_mode == 'spatiotemporal':
            clusters = self._cluster_spatiotemporal(events)
        elif clustering_mode == 'categorical':
            clusters = self._cluster_categorical(events)
        elif clustering_mode == 'multi':
            clusters = self._cluster_multidimensional(events, dimensions)
        else:
            raise ValueError(f"Unknown clustering mode: {clustering_mode}")

        # Detect temporal bursts
        burst_periods = self._detect_bursts(events)

        # Calculate statistics
        statistics = self._calculate_statistics(events, clusters, burst_periods)

        logger.info(f"✓ Found {len(clusters)} clusters, {len(burst_periods)} burst periods")

        return {
            'clusters': clusters,
            'burst_periods': burst_periods,
            'statistics': statistics,
            'events': events  # Include original events for reference
        }

    def _extract_events(
        self,
        extraction_results: List[Dict[str, Any]],
        dimensions: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Extract events from extraction results with all dimensions."""
        events = []

        for result in extraction_results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Filter dimensions if specified
            if dimensions:
                entities = {k: v for k, v in entities.items() if k in dimensions}

            if not entities:
                continue

            # Create event with all dimension data
            event = {
                'document_id': result.get('document_id', 'unknown'),
                'document_title': result.get('document_title', 'Unknown'),
                'doc_id': result.get('doc_id', result.get('chunk_id', '')),
                'source': result.get('source', 'Unknown'),
                'text': result.get('text', ''),
                'dimensions': {}
            }

            # Extract dimension-specific data
            for dim_name, dim_entities in entities.items():
                if not dim_entities:
                    continue

                # Store all entities for this dimension
                event['dimensions'][dim_name] = []

                for entity in dim_entities:
                    dim_data = {
                        'text': entity.get('text', ''),
                        'raw': entity  # Keep original entity data
                    }

                    # Extract temporal data
                    if 'normalized' in entity:
                        dim_data['temporal_value'] = entity.get('normalized')
                        dim_data['datetime'] = self._parse_datetime(entity.get('normalized'))

                    # Extract spatial data
                    if 'latitude' in entity and 'longitude' in entity:
                        dim_data['latitude'] = entity.get('latitude')
                        dim_data['longitude'] = entity.get('longitude')
                        dim_data['location_type'] = entity.get('location_type', 'unknown')

                    # Extract categorical data
                    if 'category' in entity:
                        dim_data['category'] = entity.get('category')

                    # Extract confidence if available
                    if 'confidence' in entity:
                        dim_data['confidence'] = entity.get('confidence', 1.0)

                    event['dimensions'][dim_name].append(dim_data)

            events.append(event)

        return events

    def _cluster_temporal(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster events by temporal proximity using DBSCAN."""
        # Extract temporal dimension
        temporal_events = []
        for event in events:
            temporal_entities = event['dimensions'].get('temporal', [])
            for entity in temporal_entities:
                if entity.get('datetime'):
                    temporal_events.append({
                        'event': event,
                        'datetime': entity['datetime'],
                        'entity': entity
                    })

        if len(temporal_events) < self.min_cluster_size:
            return []

        # Convert to timestamps for DBSCAN
        timestamps = np.array([
            e['datetime'].timestamp() for e in temporal_events
        ]).reshape(-1, 1)

        # DBSCAN clustering (eps in seconds)
        eps_seconds = self.eps_temporal_hours * 3600
        dbscan = DBSCAN(eps=eps_seconds, min_samples=self.min_cluster_size)
        labels = dbscan.fit_predict(timestamps)

        # Group into clusters
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Exclude noise
                clusters[label].append(temporal_events[i])

        # Format clusters
        result_clusters = []
        for cluster_id, cluster_events in clusters.items():
            datetimes = [e['datetime'] for e in cluster_events]
            result_clusters.append({
                'cluster_id': f'temporal_{cluster_id}',
                'cluster_type': 'temporal',
                'size': len(cluster_events),
                'events': cluster_events,
                'centroid': {
                    'datetime': min(datetimes) + (max(datetimes) - min(datetimes)) / 2,
                    'time_range': {
                        'start': min(datetimes).isoformat(),
                        'end': max(datetimes).isoformat()
                    }
                }
            })

        return result_clusters

    def _cluster_spatial(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster events by spatial proximity using DBSCAN."""
        # Extract spatial dimension
        spatial_events = []
        for event in events:
            spatial_entities = event['dimensions'].get('spatial', [])
            for entity in spatial_entities:
                if entity.get('latitude') and entity.get('longitude'):
                    spatial_events.append({
                        'event': event,
                        'latitude': entity['latitude'],
                        'longitude': entity['longitude'],
                        'entity': entity
                    })

        if len(spatial_events) < self.min_cluster_size:
            return []

        # Extract coordinates
        coords = np.array([
            [e['latitude'], e['longitude']] for e in spatial_events
        ])

        # DBSCAN with haversine distance (radius in km)
        # eps in radians = km / Earth radius
        eps_radians = self.spatial_radius_km / 6371.0
        dbscan = DBSCAN(
            eps=eps_radians,
            min_samples=self.min_cluster_size,
            metric='haversine'
        )
        labels = dbscan.fit_predict(np.radians(coords))

        # Group into clusters
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Exclude noise
                clusters[label].append(spatial_events[i])

        # Format clusters
        result_clusters = []
        for cluster_id, cluster_events in clusters.items():
            lats = [e['latitude'] for e in cluster_events]
            lons = [e['longitude'] for e in cluster_events]
            result_clusters.append({
                'cluster_id': f'spatial_{cluster_id}',
                'cluster_type': 'spatial',
                'size': len(cluster_events),
                'events': cluster_events,
                'centroid': {
                    'latitude': np.mean(lats),
                    'longitude': np.mean(lons),
                    'bbox': {
                        'min_lat': min(lats),
                        'max_lat': max(lats),
                        'min_lon': min(lons),
                        'max_lon': max(lons)
                    }
                }
            })

        return result_clusters

    def _cluster_spatiotemporal(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster events by both spatial and temporal proximity."""
        # Extract events with both spatial and temporal dimensions
        st_events = []
        for event in events:
            temporal_entities = event['dimensions'].get('temporal', [])
            spatial_entities = event['dimensions'].get('spatial', [])

            # Create cross-product of temporal and spatial entities
            for t_entity in temporal_entities:
                if not t_entity.get('datetime'):
                    continue
                for s_entity in spatial_entities:
                    if not (s_entity.get('latitude') and s_entity.get('longitude')):
                        continue

                    st_events.append({
                        'event': event,
                        'datetime': t_entity['datetime'],
                        'latitude': s_entity['latitude'],
                        'longitude': s_entity['longitude'],
                        'temporal_entity': t_entity,
                        'spatial_entity': s_entity
                    })

        if len(st_events) < self.min_cluster_size:
            return []

        # Normalize spatiotemporal features
        # Temporal: normalize to days since earliest
        datetimes = [e['datetime'] for e in st_events]
        min_datetime = min(datetimes)
        temporal_features = np.array([
            (e['datetime'] - min_datetime).total_seconds() / 86400  # Days
            for e in st_events
        ])

        # Spatial: convert to radians
        spatial_features = np.radians(np.array([
            [e['latitude'], e['longitude']] for e in st_events
        ]))

        # Combine features (weighted)
        # Scale temporal to match spatial scale (1 day ~ 100 km for weighting)
        temporal_weight = 0.01  # 1 day = 0.01 in spatial units
        combined_features = np.column_stack([
            spatial_features,
            temporal_features.reshape(-1, 1) * temporal_weight
        ])

        # DBSCAN clustering
        eps_combined = self.spatial_radius_km / 6371.0  # Convert to radians
        dbscan = DBSCAN(eps=eps_combined, min_samples=self.min_cluster_size)
        labels = dbscan.fit_predict(combined_features)

        # Group into clusters
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Exclude noise
                clusters[label].append(st_events[i])

        # Format clusters
        result_clusters = []
        for cluster_id, cluster_events in clusters.items():
            lats = [e['latitude'] for e in cluster_events]
            lons = [e['longitude'] for e in cluster_events]
            datetimes = [e['datetime'] for e in cluster_events]

            result_clusters.append({
                'cluster_id': f'spatiotemporal_{cluster_id}',
                'cluster_type': 'spatiotemporal',
                'size': len(cluster_events),
                'events': cluster_events,
                'centroid': {
                    'latitude': np.mean(lats),
                    'longitude': np.mean(lons),
                    'datetime': min(datetimes) + (max(datetimes) - min(datetimes)) / 2,
                    'time_range': {
                        'start': min(datetimes).isoformat(),
                        'end': max(datetimes).isoformat()
                    },
                    'bbox': {
                        'min_lat': min(lats),
                        'max_lat': max(lats),
                        'min_lon': min(lons),
                        'max_lon': max(lons)
                    }
                }
            })

        return result_clusters

    def _cluster_categorical(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group events by categorical dimension values."""
        # Group by all categorical dimensions
        category_groups = defaultdict(lambda: defaultdict(list))

        for event in events:
            for dim_name, dim_entities in event['dimensions'].items():
                for entity in dim_entities:
                    if 'category' in entity:
                        category_value = entity['category']
                        category_groups[dim_name][category_value].append({
                            'event': event,
                            'entity': entity
                        })

        # Format as clusters
        result_clusters = []
        cluster_id = 0
        for dim_name, categories in category_groups.items():
            for category_value, category_events in categories.items():
                if len(category_events) >= self.min_cluster_size:
                    result_clusters.append({
                        'cluster_id': f'categorical_{dim_name}_{cluster_id}',
                        'cluster_type': 'categorical',
                        'dimension': dim_name,
                        'category_value': category_value,
                        'size': len(category_events),
                        'events': category_events
                    })
                    cluster_id += 1

        return result_clusters

    def _cluster_multidimensional(
        self,
        events: List[Dict[str, Any]],
        dimensions: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Cluster by multiple dimensions simultaneously.

        Combines spatiotemporal + categorical dimensions for fine-grained clustering.
        Example: measles cases in Perth during March 2024.
        """
        # First do spatiotemporal clustering
        st_clusters = self._cluster_spatiotemporal(events)

        # Then split each spatiotemporal cluster by categorical dimensions
        result_clusters = []

        for st_cluster in st_clusters:
            # Group events in this cluster by categories
            category_groups = defaultdict(list)

            for cluster_event in st_cluster['events']:
                event = cluster_event['event']

                # Build category signature (all categorical dimensions)
                category_sig = []
                for dim_name, dim_entities in event['dimensions'].items():
                    if dim_name in ['temporal', 'spatial']:
                        continue
                    for entity in dim_entities:
                        if 'category' in entity:
                            category_sig.append(f"{dim_name}:{entity['category']}")

                category_key = tuple(sorted(category_sig)) if category_sig else ('uncategorized',)
                category_groups[category_key].append(cluster_event)

            # Create sub-clusters
            for category_key, category_events in category_groups.items():
                if len(category_events) >= self.min_cluster_size:
                    result_clusters.append({
                        'cluster_id': f"{st_cluster['cluster_id']}_cat_{len(result_clusters)}",
                        'cluster_type': 'multidimensional',
                        'parent_cluster': st_cluster['cluster_id'],
                        'categories': dict([cat.split(':', 1) for cat in category_key if cat != 'uncategorized']),
                        'size': len(category_events),
                        'events': category_events,
                        'centroid': st_cluster['centroid']  # Inherit spatiotemporal centroid
                    })

        return result_clusters

    def _detect_bursts(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect temporal bursts (sudden increases in activity).

        Uses sliding time windows to identify periods with abnormally high activity.
        """
        import pandas as pd

        # Extract all temporal values
        datetimes = []
        for event in events:
            temporal_entities = event['dimensions'].get('temporal', [])
            for entity in temporal_entities:
                if entity.get('datetime'):
                    datetimes.append(entity['datetime'])

        if not datetimes:
            return []

        # Create time series
        df = pd.DataFrame({'datetime': datetimes})
        df['count'] = 1

        # Resample by time window
        df = df.set_index('datetime').resample(self.temporal_window).count()

        # Detect bursts (count > mean + 2*std)
        mean_count = df['count'].mean()
        std_count = df['count'].std()
        burst_threshold = mean_count + 2 * std_count

        bursts = df[df['count'] > burst_threshold]

        # Format burst periods
        burst_periods = []
        for timestamp, row in bursts.iterrows():
            burst_periods.append({
                'start': timestamp.isoformat(),
                'end': (timestamp + pd.Timedelta(self.temporal_window)).isoformat(),
                'event_count': int(row['count']),
                'burst_intensity': (row['count'] - mean_count) / std_count if std_count > 0 else 0
            })

        return burst_periods

    def _calculate_statistics(
        self,
        events: List[Dict[str, Any]],
        clusters: List[Dict[str, Any]],
        burst_periods: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate clustering statistics."""
        # Count events per dimension
        dimension_counts = defaultdict(int)
        for event in events:
            for dim_name in event['dimensions'].keys():
                dimension_counts[dim_name] += len(event['dimensions'][dim_name])

        # Cluster size distribution
        cluster_sizes = [c['size'] for c in clusters]

        return {
            'total_events': len(events),
            'total_clusters': len(clusters),
            'total_bursts': len(burst_periods),
            'dimension_counts': dict(dimension_counts),
            'cluster_sizes': {
                'min': min(cluster_sizes) if cluster_sizes else 0,
                'max': max(cluster_sizes) if cluster_sizes else 0,
                'mean': np.mean(cluster_sizes) if cluster_sizes else 0,
                'median': np.median(cluster_sizes) if cluster_sizes else 0
            },
            'clustering_params': {
                'temporal_window': self.temporal_window,
                'spatial_radius_km': self.spatial_radius_km,
                'min_cluster_size': self.min_cluster_size,
                'eps_temporal_hours': self.eps_temporal_hours
            }
        }

    def _parse_datetime(self, iso_string: str) -> Optional[datetime]:
        """Parse ISO 8601 datetime string."""
        try:
            # Handle intervals
            if '/' in iso_string:
                iso_string = iso_string.split('/')[0]

            # Handle date-only strings
            if 'T' not in iso_string:
                iso_string = f"{iso_string}T00:00:00"

            return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
