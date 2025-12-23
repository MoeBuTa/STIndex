"""
Story arc detection for connecting related events across space and time.

Extracts narrative structures from clustered events:
- Temporal progression (event sequences) - FIXED dimension
- Spatial diffusion (geographic spread) - FIXED dimension
- Entity co-occurrence (shared entities across any dimensions from config)
- Thematic evolution (changes in categorical dimensions from config)

All custom dimensions are loaded from config YAML - no hardcoding.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger


class StoryArcDetector:
    """
    Detect story arcs (narrative sequences) from clustered events.

    A story arc connects multiple events that share:
    - Temporal proximity (events happen in sequence)
    - Entity overlap (same entities across dimensions)
    - Spatial relationships (geographic progression)
    - Categorical similarity (related categories in custom dimensions)

    Only temporal and spatial dimensions are fixed.
    All other dimensions are auto-detected from extraction results.

    Example:
        detector = StoryArcDetector(
            max_temporal_gap_days=7,        # Events within 7 days can be linked
            min_entity_overlap=0.3,          # 30% shared entities required
            min_arc_length=2                 # At least 2 events per story
        )

        # Works with any configured dimensions
        story_arcs = detector.detect_story_arcs(
            clusters=clusters,
            events=events,
            dimensions=None  # None = use all discovered dimensions
        )
    """

    def __init__(
        self,
        max_temporal_gap_days: int = 7,
        min_entity_overlap: float = 0.3,
        min_arc_length: int = 2,
        min_confidence: float = 0.5
    ):
        """
        Initialize story arc detector.

        Args:
            max_temporal_gap_days: Maximum days between connected events
            min_entity_overlap: Minimum overlap ratio for entity linking (0.0-1.0)
            min_arc_length: Minimum events per story arc
            min_confidence: Minimum confidence for story arc
        """
        self.max_temporal_gap_days = max_temporal_gap_days
        self.min_entity_overlap = min_entity_overlap
        self.min_arc_length = min_arc_length
        self.min_confidence = min_confidence

    def detect_story_arcs(
        self,
        clusters: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        dimensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect story arcs from clustered events.

        Args:
            clusters: Clusters from EventClusterAnalyzer
            events: Original events
            dimensions: Dimensions to consider for story linking

        Returns:
            List of story arc objects with:
                - story_id: Unique identifier
                - events: List of connected events (temporally ordered)
                - progression_type: 'temporal', 'spatial', 'thematic'
                - narrative_summary: Key entities and progression
                - confidence: Story coherence score
        """
        logger.info("Detecting story arcs from clusters...")

        if not clusters:
            logger.warning("No clusters provided for story arc detection")
            return []

        # Build event relationship graph
        graph = self._build_event_graph(clusters, events, dimensions)

        # Find connected components (potential stories)
        story_arcs = []
        for component in nx.connected_components(graph.to_undirected()):
            if len(component) < self.min_arc_length:
                continue

            # Extract subgraph for this component
            subgraph = graph.subgraph(component)

            # Analyze story arc
            story_arc = self._analyze_story_arc(subgraph, events)

            if story_arc and story_arc['confidence'] >= self.min_confidence:
                story_arcs.append(story_arc)

        logger.info(f"✓ Detected {len(story_arcs)} story arcs")

        return sorted(story_arcs, key=lambda x: x['confidence'], reverse=True)

    def _build_event_graph(
        self,
        clusters: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        dimensions: Optional[List[str]]
    ) -> nx.DiGraph:
        """
        Build directed graph of event relationships.

        Edges represent:
        - Temporal sequence (earlier → later)
        - Entity overlap (shared entities)
        - Spatial proximity (nearby locations)
        """
        graph = nx.DiGraph()

        # Create event index
        event_index = {e.get('doc_id', e.get('chunk_id', '')): e for e in events if e.get('doc_id') or e.get('chunk_id')}

        # Add nodes from clusters
        for cluster in clusters:
            cluster_events = cluster.get('events', [])

            for cluster_event in cluster_events:
                # Extract base event
                if 'event' in cluster_event:
                    base_event = cluster_event['event']
                else:
                    base_event = cluster_event

                node_id = base_event.get('doc_id', base_event.get('chunk_id', f"event_{len(graph.nodes)}"))

                if not graph.has_node(node_id):
                    graph.add_node(
                        node_id,
                        event=base_event,
                        cluster_id=cluster.get('cluster_id'),
                        cluster_type=cluster.get('cluster_type')
                    )

        # Add edges based on relationships
        nodes = list(graph.nodes(data=True))

        for i, (node_id_1, data_1) in enumerate(nodes):
            event_1 = data_1['event']

            for node_id_2, data_2 in nodes[i+1:]:
                event_2 = data_2['event']

                # Calculate relationship strength
                relationship = self._calculate_relationship(
                    event_1, event_2, dimensions
                )

                if relationship['score'] > 0:
                    # Add directed edge (earlier → later)
                    if relationship['temporal_order'] == 'forward':
                        graph.add_edge(
                            node_id_1, node_id_2,
                            weight=relationship['score'],
                            relationship_type=relationship['type'],
                            shared_entities=relationship['shared_entities']
                        )
                    elif relationship['temporal_order'] == 'backward':
                        graph.add_edge(
                            node_id_2, node_id_1,
                            weight=relationship['score'],
                            relationship_type=relationship['type'],
                            shared_entities=relationship['shared_entities']
                        )
                    else:
                        # No temporal order, add undirected (both directions)
                        graph.add_edge(
                            node_id_1, node_id_2,
                            weight=relationship['score'],
                            relationship_type=relationship['type'],
                            shared_entities=relationship['shared_entities']
                        )

        return graph

    def _calculate_relationship(
        self,
        event_1: Dict[str, Any],
        event_2: Dict[str, Any],
        dimensions: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Calculate relationship strength between two events.

        Returns:
            Dict with:
                - score: Relationship strength (0.0-1.0)
                - type: 'temporal', 'spatial', 'entity_overlap', 'thematic'
                - temporal_order: 'forward', 'backward', 'none'
                - shared_entities: List of shared entity texts
        """
        # Extract entities from both events
        entities_1 = self._extract_entity_set(event_1, dimensions)
        entities_2 = self._extract_entity_set(event_2, dimensions)

        # Calculate entity overlap
        shared_entities = entities_1.intersection(entities_2)
        total_entities = len(entities_1.union(entities_2))

        if total_entities == 0:
            overlap_ratio = 0
        else:
            overlap_ratio = len(shared_entities) / total_entities

        # Check temporal relationship
        temporal_order = 'none'
        temporal_gap_days = None

        temporal_1 = self._get_earliest_datetime(event_1)
        temporal_2 = self._get_earliest_datetime(event_2)

        if temporal_1 and temporal_2:
            time_diff = (temporal_2 - temporal_1).total_seconds() / 86400  # Days

            if abs(time_diff) <= self.max_temporal_gap_days:
                if time_diff > 0:
                    temporal_order = 'forward'
                    temporal_gap_days = time_diff
                elif time_diff < 0:
                    temporal_order = 'backward'
                    temporal_gap_days = abs(time_diff)
                else:
                    temporal_order = 'simultaneous'
                    temporal_gap_days = 0

        # Calculate relationship score
        score = 0
        relationship_type = []

        # Entity overlap contribution
        if overlap_ratio >= self.min_entity_overlap:
            score += overlap_ratio * 0.6
            relationship_type.append('entity_overlap')

        # Temporal contribution
        if temporal_order != 'none' and temporal_gap_days is not None:
            # Decay score based on temporal distance
            temporal_score = max(0, 1 - (temporal_gap_days / self.max_temporal_gap_days))
            score += temporal_score * 0.4
            relationship_type.append('temporal')

        return {
            'score': min(score, 1.0),
            'type': '+'.join(relationship_type) if relationship_type else 'weak',
            'temporal_order': temporal_order,
            'temporal_gap_days': temporal_gap_days,
            'shared_entities': list(shared_entities)
        }

    def _analyze_story_arc(
        self,
        subgraph: nx.DiGraph,
        events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a connected component to extract story arc structure.

        Returns story arc object or None if not coherent.
        """
        if len(subgraph.nodes) < self.min_arc_length:
            return None

        # Get nodes with event data
        nodes = list(subgraph.nodes(data=True))

        # Extract events
        arc_events = [data['event'] for node_id, data in nodes]

        # Sort by temporal order
        arc_events_sorted = sorted(
            arc_events,
            key=lambda e: self._get_earliest_datetime(e) or self._get_latest_datetime(e) or 0
        )

        # Identify progression type
        progression_type = self._identify_progression_type(arc_events_sorted)

        # Extract narrative summary
        narrative_summary = self._extract_narrative_summary(
            arc_events_sorted,
            subgraph
        )

        # Calculate story confidence
        confidence = self._calculate_story_confidence(
            arc_events_sorted,
            subgraph
        )

        # Extract key dimensions
        key_dimensions = self._extract_key_dimensions(arc_events_sorted)

        return {
            'story_id': f"story_{hash(tuple(sorted([e.get('doc_id', e.get('chunk_id', '')) for e in arc_events]))) % 10000}",
            'length': len(arc_events),
            'events': arc_events_sorted,
            'progression_type': progression_type,
            'narrative_summary': narrative_summary,
            'key_dimensions': key_dimensions,
            'confidence': confidence,
            'temporal_span': self._calculate_temporal_span(arc_events_sorted),
            'spatial_span': self._calculate_spatial_span(arc_events_sorted)
        }

    def _identify_progression_type(
        self,
        events: List[Dict[str, Any]]
    ) -> str:
        """
        Identify the type of story progression.

        Types:
        - 'temporal': Events unfold over time at same location
        - 'spatial': Events spread geographically
        - 'thematic': Events evolve through different categories
        - 'mixed': Combination of above
        """
        # Check for spatial spread
        locations = set()
        for event in events:
            spatial_entities = event.get('dimensions', {}).get('spatial', [])
            for entity in spatial_entities:
                if entity.get('latitude') and entity.get('longitude'):
                    locations.add((entity['latitude'], entity['longitude']))

        has_spatial_spread = len(locations) > 1

        # Check for temporal progression
        datetimes = []
        for event in events:
            dt = self._get_earliest_datetime(event)
            if dt:
                datetimes.append(dt)

        has_temporal_progression = len(set(datetimes)) > 1 if datetimes else False

        # Check for thematic evolution
        categories = defaultdict(set)
        for event in events:
            for dim_name, dim_entities in event.get('dimensions', {}).items():
                if dim_name in ['temporal', 'spatial']:
                    continue
                for entity in dim_entities:
                    if 'category' in entity:
                        categories[dim_name].add(entity['category'])

        has_thematic_evolution = any(len(cats) > 1 for cats in categories.values())

        # Determine type
        types = []
        if has_temporal_progression:
            types.append('temporal')
        if has_spatial_spread:
            types.append('spatial')
        if has_thematic_evolution:
            types.append('thematic')

        if len(types) > 1:
            return 'mixed'
        elif types:
            return types[0]
        else:
            return 'static'

    def _extract_narrative_summary(
        self,
        events: List[Dict[str, Any]],
        subgraph: nx.DiGraph
    ) -> Dict[str, Any]:
        """
        Extract narrative summary of story arc.

        Returns key entities, progression, and highlights.
        """
        # Extract all entity mentions
        entity_mentions = defaultdict(lambda: defaultdict(int))

        for event in events:
            for dim_name, dim_entities in event.get('dimensions', {}).items():
                for entity in dim_entities:
                    text = entity.get('text', '')
                    if text:
                        entity_mentions[dim_name][text] += 1

        # Find most frequent entities per dimension
        key_entities = {}
        for dim_name, mentions in entity_mentions.items():
            # Top 3 entities per dimension
            top_entities = sorted(
                mentions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            key_entities[dim_name] = [e[0] for e in top_entities]

        # Extract progression
        progression = []
        for event in events:
            temporal = self._get_earliest_datetime(event)
            spatial_entities = event.get('dimensions', {}).get('spatial', [])
            location = spatial_entities[0].get('text', 'Unknown') if spatial_entities else 'Unknown'

            progression.append({
                'timestamp': temporal.isoformat() if temporal else None,
                'location': location,
                'document_title': event.get('document_title', 'Unknown')
            })

        return {
            'key_entities': key_entities,
            'progression': progression,
            'total_events': len(events)
        }

    def _calculate_story_confidence(
        self,
        events: List[Dict[str, Any]],
        subgraph: nx.DiGraph
    ) -> float:
        """
        Calculate confidence score for story arc.

        Based on:
        - Entity overlap between events
        - Temporal coherence
        - Graph connectivity
        """
        if len(events) < 2:
            return 0.0

        # Entity overlap score
        overlap_scores = []
        for i in range(len(events) - 1):
            entities_1 = self._extract_entity_set(events[i])
            entities_2 = self._extract_entity_set(events[i + 1])

            if entities_1 and entities_2:
                overlap = len(entities_1.intersection(entities_2)) / len(entities_1.union(entities_2))
                overlap_scores.append(overlap)

        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

        # Temporal coherence score
        datetimes = [self._get_earliest_datetime(e) for e in events if self._get_earliest_datetime(e)]
        temporal_coherence = 1.0

        if len(datetimes) > 1:
            gaps = [(datetimes[i+1] - datetimes[i]).total_seconds() / 86400 for i in range(len(datetimes) - 1)]
            max_gap = max(gaps) if gaps else 0
            temporal_coherence = max(0, 1 - (max_gap / (2 * self.max_temporal_gap_days)))

        # Graph connectivity score
        connectivity = nx.density(subgraph) if len(subgraph.nodes) > 1 else 1.0

        # Combined score
        confidence = (avg_overlap * 0.4 + temporal_coherence * 0.3 + connectivity * 0.3)

        return min(confidence, 1.0)

    def _extract_key_dimensions(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Extract key dimension values across story arc."""
        dimension_values = defaultdict(set)

        for event in events:
            for dim_name, dim_entities in event.get('dimensions', {}).items():
                for entity in dim_entities:
                    # Add text value
                    if entity.get('text'):
                        dimension_values[f"{dim_name}_text"].add(entity['text'])

                    # Add category value
                    if entity.get('category'):
                        dimension_values[f"{dim_name}_category"].add(entity['category'])

        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in dimension_values.items()}

    def _calculate_temporal_span(
        self,
        events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """Calculate temporal span of story arc."""
        datetimes = [self._get_earliest_datetime(e) for e in events if self._get_earliest_datetime(e)]

        if not datetimes:
            return None

        return {
            'start': min(datetimes).isoformat(),
            'end': max(datetimes).isoformat(),
            'duration_days': (max(datetimes) - min(datetimes)).total_seconds() / 86400
        }

    def _calculate_spatial_span(
        self,
        events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Calculate spatial span of story arc."""
        locations = []

        for event in events:
            spatial_entities = event.get('dimensions', {}).get('spatial', [])
            for entity in spatial_entities:
                if entity.get('latitude') and entity.get('longitude'):
                    locations.append({
                        'text': entity.get('text', ''),
                        'latitude': entity['latitude'],
                        'longitude': entity['longitude']
                    })

        if not locations:
            return None

        lats = [loc['latitude'] for loc in locations]
        lons = [loc['longitude'] for loc in locations]

        return {
            'locations': locations,
            'bbox': {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons)
            },
            'centroid': {
                'latitude': sum(lats) / len(lats),
                'longitude': sum(lons) / len(lons)
            }
        }

    def _extract_entity_set(
        self,
        event: Dict[str, Any],
        dimensions: Optional[List[str]] = None
    ) -> Set[str]:
        """Extract set of entity texts from event."""
        entities = set()

        for dim_name, dim_entities in event.get('dimensions', {}).items():
            if dimensions and dim_name not in dimensions:
                continue

            for entity in dim_entities:
                # Add text
                if entity.get('text'):
                    entities.add(f"{dim_name}:{entity['text']}")

                # Add category
                if entity.get('category'):
                    entities.add(f"{dim_name}:{entity['category']}")

        return entities

    def _get_earliest_datetime(self, event: Dict[str, Any]):
        """Get earliest datetime from event."""
        temporal_entities = event.get('dimensions', {}).get('temporal', [])
        datetimes = [e.get('datetime') for e in temporal_entities if e.get('datetime')]
        return min(datetimes) if datetimes else None

    def _get_latest_datetime(self, event: Dict[str, Any]):
        """Get latest datetime from event."""
        temporal_entities = event.get('dimensions', {}).get('temporal', [])
        datetimes = [e.get('datetime') for e in temporal_entities if e.get('datetime')]
        return max(datetimes) if datetimes else None
