"""
Generic multi-dimensional analysis for any configured dimensions.

Provides statistical analysis and insights for:
- Temporal patterns (trends, seasonality, peaks)
- Spatial patterns (hotspots, dispersion, coverage)
- Categorical distributions (frequencies, co-occurrences)
- Cross-dimensional correlations
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class DimensionAnalyzer:
    """
    Analyze extraction results across any configured dimensions.

    Fixed dimensions (built-in):
    - temporal: ISO 8601 normalized dates
    - spatial: Geocoded locations with coordinates

    Custom dimensions (auto-detected from config YAML):
    - Any dimensions defined in dimensions.yml or custom config
    - Automatically detects type: categorical, free_text, or structured

    Example:
        analyzer = DimensionAnalyzer()

        # Analyze all dimensions found in results (auto-detected)
        analysis = analyzer.analyze(extraction_results=results)

        # Or filter to specific dimensions from your config
        analysis = analyzer.analyze(
            extraction_results=results,
            dimensions=['temporal', 'spatial']  # Only analyze these
        )

        # Access results by dimension name from your config
        for dim_name, dim_stats in analysis.items():
            if dim_name in ['global', 'cross_dimensional']:
                continue
            print(f"{dim_name}: {dim_stats}")
    """

    def analyze(
        self,
        extraction_results: List[Dict[str, Any]],
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze extraction results across all dimensions.

        Args:
            extraction_results: List of extraction results from pipeline
            dimensions: Dimensions to analyze (None = all discovered dimensions)

        Returns:
            Comprehensive analysis dict with dimension-specific statistics
        """
        logger.info("Analyzing multi-dimensional extraction results...")

        # Extract all dimension data
        dimension_data = self._extract_dimension_data(extraction_results, dimensions)

        if not dimension_data:
            logger.warning("No dimension data found in extraction results")
            return {}

        analysis = {}

        # Analyze each dimension based on its data structure
        for dim_name, dim_entities in dimension_data.items():
            logger.info(f"  Analyzing {dim_name}: {len(dim_entities)} entities")

            # Only temporal and spatial are treated specially (fixed dimensions)
            if dim_name == 'temporal':
                analysis[dim_name] = self._analyze_temporal(dim_entities)
            elif dim_name == 'spatial':
                analysis[dim_name] = self._analyze_spatial(dim_entities)
            else:
                # Auto-detect dimension type from data structure
                analysis[dim_name] = self._analyze_custom_dimension(dim_entities, dim_name)

        # Cross-dimensional analysis
        analysis['cross_dimensional'] = self._analyze_cross_dimensional(
            dimension_data
        )

        # Global statistics
        analysis['global'] = self._calculate_global_stats(
            extraction_results,
            dimension_data
        )

        logger.info(f"✓ Analysis complete for {len(analysis)} dimension groups")

        return analysis

    def _extract_dimension_data(
        self,
        extraction_results: List[Dict[str, Any]],
        dimensions: Optional[List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all entities grouped by dimension."""
        dimension_data = defaultdict(list)

        for result in extraction_results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Filter dimensions if specified
            if dimensions:
                entities = {k: v for k, v in entities.items() if k in dimensions}

            # Collect entities
            for dim_name, dim_entities in entities.items():
                for entity in dim_entities:
                    # Add context
                    entity_with_context = entity.copy()
                    entity_with_context.update({
                        'document_id': result.get('document_id'),
                        'document_title': result.get('document_title'),
                        'chunk_id': result.get('chunk_id'),
                        'source': result.get('source')
                    })
                    dimension_data[dim_name].append(entity_with_context)

        return dict(dimension_data)

    def _analyze_temporal(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal dimension."""
        import pandas as pd

        # Parse dates
        dates = []
        failed_parses = 0

        for entity in entities:
            normalized = entity.get('normalized', '')
            try:
                # Handle intervals
                if '/' in normalized:
                    normalized = normalized.split('/')[0]

                # Handle date-only
                if 'T' not in normalized:
                    normalized = f"{normalized}T00:00:00"

                date = pd.to_datetime(normalized, errors='raise')
                dates.append({
                    'datetime': date,
                    'text': entity.get('text', ''),
                    'type': entity.get('temporal_type', 'unknown'),
                    'source': entity.get('source', 'unknown')
                })
            except:
                failed_parses += 1

        if failed_parses > 0:
            logger.warning(f"Failed to parse {failed_parses} temporal values")

        if not dates:
            return {'error': 'No valid temporal data'}

        # Create DataFrame
        df = pd.DataFrame(dates)
        df['date'] = df['datetime'].dt.date
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek

        # Calculate statistics
        return {
            'total_mentions': len(dates),
            'unique_dates': df['date'].nunique(),
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat(),
                'span_days': (df['datetime'].max() - df['datetime'].min()).days
            },
            'distribution': {
                'by_year': df['year'].value_counts().to_dict(),
                'by_month': df['month'].value_counts().to_dict(),
                'by_day_of_week': df['day_of_week'].value_counts().to_dict()
            },
            'temporal_types': df['type'].value_counts().to_dict(),
            'by_source': df.groupby('source')['datetime'].count().to_dict(),
            'timeline': df.groupby('date').size().to_dict()  # For timeline visualization
        }

    def _analyze_spatial(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze spatial dimension."""
        # Filter valid coordinates
        locations = []
        failed_geocodes = 0

        for entity in entities:
            lat = entity.get('latitude')
            lon = entity.get('longitude')

            if lat and lon:
                locations.append({
                    'text': entity.get('text', ''),
                    'latitude': lat,
                    'longitude': lon,
                    'location_type': entity.get('location_type', 'unknown'),
                    'source': entity.get('source', 'unknown')
                })
            else:
                failed_geocodes += 1

        if failed_geocodes > 0:
            logger.warning(f"Failed to geocode {failed_geocodes} locations")

        if not locations:
            return {'error': 'No valid spatial data'}

        # Calculate statistics
        lats = [loc['latitude'] for loc in locations]
        lons = [loc['longitude'] for loc in locations]

        # Location frequency
        location_counts = defaultdict(int)
        for loc in locations:
            location_counts[loc['text']] += 1

        # Location types
        type_counts = defaultdict(int)
        for loc in locations:
            type_counts[loc['location_type']] += 1

        return {
            'total_mentions': len(locations),
            'unique_locations': len(set(loc['text'] for loc in locations)),
            'geocoding_success_rate': len(locations) / (len(locations) + failed_geocodes),
            'bounding_box': {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons)
            },
            'centroid': {
                'latitude': np.mean(lats),
                'longitude': np.mean(lons)
            },
            'top_locations': dict(sorted(
                location_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),
            'location_types': dict(type_counts),
            'all_locations': locations  # For map visualization
        }

    def _analyze_custom_dimension(
        self,
        entities: List[Dict[str, Any]],
        dimension_name: str
    ) -> Dict[str, Any]:
        """
        Analyze any custom dimension by auto-detecting its type from data structure.

        Detects:
        - Categorical: Has 'category' field
        - Free text: Only has 'text' field
        - Structured: Has multiple custom fields
        """
        if not entities:
            return {'error': 'No entities for analysis'}

        # Auto-detect dimension type from first entity
        sample_entity = entities[0]
        has_category = 'category' in sample_entity
        has_text = 'text' in sample_entity

        # Extract data based on detected type
        if has_category:
            # Categorical dimension
            categories = defaultdict(int)
            text_counts = defaultdict(int)
            by_source = defaultdict(lambda: defaultdict(int))

            for entity in entities:
                category = entity.get('category', 'unknown')
                text = entity.get('text', '')
                source = entity.get('source', 'unknown')

                categories[category] += 1
                if text:
                    text_counts[text] += 1
                by_source[source][category] += 1

            return {
                'dimension_type': 'categorical',
                'total_mentions': len(entities),
                'unique_categories': len(categories),
                'category_distribution': dict(sorted(
                    categories.items(),
                    key=lambda x: x[1],
                    reverse=True
                )),
                'top_texts': dict(sorted(
                    text_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),
                'by_source': {
                    source: dict(cats) for source, cats in by_source.items()
                }
            }
        elif has_text:
            # Free text dimension
            text_counts = defaultdict(int)
            by_source = defaultdict(int)

            for entity in entities:
                text = entity.get('text', '')
                source = entity.get('source', 'unknown')

                if text:
                    text_counts[text] += 1
                by_source[source] += 1

            return {
                'dimension_type': 'free_text',
                'total_mentions': len(entities),
                'unique_values': len(text_counts),
                'top_values': dict(sorted(
                    text_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20]),
                'by_source': dict(by_source)
            }
        else:
            # Structured dimension (custom fields)
            return {
                'dimension_type': 'structured',
                'total_mentions': len(entities),
                'sample_entity': sample_entity,
                'note': 'Structured dimensions require custom analysis'
            }

    def _analyze_cross_dimensional(
        self,
        dimension_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze relationships between dimensions."""
        cross_analysis = {}

        # Find co-occurrences
        dimension_names = list(dimension_data.keys())

        for i, dim1 in enumerate(dimension_names):
            for dim2 in dimension_names[i+1:]:
                # Group by document
                co_occurrences = defaultdict(lambda: defaultdict(int))

                # Index dim1 by document
                dim1_by_doc = defaultdict(set)
                for entity in dimension_data[dim1]:
                    doc_id = entity.get('chunk_id', entity.get('document_id'))
                    if doc_id:
                        value = entity.get('category') or entity.get('text')
                        if value:
                            dim1_by_doc[doc_id].add(value)

                # Index dim2 by document
                dim2_by_doc = defaultdict(set)
                for entity in dimension_data[dim2]:
                    doc_id = entity.get('chunk_id', entity.get('document_id'))
                    if doc_id:
                        value = entity.get('category') or entity.get('text')
                        if value:
                            dim2_by_doc[doc_id].add(value)

                # Find co-occurrences
                for doc_id in set(dim1_by_doc.keys()).intersection(dim2_by_doc.keys()):
                    for val1 in dim1_by_doc[doc_id]:
                        for val2 in dim2_by_doc[doc_id]:
                            co_occurrences[val1][val2] += 1

                # Store top co-occurrences
                if co_occurrences:
                    top_pairs = []
                    for val1, val2_counts in co_occurrences.items():
                        for val2, count in val2_counts.items():
                            top_pairs.append((val1, val2, count))

                    top_pairs.sort(key=lambda x: x[2], reverse=True)

                    cross_analysis[f'{dim1}_x_{dim2}'] = {
                        'total_pairs': len(top_pairs),
                        'top_co_occurrences': [
                            {dim1: p[0], dim2: p[1], 'count': p[2]}
                            for p in top_pairs[:20]
                        ]
                    }

        return cross_analysis

    def _calculate_global_stats(
        self,
        extraction_results: List[Dict[str, Any]],
        dimension_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate global statistics."""
        # Count successes/failures
        successful = sum(1 for r in extraction_results if r.get('extraction', {}).get('success'))
        failed = len(extraction_results) - successful

        # Entity counts by dimension
        entity_counts = {
            dim: len(entities) for dim, entities in dimension_data.items()
        }

        # Source distribution
        sources = defaultdict(int)
        for result in extraction_results:
            source = result.get('source', 'unknown')
            sources[source] += 1

        return {
            'total_chunks': len(extraction_results),
            'successful_extractions': successful,
            'failed_extractions': failed,
            'success_rate': successful / len(extraction_results) if extraction_results else 0,
            'entities_by_dimension': entity_counts,
            'total_entities': sum(entity_counts.values()),
            'chunks_by_source': dict(sources)
        }
