"""
Statistical summary generator for extraction results.

Generates comprehensive statistics:
- Overall success rates
- Dimension-wise entity counts
- Data source statistics
- Processing time metrics
"""

from typing import Any, Dict, List
from collections import Counter
import pandas as pd
from loguru import logger


class StatisticalSummary:
    """Generate statistical summaries from extraction results."""

    def __init__(self):
        """Initialize summary generator."""
        pass

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.

        Args:
            results: List of extraction results

        Returns:
            Dictionary containing summary statistics
        """
        logger.info("Generating statistical summary...")

        summary = {
            'overview': self._overview_stats(results),
            'dimensions': self._dimension_stats(results),
            'sources': self._source_stats(results),
            'performance': self._performance_stats(results),
            'temporal_coverage': self._temporal_coverage(results),
            'spatial_coverage': self._spatial_coverage(results),
        }

        logger.info("✓ Statistical summary generated")
        return summary

    def _overview_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overview statistics."""
        total_chunks = len(results)
        successful = sum(1 for r in results if r.get('extraction', {}).get('success'))
        failed = total_chunks - successful

        return {
            'total_chunks': total_chunks,
            'successful_extractions': successful,
            'failed_extractions': failed,
            'success_rate': (successful / total_chunks * 100) if total_chunks > 0 else 0,
        }

    def _dimension_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Generate dimension-wise statistics."""
        dimension_stats = {}

        for result in results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            for dim_name, dim_entities in entities.items():
                if dim_name not in dimension_stats:
                    dimension_stats[dim_name] = {
                        'chunks_with_entities': 0,
                        'total_entities': 0,
                        'unique_values': set()
                    }

                if dim_entities:
                    dimension_stats[dim_name]['chunks_with_entities'] += 1
                    dimension_stats[dim_name]['total_entities'] += len(dim_entities)

                    # Track unique values
                    for entity in dim_entities:
                        text = entity.get('text') or entity.get('category') or entity.get('normalized')
                        if text:
                            dimension_stats[dim_name]['unique_values'].add(text)

        # Convert sets to counts
        for dim_name in dimension_stats:
            dimension_stats[dim_name]['unique_count'] = len(dimension_stats[dim_name]['unique_values'])
            del dimension_stats[dim_name]['unique_values']  # Remove set for JSON serialization

        return dimension_stats

    def _source_stats(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate data source statistics."""
        sources = [r.get('source', 'Unknown') for r in results]
        source_counts = Counter(sources)

        return dict(source_counts)

    def _performance_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate performance statistics."""
        processing_times = [
            r['extraction'].get('processing_time', 0)
            for r in results
            if r.get('extraction', {}).get('success')
        ]

        if not processing_times:
            return {
                'mean_time': 0,
                'median_time': 0,
                'min_time': 0,
                'max_time': 0,
                'total_time': 0
            }

        return {
            'mean_time': sum(processing_times) / len(processing_times),
            'median_time': sorted(processing_times)[len(processing_times) // 2],
            'min_time': min(processing_times),
            'max_time': max(processing_times),
            'total_time': sum(processing_times)
        }

    def _temporal_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate temporal coverage statistics."""
        temporal_entities = []

        for result in results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Find temporal dimension (may have different names)
            for dim_name in ['temporal', 'time', 'date']:
                if dim_name in entities:
                    temporal_entities.extend(entities[dim_name])

        if not temporal_entities:
            return {
                'has_temporal_data': False,
                'count': 0
            }

        # Extract dates
        dates = []
        for entity in temporal_entities:
            normalized = entity.get('normalized', '')
            if normalized:
                try:
                    # Handle intervals
                    date_str = normalized.split('/')[0] if '/' in normalized else normalized
                    # Extract date part
                    if 'T' in date_str:
                        date_str = date_str.split('T')[0]
                    dates.append(pd.to_datetime(date_str))
                except:
                    pass

        if not dates:
            return {
                'has_temporal_data': True,
                'count': len(temporal_entities),
                'parsed_dates': 0
            }

        return {
            'has_temporal_data': True,
            'count': len(temporal_entities),
            'parsed_dates': len(dates),
            'earliest_date': min(dates).isoformat(),
            'latest_date': max(dates).isoformat(),
            'date_range_days': (max(dates) - min(dates)).days
        }

    def _spatial_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate spatial coverage statistics."""
        spatial_entities = []

        for result in results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Find spatial dimension
            for dim_name in ['spatial', 'location', 'place']:
                if dim_name in entities:
                    spatial_entities.extend(entities[dim_name])

        if not spatial_entities:
            return {
                'has_spatial_data': False,
                'count': 0
            }

        # Count geocoded entities
        geocoded = sum(1 for e in spatial_entities if e.get('latitude') and e.get('longitude'))

        # Get coordinate bounds if geocoded
        coords = [(e['latitude'], e['longitude']) for e in spatial_entities if e.get('latitude') and e.get('longitude')]

        if coords:
            lats, lons = zip(*coords)
            return {
                'has_spatial_data': True,
                'count': len(spatial_entities),
                'geocoded_count': geocoded,
                'geocoding_rate': (geocoded / len(spatial_entities) * 100),
                'bounds': {
                    'min_lat': min(lats),
                    'max_lat': max(lats),
                    'min_lon': min(lons),
                    'max_lon': max(lons)
                }
            }

        return {
            'has_spatial_data': True,
            'count': len(spatial_entities),
            'geocoded_count': geocoded,
            'geocoding_rate': (geocoded / len(spatial_entities) * 100) if spatial_entities else 0
        }
