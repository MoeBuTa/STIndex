"""
Generic visualization module for STIndex.

Provides visualization capabilities for extraction results:
- Interactive maps (Folium) for geocoded dimensions
- Statistical plots (matplotlib, plotly) for all dimensions
- Summary statistics
- HTML report generation
"""

from stindex.visualization.visualizer import STIndexVisualizer

__all__ = ["STIndexVisualizer"]
