"""
HTML report generator for extraction results.

Generates comprehensive HTML report combining:
- Statistical summary
- Interactive maps
- Statistical plots
- Dimensional analysis
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class HTMLReportGenerator:
    """Generate HTML reports from extraction results."""

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        plots: List[str],
        map_file: Optional[str],
        output_file: str
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            results: List of extraction results
            summary: Statistical summary
            plots: List of plot file paths
            map_file: Path to map HTML file
            output_file: Path to save main HTML report

        Returns:
            Path to generated HTML report
        """
        logger.info("Generating HTML report...")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create source directory for assets
        source_dir_name = output_path.stem + "_source"
        source_dir = output_path.parent / source_dir_name
        source_dir.mkdir(exist_ok=True)

        # Copy plots to source directory (if not already there)
        plot_files = []
        for plot_path in plots:
            plot_path_obj = Path(plot_path)
            if plot_path_obj.exists():
                dest = source_dir / plot_path_obj.name
                # Only copy if source and destination are different
                if plot_path_obj.resolve() != dest.resolve():
                    shutil.copy(plot_path, dest)
                plot_files.append(plot_path_obj.name)

        # Copy map to source directory (if not already there)
        map_filename = None
        if map_file and Path(map_file).exists():
            map_filename = "map.html"
            map_path = Path(map_file)
            dest = source_dir / map_filename
            # Only copy if source and destination are different
            if map_path.resolve() != dest.resolve():
                shutil.copy(map_file, dest)

        # Generate HTML
        html_content = self._generate_html(
            results=results,
            summary=summary,
            plot_files=plot_files,
            map_filename=map_filename,
            source_dir_name=source_dir_name
        )

        # Save HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✓ HTML report saved to: {output_path}")
        logger.info(f"  Source files in: {source_dir}")

        return str(output_path)

    def _generate_html(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        plot_files: List[str],
        map_filename: Optional[str],
        source_dir_name: str
    ) -> str:
        """Generate HTML content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STIndex Extraction Report</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>STIndex Extraction Report</h1>
            <p class="subtitle">Multi-Dimensional Information Extraction Analysis</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <!-- Table of Contents -->
        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#dimensions">Dimensional Analysis</a></li>
                <li><a href="#map">Interactive Map</a></li>
                <li><a href="#plots">Statistical Visualizations</a></li>
                <li><a href="#performance">Performance Metrics</a></li>
                <li><a href="#data">Data Sources</a></li>
            </ul>
        </nav>

        <!-- Overview Section -->
        <section id="overview">
            <h2>Overview</h2>
            {self._overview_section(summary)}
        </section>

        <!-- Dimensional Analysis -->
        <section id="dimensions">
            <h2>Dimensional Analysis</h2>
            {self._dimensions_section(summary)}
        </section>

        <!-- Interactive Map -->
        {self._map_section(map_filename, source_dir_name)}

        <!-- Statistical Visualizations -->
        <section id="plots">
            <h2>Statistical Visualizations</h2>
            {self._plots_section(plot_files, source_dir_name)}
        </section>

        <!-- Performance Metrics -->
        <section id="performance">
            <h2>Performance Metrics</h2>
            {self._performance_section(summary)}
        </section>

        <!-- Data Sources -->
        <section id="data">
            <h2>Data Sources</h2>
            {self._data_sources_section(summary)}
        </section>

        <!-- Footer -->
        <footer>
            <p>Generated by <strong>STIndex v0.4.0</strong></p>
            <p>Multi-Dimensional Spatiotemporal Information Extraction</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Get CSS styles for report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(to bottom, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }

        header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }

        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #764ba2;
            font-size: 1.2em;
            font-weight: 500;
        }

        .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }

        .toc {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }

        .toc h2 {
            color: #667eea;
            margin-bottom: 15px;
        }

        .toc ul {
            list-style: none;
        }

        .toc li {
            margin: 8px 0;
        }

        .toc a {
            color: #764ba2;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }

        .toc a:hover {
            color: #667eea;
        }

        section {
            margin-bottom: 50px;
        }

        h2 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }

        h3 {
            color: #764ba2;
            font-size: 1.5em;
            margin: 20px 0 15px 0;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card h3 {
            color: white;
            font-size: 1.1em;
            margin-bottom: 10px;
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .dimension-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .dimension-table th,
        .dimension-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        .dimension-table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }

        .dimension-table tr:hover {
            background: #f8f9fa;
        }

        .map-container {
            margin: 30px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .map-container iframe {
            width: 100%;
            height: 600px;
            border: none;
        }

        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }

        .plot-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .plot-card img {
            width: 100%;
            height: auto;
            display: block;
        }

        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }

        .alert-info {
            background: #e3f2fd;
            border-color: #2196f3;
            color: #1976d2;
        }

        .alert-warning {
            background: #fff3e0;
            border-color: #ff9800;
            color: #f57c00;
        }

        footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            color: #666;
        }

        footer strong {
            color: #667eea;
        }
        """

    def _overview_section(self, summary: Dict[str, Any]) -> str:
        """Generate overview section."""
        overview = summary.get('overview', {})

        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Chunks</h3>
                <div class="stat-value">{overview.get('total_chunks', 0)}</div>
                <div class="stat-label">Processed</div>
            </div>
            <div class="stat-card">
                <h3>Success Rate</h3>
                <div class="stat-value">{overview.get('success_rate', 0):.1f}%</div>
                <div class="stat-label">{overview.get('successful_extractions', 0)} successful</div>
            </div>
            <div class="stat-card">
                <h3>Failed</h3>
                <div class="stat-value">{overview.get('failed_extractions', 0)}</div>
                <div class="stat-label">Extractions</div>
            </div>
        </div>
        """

    def _dimensions_section(self, summary: Dict[str, Any]) -> str:
        """Generate dimensions section."""
        dimensions = summary.get('dimensions', {})

        if not dimensions:
            return '<p class="alert alert-warning">No dimensional data available.</p>'

        html = '<table class="dimension-table">'
        html += '''
        <thead>
            <tr>
                <th>Dimension</th>
                <th>Chunks with Entities</th>
                <th>Total Entities</th>
                <th>Unique Values</th>
            </tr>
        </thead>
        <tbody>
        '''

        for dim_name, stats in dimensions.items():
            html += f'''
            <tr>
                <td><strong>{dim_name.replace("_", " ").title()}</strong></td>
                <td>{stats.get('chunks_with_entities', 0)}</td>
                <td>{stats.get('total_entities', 0)}</td>
                <td>{stats.get('unique_count', 0)}</td>
            </tr>
            '''

        html += '</tbody></table>'
        return html

    def _map_section(self, map_filename: Optional[str], source_dir: str) -> str:
        """Generate map section."""
        if not map_filename:
            return '''
            <section id="map">
                <h2>Interactive Map</h2>
                <p class="alert alert-info">No geocoded spatial data available for map visualization.</p>
            </section>
            '''

        return f'''
        <section id="map">
            <h2>Interactive Map</h2>
            <div class="map-container">
                <iframe src="{source_dir}/{map_filename}"></iframe>
            </div>
        </section>
        '''

    def _plots_section(self, plot_files: List[str], source_dir: str) -> str:
        """Generate plots section."""
        if not plot_files:
            return '<p class="alert alert-info">No plots generated.</p>'

        # Separate static and interactive plots
        static_plots = [p for p in plot_files if p.endswith('.png')]
        interactive_plots = [p for p in plot_files if p.endswith('.html')]

        html = ''

        # Static plots
        if static_plots:
            html += '<h3>Statistical Charts</h3>'
            html += '<div class="plot-grid">'
            for plot in sorted(static_plots):
                html += f'''
                <div class="plot-card">
                    <img src="{source_dir}/{plot}" alt="{plot}">
                </div>
                '''
            html += '</div>'

        # Interactive plots
        if interactive_plots:
            html += '<h3>Interactive Visualizations</h3>'
            for plot in sorted(interactive_plots):
                html += f'''
                <div class="map-container" style="margin-top: 20px;">
                    <iframe src="{source_dir}/{plot}"></iframe>
                </div>
                '''

        return html

    def _performance_section(self, summary: Dict[str, Any]) -> str:
        """Generate performance section."""
        perf = summary.get('performance', {})

        return f'''
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Mean Time</h3>
                <div class="stat-value">{perf.get('mean_time', 0):.2f}s</div>
                <div class="stat-label">Per extraction</div>
            </div>
            <div class="stat-card">
                <h3>Total Time</h3>
                <div class="stat-value">{perf.get('total_time', 0):.1f}s</div>
                <div class="stat-label">All extractions</div>
            </div>
            <div class="stat-card">
                <h3>Min / Max</h3>
                <div class="stat-value" style="font-size: 1.5em;">
                    {perf.get('min_time', 0):.2f}s / {perf.get('max_time', 0):.2f}s
                </div>
                <div class="stat-label">Processing time range</div>
            </div>
        </div>
        '''

    def _data_sources_section(self, summary: Dict[str, Any]) -> str:
        """Generate data sources section."""
        sources = summary.get('sources', {})

        if not sources:
            return '<p class="alert alert-info">No source information available.</p>'

        html = '<table class="dimension-table">'
        html += '''
        <thead>
            <tr>
                <th>Data Source</th>
                <th>Chunks Processed</th>
            </tr>
        </thead>
        <tbody>
        '''

        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            html += f'''
            <tr>
                <td><strong>{source}</strong></td>
                <td>{count}</td>
            </tr>
            '''

        html += '</tbody></table>'
        return html
