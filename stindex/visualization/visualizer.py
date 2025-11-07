"""
Main visualization orchestrator for STIndex.

Coordinates all visualization components to generate comprehensive
analysis reports from extraction results.
"""

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from stindex.visualization.html_report import HTMLReportGenerator
from stindex.visualization.map_generator import MapGenerator
from stindex.visualization.plot_generator import PlotGenerator
from stindex.visualization.statistical_summary import StatisticalSummary


class STIndexVisualizer:
    """
    Main visualizer for STIndex extraction results.

    Generates comprehensive HTML reports with:
    - Interactive maps (geocoded spatial data)
    - Statistical plots (all dimensions)
    - Summary statistics
    - Cross-dimensional analysis

    Usage:
        visualizer = STIndexVisualizer()
        visualizer.visualize(
            results_file="data/results/extraction_results.json",
            output_dir="data/visualizations"
        )
    """

    def __init__(
        self,
        temporal_dim: str = "temporal",
        spatial_dim: str = "spatial",
        category_dim: Optional[str] = None
    ):
        """
        Initialize visualizer.

        Args:
            temporal_dim: Name of temporal dimension for timeline
            spatial_dim: Name of spatial dimension for mapping
            category_dim: Name of categorical dimension for color coding
        """
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.category_dim = category_dim

        # Initialize components
        try:
            self.map_generator = MapGenerator()
        except ImportError:
            logger.warning("Map generation not available (folium not installed)")
            self.map_generator = None

        try:
            self.plot_generator = PlotGenerator()
        except ImportError:
            logger.warning("Plot generation not available (matplotlib not installed)")
            self.plot_generator = None

        self.summary_generator = StatisticalSummary()
        self.report_generator = HTMLReportGenerator()

    def visualize(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        results_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        animated_map: bool = True,
        dimensions: Optional[List[str]] = None
    ) -> str:
        """
        Generate comprehensive visualization report.

        Automatically creates a zip archive containing the HTML report and all source files.

        Args:
            results: Extraction results (list of dicts)
            results_file: Path to results JSON file (if results not provided)
            output_dir: Output directory for report
            animated_map: Create animated timeline map if True
            dimensions: List of dimensions to visualize (None = all)

        Returns:
            Path to generated zip file
        """
        logger.info("=" * 80)
        logger.info("STIndex Visualization")
        logger.info("=" * 80)

        # Load results
        if results is None:
            if results_file is None:
                raise ValueError("Either results or results_file must be provided")
            results = self._load_results(results_file)

        if not results:
            logger.error("No results to visualize")
            return None

        # Setup output directory
        if output_dir is None:
            output_dir = "data/visualizations"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"stindex_report_{timestamp}.html"
        source_dir_name = f"stindex_report_{timestamp}_source"

        # Create source directory
        source_dir = output_path / source_dir_name
        source_dir.mkdir(exist_ok=True)

        # Step 1: Generate statistical summary
        logger.info("\n[1/5] Generating statistical summary...")
        summary = self.summary_generator.generate_summary(results)

        # Save summary JSON
        summary_file = source_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Step 2: Generate plots
        plot_files = []
        if self.plot_generator:
            logger.info("\n[2/5] Generating statistical plots...")
            try:
                plot_files = self.plot_generator.generate_plots(
                    results=results,
                    output_dir=str(source_dir),
                    dimensions=dimensions
                )
            except Exception as e:
                logger.error(f"Plot generation failed: {e}")

        # Step 3: Generate map
        map_file = None
        if self.map_generator:
            logger.info("\n[3/5] Generating interactive map...")
            try:
                map_file = self.map_generator.generate_map(
                    results=results,
                    output_file=str(source_dir / "map.html"),
                    temporal_dim=self.temporal_dim,
                    spatial_dim=self.spatial_dim,
                    category_dim=self.category_dim,
                    animated=animated_map
                )
            except Exception as e:
                logger.error(f"Map generation failed: {e}")

        # Step 4: Generate HTML report
        logger.info("\n[4/5] Generating HTML report...")
        report_path = output_path / report_name

        try:
            report_file = self.report_generator.generate_report(
                results=results,
                summary=summary,
                plots=plot_files,
                map_file=map_file,
                output_file=str(report_path)
            )
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            raise

        # Step 5: Create zip file
        logger.info("\n[5/5] Creating zip archive...")
        try:
            zip_path = self._create_zip_archive(
                report_path=report_path,
                source_dir=source_dir,
                output_dir=output_path
            )
        except Exception as e:
            logger.error(f"Zip creation failed: {e}")
            # Fall back to returning HTML path if zip fails
            zip_path = str(report_path)

        logger.info("\n" + "=" * 80)
        logger.info("Visualization Complete!")
        logger.info("=" * 80)
        logger.info(f"\n✓ HTML Report: {report_path}")
        logger.info(f"  Source files: {source_dir}")
        logger.info(f"📦 Zip Archive: {zip_path}")
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total chunks: {summary['overview']['total_chunks']}")
        logger.info(f"  - Success rate: {summary['overview']['success_rate']:.1f}%")
        logger.info(f"  - Dimensions: {len(summary['dimensions'])}")
        logger.info(f"  - Plots generated: {len(plot_files)}")
        logger.info(f"  - Map: {'Yes' if map_file else 'No'}")

        return str(zip_path)

    def visualize_from_file(
        self,
        results_file: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convenience method to visualize from results file.

        Args:
            results_file: Path to extraction results JSON
            output_dir: Output directory for report
            **kwargs: Additional arguments for visualize()

        Returns:
            Path to generated HTML report
        """
        return self.visualize(
            results_file=results_file,
            output_dir=output_dir,
            **kwargs
        )

    def _create_zip_archive(
        self,
        report_path: Path,
        source_dir: Path,
        output_dir: Path
    ) -> str:
        """
        Create zip archive containing HTML report and source files.

        Args:
            report_path: Path to HTML report file
            source_dir: Path to source directory with assets
            output_dir: Output directory for zip file

        Returns:
            Path to created zip file
        """
        # Create zip file name (same as HTML report, but with .zip extension)
        zip_name = report_path.stem + ".zip"
        zip_path = output_dir / zip_name

        logger.info(f"Creating zip archive: {zip_path}")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add HTML report
            zipf.write(report_path, report_path.name)
            logger.debug(f"  Added: {report_path.name}")

            # Add all files from source directory
            file_count = 0
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    # Create relative path in zip (preserve directory structure)
                    arcname = source_dir.name / file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)
                    file_count += 1
                    logger.debug(f"  Added: {arcname}")

            logger.info(f"✓ Zip archive created with {file_count + 1} files")

        return str(zip_path)

    def _load_results(self, results_file: str) -> List[Dict[str, Any]]:
        """Load extraction results from JSON file."""
        logger.info(f"Loading results from: {results_file}")

        results_path = Path(results_file)

        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(results_path, 'r') as f:
            results = json.load(f)

        logger.info(f"✓ Loaded {len(results)} results")
        return results
