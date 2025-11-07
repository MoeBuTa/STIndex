"""
Generic plot generator for dimensional data.

Creates statistical visualizations for all dimensions:
- Temporal distributions
- Spatial distributions
- Categorical distributions
- Cross-dimensional analysis
- Extraction metrics
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib not installed. Install with: pip install matplotlib seaborn")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("plotly not installed. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class PlotGenerator:
    """Generate statistical plots from extraction results."""

    def __init__(self):
        """Initialize plot generator."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plot generation")

        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def generate_plots(
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        dimensions: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate all statistical plots.

        Args:
            results: List of extraction results
            output_dir: Directory to save plots
            dimensions: List of dimensions to visualize (None = all)

        Returns:
            List of generated plot file paths
        """
        logger.info("Generating statistical plots...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to dataframe
        df = self._results_to_dataframe(results, dimensions)

        if df.empty:
            logger.warning("No data to visualize")
            return []

        generated_files = []

        # Generate plots for each dimension type
        for dim_name in df.columns:
            if dim_name in ['document_title', 'source', 'chunk_id']:
                continue

            # Check dimension type and generate appropriate plots
            if '_normalized' in dim_name or dim_name == 'temporal':
                # Temporal dimension
                plots = self._plot_temporal_dimension(df, dim_name, output_path)
                generated_files.extend(plots)

            elif '_latitude' in dim_name or '_longitude' in dim_name:
                # Skip coordinate columns (handled by map)
                continue

            elif '_text' in dim_name or dim_name == 'spatial':
                # Text dimension (spatial locations)
                plots = self._plot_text_dimension(df, dim_name, output_path)
                generated_files.extend(plots)

            elif '_category' in dim_name or dim_name in ['disease', 'event_type', 'venue_type']:
                # Categorical dimension
                plots = self._plot_categorical_dimension(df, dim_name, output_path)
                generated_files.extend(plots)

        # Generate cross-dimensional plots
        cross_plots = self._plot_cross_dimensional(df, output_path)
        generated_files.extend(cross_plots)

        # Generate extraction metrics
        metrics_plots = self._plot_extraction_metrics(results, output_path)
        generated_files.extend(metrics_plots)

        # Generate interactive plots if plotly available
        if PLOTLY_AVAILABLE:
            interactive_plots = self._create_interactive_plots(df, output_path)
            generated_files.extend(interactive_plots)

        logger.info(f"✓ Generated {len(generated_files)} plots")
        return generated_files

    def _results_to_dataframe(
        self,
        results: List[Dict[str, Any]],
        dimensions: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Convert extraction results to pandas DataFrame."""
        rows = []

        for result in results:
            if not result.get('extraction', {}).get('success'):
                continue

            entities = result['extraction'].get('entities', {})

            # Filter dimensions if specified
            if dimensions:
                entities = {k: v for k, v in entities.items() if k in dimensions}

            # Build row with all dimension data
            row = {
                'document_title': result.get('document_title', 'Unknown'),
                'source': result.get('source', 'Unknown'),
                'chunk_id': result.get('chunk_id', ''),
            }

            # Add each dimension's entities
            for dim_name, dim_entities in entities.items():
                if not dim_entities:
                    continue

                # Take first entity of each dimension (simplified)
                entity = dim_entities[0]

                # Add different fields based on entity type
                row[f'{dim_name}_text'] = entity.get('text', '')

                if 'normalized' in entity:
                    row[f'{dim_name}_normalized'] = entity.get('normalized')

                if 'latitude' in entity and 'longitude' in entity:
                    row[f'{dim_name}_latitude'] = entity.get('latitude')
                    row[f'{dim_name}_longitude'] = entity.get('longitude')

                if 'category' in entity:
                    row[f'{dim_name}_category'] = entity.get('category')

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Created dataframe with {len(df)} rows, {len(df.columns)} columns")
        return df

    def _plot_temporal_dimension(
        self,
        df: pd.DataFrame,
        dim_name: str,
        output_dir: Path
    ) -> List[str]:
        """Plot temporal distribution."""
        generated = []

        temporal_col = f'{dim_name}_normalized' if f'{dim_name}_normalized' in df.columns else dim_name

        if temporal_col not in df.columns:
            return generated

        temporal_df = df[df[temporal_col].notna()].copy()

        if temporal_df.empty:
            return generated

        # Parse dates
        dates = []
        for norm in temporal_df[temporal_col]:
            try:
                date_str = norm.split('/')[0] if '/' in str(norm) else str(norm)
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                dates.append(pd.to_datetime(date_str))
            except:
                pass

        if not dates:
            return generated

        # Create temporal distribution plot
        date_df = pd.DataFrame({'date': dates})
        date_df['year_month'] = date_df['date'].dt.to_period('M')
        monthly_counts = date_df['year_month'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_counts.plot(kind='bar', ax=ax, color='#667eea')
        ax.set_title(f'{dim_name.title()} Distribution Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = output_dir / f'{dim_name}_temporal_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        generated.append(str(output_file))
        logger.info(f"  ✓ {output_file.name}")

        return generated

    def _plot_text_dimension(
        self,
        df: pd.DataFrame,
        dim_name: str,
        output_dir: Path
    ) -> List[str]:
        """Plot distribution of text dimension (e.g., locations)."""
        generated = []

        text_col = f'{dim_name}_text' if f'{dim_name}_text' in df.columns else dim_name

        if text_col not in df.columns:
            return generated

        text_df = df[df[text_col].notna()]

        if text_df.empty:
            return generated

        # Get top values
        value_counts = text_df[text_col].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        value_counts.plot(kind='barh', ax=ax, color='#45B7D1')
        ax.set_title(f'Top 15 {dim_name.title()} by Count', fontsize=16, fontweight='bold')
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel(dim_name.title(), fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        output_file = output_dir / f'{dim_name}_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        generated.append(str(output_file))
        logger.info(f"  ✓ {output_file.name}")

        return generated

    def _plot_categorical_dimension(
        self,
        df: pd.DataFrame,
        dim_name: str,
        output_dir: Path
    ) -> List[str]:
        """Plot categorical dimension distribution."""
        generated = []

        cat_col = f'{dim_name}_category' if f'{dim_name}_category' in df.columns else dim_name

        if cat_col not in df.columns:
            return generated

        cat_df = df[df[cat_col].notna()]

        if cat_df.empty:
            return generated

        cat_counts = cat_df[cat_col].value_counts()

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        cat_counts.plot(kind='bar', ax=ax, color='#764ba2')
        ax.set_title(f'{dim_name.replace("_", " ").title()} Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel(dim_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Format labels
        labels = [label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = output_dir / f'{dim_name}_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        generated.append(str(output_file))
        logger.info(f"  ✓ {output_file.name}")

        return generated

    def _plot_cross_dimensional(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[str]:
        """Plot cross-dimensional analysis."""
        generated = []

        # Find categorical dimensions
        cat_dims = [col for col in df.columns if '_category' in col or col in ['disease', 'event_type', 'venue_type']]

        if len(cat_dims) >= 2:
            # Cross-tab of first two categorical dimensions
            dim1, dim2 = cat_dims[0], cat_dims[1]

            cross_df = df[[dim1, dim2]].dropna()

            if not cross_df.empty:
                cross_tab = pd.crosstab(cross_df[dim1], cross_df[dim2])

                fig, ax = plt.subplots(figsize=(14, 6))
                cross_tab.plot(kind='bar', ax=ax, stacked=False)
                ax.set_title(f'{dim1.replace("_", " ").title()} by {dim2.replace("_", " ").title()}',
                           fontsize=16, fontweight='bold')
                ax.set_xlabel(dim1.replace('_', ' ').title(), fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                ax.legend(title=dim2.replace('_', ' ').title())
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                output_file = output_dir / f'cross_{dim1}_{dim2}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

                generated.append(str(output_file))
                logger.info(f"  ✓ {output_file.name}")

        return generated

    def _plot_extraction_metrics(
        self,
        results: List[Dict[str, Any]],
        output_dir: Path
    ) -> List[str]:
        """Plot extraction performance metrics."""
        generated = []

        # Count entities by dimension
        dimension_stats = {}

        for result in results:
            if result.get('extraction', {}).get('success'):
                entities = result['extraction'].get('entities', {})
                for dim_name, dim_entities in entities.items():
                    if dim_name not in dimension_stats:
                        dimension_stats[dim_name] = {'total': 0, 'count': 0}
                    dimension_stats[dim_name]['total'] += 1
                    if dim_entities:
                        dimension_stats[dim_name]['count'] += len(dim_entities)

        if not dimension_stats:
            return generated

        dimensions = list(dimension_stats.keys())
        entity_counts = [dimension_stats[dim]['count'] for dim in dimensions]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(dimensions, entity_counts, color='#667eea')
        ax.set_title('Entity Count by Dimension', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Entities', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for i, v in enumerate(entity_counts):
            ax.text(i, v + max(entity_counts) * 0.01, str(v), ha='center', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        output_file = output_dir / 'extraction_metrics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        generated.append(str(output_file))
        logger.info(f"  ✓ {output_file.name}")

        return generated

    def _create_interactive_plots(
        self,
        df: pd.DataFrame,
        output_dir: Path
    ) -> List[str]:
        """Create interactive Plotly visualizations."""
        generated = []

        # Find temporal dimension
        temporal_cols = [col for col in df.columns if '_normalized' in col or col == 'temporal']

        if not temporal_cols:
            return generated

        temporal_col = temporal_cols[0]
        temporal_df = df[df[temporal_col].notna()].copy()

        if temporal_df.empty:
            return generated

        # Parse dates
        dates = []
        for norm in temporal_df[temporal_col]:
            try:
                date_str = str(norm).split('/')[0] if '/' in str(norm) else str(norm)
                dates.append(pd.to_datetime(date_str))
            except:
                pass

        if not dates:
            return generated

        # Create interactive timeline
        plot_df = pd.DataFrame({'date': dates})
        plot_df['count'] = 1
        plot_df['cumulative'] = plot_df['count'].cumsum()

        fig = px.line(plot_df, x='date', y='cumulative',
                     title='Cumulative Events Over Time',
                     labels={'cumulative': 'Cumulative Count', 'date': 'Date'},
                     markers=True)

        fig.update_layout(
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12),
            title_font=dict(size=18),
            height=500
        )

        output_file = output_dir / 'interactive_timeline.html'
        fig.write_html(str(output_file))

        generated.append(str(output_file))
        logger.info(f"  ✓ {output_file.name}")

        return generated
