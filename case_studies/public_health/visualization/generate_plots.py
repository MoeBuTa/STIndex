"""
Generate statistical plots and visualizations from extraction results.

Creates comprehensive visualizations for the WWW demo paper including:
- Temporal analysis (events over time)
- Spatial analysis (events by location)
- Categorical analysis (disease, venue, event types)
- Cross-dimensional analysis
- Performance metrics
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger


# Data source mappings with URLs
SOURCE_MAPPING = {
    'wa_health_au': {
        'display_name': 'WA Health AU',
        'url': 'health.wa.gov.au',
        'full_url': 'https://www.health.wa.gov.au/~/media/Files/Corporate/general-documents/Medical-notifications/PDF/Measles-quick-guide-for-primary-healthcare-workers.pdf'
    },
    'wa_doh_us': {
        'display_name': 'WA DOH US',
        'url': 'doh.wa.gov',
        'full_url': 'https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025'
    },
    'immunisation_coalition_au': {
        'display_name': 'AU Flu Stats',
        'url': 'immunisationcoalition.org.au',
        'full_url': 'https://immunisationcoalition.org.au/influenza-statistics/'
    }
}

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """Load extraction results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} extraction results")
    return results


def extract_data_for_analysis(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract and flatten data for analysis."""
    rows = []

    for result in results:
        if not result.get('extraction', {}).get('success'):
            continue

        entities = result['extraction'].get('entities', {})

        # Get entity data
        temporal_entities = entities.get('temporal', [])
        spatial_entities = entities.get('spatial', [])
        event_types = entities.get('event_type', [])
        venue_types = entities.get('venue_type', [])
        diseases = entities.get('disease', [])

        # Create row for each event
        for i in range(max(len(temporal_entities), len(spatial_entities), 1)):
            # Map source name
            raw_source = result.get('source', 'Unknown')
            source_display = SOURCE_MAPPING.get(raw_source, {}).get('display_name', raw_source)
            source_url = SOURCE_MAPPING.get(raw_source, {}).get('url', '')

            row = {
                'document_title': result.get('document_title', 'Unknown'),
                'source': source_display,
                'source_url': source_url,
                'chunk_id': result.get('chunk_id', ''),
                # Temporal
                'temporal_text': temporal_entities[i]['text'] if i < len(temporal_entities) else None,
                'temporal_normalized': temporal_entities[i]['normalized'] if i < len(temporal_entities) else None,
                # Spatial
                'spatial_text': spatial_entities[i]['text'] if i < len(spatial_entities) else None,
                'spatial_latitude': spatial_entities[i].get('latitude') if i < len(spatial_entities) else None,
                'spatial_longitude': spatial_entities[i].get('longitude') if i < len(spatial_entities) else None,
                # Categorical
                'event_type': event_types[0]['category'] if event_types else None,
                'venue_type': venue_types[0]['category'] if venue_types else None,
                'disease': diseases[0]['category'] if diseases else None,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Created dataframe with {len(df)} rows")
    return df


def plot_temporal_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot temporal distribution of events."""
    # Parse temporal data
    temporal_df = df[df['temporal_normalized'].notna()].copy()

    # Extract dates from normalized temporal expressions
    dates = []
    for norm in temporal_df['temporal_normalized']:
        try:
            # Handle intervals (take start date)
            date_str = norm.split('/')[0] if '/' in norm else norm
            # Extract just the date part
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            dates.append(pd.to_datetime(date_str))
        except:
            pass

    if not dates:
        logger.warning("No valid dates found for temporal distribution")
        return

    temporal_df = pd.DataFrame({'date': dates})
    temporal_df['year_month'] = temporal_df['date'].dt.to_period('M')

    # Count events by month
    monthly_counts = temporal_df['year_month'].value_counts().sort_index()

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_counts.plot(kind='bar', ax=ax, color='#667eea')
    ax.set_title('Health Events Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved temporal distribution plot")


def plot_disease_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot disease distribution."""
    disease_counts = df[df['disease'].notna()]['disease'].value_counts()

    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    ax1.pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 10})
    ax1.set_title('Disease Distribution (Pie Chart)', fontsize=14, fontweight='bold')

    # Bar chart
    disease_counts.plot(kind='barh', ax=ax2, color='#764ba2')
    ax2.set_title('Disease Distribution (Count)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Events', fontsize=12)
    ax2.set_ylabel('Disease', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved disease distribution plot")


def plot_venue_type_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot venue type distribution."""
    venue_counts = df[df['venue_type'].notna()]['venue_type'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    venue_counts.plot(kind='bar', ax=ax, color='#667eea')
    ax.set_title('Events by Venue Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Venue Type', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'venue_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved venue type distribution plot")


def plot_event_type_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot event type distribution."""
    event_counts = df[df['event_type'].notna()]['event_type'].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    event_counts.plot(kind='barh', ax=ax, color='#764ba2')
    ax.set_title('Events by Event Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Event Type', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

    # Format labels (replace underscores)
    labels = [label.get_text().replace('_', ' ').title() for label in ax.get_yticklabels()]
    ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(output_dir / 'event_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved event type distribution plot")


def plot_spatial_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot spatial distribution by location."""
    spatial_counts = df[df['spatial_text'].notna()]['spatial_text'].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    spatial_counts.plot(kind='barh', ax=ax, color='#45B7D1')
    ax.set_title('Top 15 Locations by Event Count', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Location', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved spatial distribution plot")


def plot_disease_by_source(df: pd.DataFrame, output_dir: Path):
    """Plot disease distribution by data source."""
    cross_tab = pd.crosstab(df['disease'], df['source'])

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = cross_tab.plot(kind='bar', ax=ax, stacked=False)
    ax.set_title('Disease Distribution by Data Source', fontsize=16, fontweight='bold')
    ax.set_xlabel('Disease', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)

    # Create legend labels with URLs
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        # Find the source URL from the dataframe
        matching_rows = df[df['source'] == label]
        if len(matching_rows) > 0 and 'source_url' in matching_rows.columns:
            url = matching_rows.iloc[0]['source_url']
            if url:
                new_labels.append(f"{label}\n({url})")
            else:
                new_labels.append(label)
        else:
            new_labels.append(label)

    ax.legend(handles, new_labels, title='Data Source', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'disease_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved disease by source plot")


def plot_extraction_metrics(results: List[Dict[str, Any]], output_dir: Path):
    """Plot extraction performance metrics."""
    # Count successes by dimension
    dimension_stats = {
        'temporal': {'total': 0, 'success': 0},
        'spatial': {'total': 0, 'success': 0},
        'event_type': {'total': 0, 'success': 0},
        'venue_type': {'total': 0, 'success': 0},
        'disease': {'total': 0, 'success': 0},
    }

    for result in results:
        if result.get('extraction', {}).get('success'):
            entities = result['extraction'].get('entities', {})
            for dim in dimension_stats.keys():
                dimension_stats[dim]['total'] += 1
                if entities.get(dim):
                    dimension_stats[dim]['success'] += 1

    # Calculate success rates
    dimensions = list(dimension_stats.keys())
    success_rates = [
        (dimension_stats[dim]['success'] / dimension_stats[dim]['total'] * 100)
        if dimension_stats[dim]['total'] > 0 else 0
        for dim in dimensions
    ]
    entity_counts = [dimension_stats[dim]['success'] for dim in dimensions]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Success rates
    ax1.bar(dimensions, success_rates, color='#4caf50')
    ax1.set_title('Extraction Success Rate by Dimension', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Entity counts
    ax2.bar(dimensions, entity_counts, color='#667eea')
    ax2.set_title('Entity Count by Dimension', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Entities', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(entity_counts):
        ax2.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'extraction_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved extraction metrics plot")


def plot_geocoding_success(df: pd.DataFrame, output_dir: Path):
    """Plot geocoding success rate."""
    total_spatial = df['spatial_text'].notna().sum()
    geocoded = df['spatial_latitude'].notna().sum()
    failed = total_spatial - geocoded

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = [geocoded, failed]
    labels = [f'Geocoded\n({geocoded})', f'Failed\n({failed})']
    colors = ['#4caf50', '#f44336']
    explode = (0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title(f'Geocoding Success Rate\n(Total: {total_spatial} locations)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'geocoding_success.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved geocoding success plot")


def create_interactive_timeline(df: pd.DataFrame, output_dir: Path):
    """Create interactive timeline using Plotly."""
    # Parse temporal data
    temporal_df = df[df['temporal_normalized'].notna()].copy()

    dates = []
    diseases = []
    locations = []

    for _, row in temporal_df.iterrows():
        try:
            # Handle intervals (take start date)
            date_str = row['temporal_normalized'].split('/')[0] if '/' in row['temporal_normalized'] else row['temporal_normalized']
            dates.append(pd.to_datetime(date_str))
            diseases.append(row['disease'] if pd.notna(row['disease']) else 'Unknown')
            locations.append(row['spatial_text'] if pd.notna(row['spatial_text']) else 'Unknown')
        except:
            pass

    if not dates:
        logger.warning("No valid dates for interactive timeline")
        return

    plot_df = pd.DataFrame({'date': dates, 'disease': diseases, 'location': locations})
    plot_df = plot_df.sort_values('date')

    # Create cumulative count by disease
    plot_df['count'] = 1
    plot_df['cumulative'] = plot_df.groupby('disease')['count'].cumsum()

    # Create interactive plot
    fig = px.line(plot_df, x='date', y='cumulative', color='disease',
                  title='Cumulative Health Events Over Time',
                  labels={'cumulative': 'Cumulative Event Count', 'date': 'Date', 'disease': 'Disease'},
                  markers=True)

    fig.update_layout(
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=12),
        title_font=dict(size=18, family='Arial Black'),
        height=500
    )

    fig.write_html(output_dir / 'interactive_timeline.html')
    logger.info(f"✓ Saved interactive timeline")


def generate_all_plots(
    results_file: str = "case_studies/public_health/data/results/batch_extraction_results.json",
    output_dir: str = "case_studies/public_health/data/results/plots"
):
    """Generate all visualization plots."""
    logger.info("=" * 80)
    logger.info("Generating Statistical Visualizations")
    logger.info("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_file)
    df = extract_data_for_analysis(results)

    logger.info(f"\nDataframe shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Generate plots
    logger.info("\nGenerating plots...")

    plot_temporal_distribution(df, output_path)
    plot_disease_distribution(df, output_path)
    plot_venue_type_distribution(df, output_path)
    plot_event_type_distribution(df, output_path)
    plot_spatial_distribution(df, output_path)
    plot_disease_by_source(df, output_path)
    plot_extraction_metrics(results, output_path)
    plot_geocoding_success(df, output_path)
    create_interactive_timeline(df, output_path)

    logger.info("\n" + "=" * 80)
    logger.info("All plots generated successfully!")
    logger.info("=" * 80)
    logger.info(f"\n✓ Plots saved to: {output_path}")
    logger.info(f"\nGenerated files:")
    for file in sorted(output_path.glob("*")):
        logger.info(f"  - {file.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate statistical visualizations from extraction results")
    parser.add_argument(
        "--results",
        type=str,
        default="case_studies/public_health/data/results/batch_extraction_results.json",
        help="Path to extraction results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="case_studies/public_health/data/results/plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    try:
        generate_all_plots(args.results, args.output)
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
