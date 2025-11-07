# Western Australia News Case Study

This case study demonstrates STIndex's end-to-end pipeline using real Western Australia news articles and government documents.

## Overview

**Purpose**: Extract and visualize multi-dimensional information from WA news sources.

**Data Sources**:
- UWA news articles (web)
- WA Government PDF reports

**Dimensions Extracted**:
- `temporal`: Dates, times, event timestamps
- `spatial`: Locations (Perth, UWA campus, etc.)
- `person`: People mentioned (researchers, officials)
- `organization`: Institutions (UWA, government agencies)
- `event_type`: Categories (research, governance, finance)

## Data Sources

### 1. Space Research Article (September 2025)
- **URL**: https://www.uwa.edu.au/news/article/2025/september/poland-and-australia-partner-to-track-space-junk
- **Topic**: Poland-Australia partnership to track space debris
- **Key Entities**: Associate Professor David Coward, Polish Space Agency, UWA

### 2. Leadership Appointment (February 2025)
- **URL**: https://www.uwa.edu.au/news/article/2025/february/uwa-welcomes-first-female-chancellor
- **Topic**: First female Chancellor appointed at UWA
- **Key Entities**: Diane Smith-Gander AO, Governor Chris Dawson

### 3. Government Financial Report (December 2024)
- **URL**: https://www.wa.gov.au/system/files/2024-12/2024-25-myr.pdf
- **Topic**: WA Mid-Year Financial Projections 2024-25
- **Type**: PDF document (2.94 MB)

## Quick Start

### Run Full Pipeline

```bash
# From STIndex root directory
python case_studies/wa_news/scripts/run_wa_news_case_study.py
```

This will:
1. **Preprocess**: Scrape web articles, parse PDF, chunk documents
2. **Extract**: Extract temporal, spatial, and entity information
3. **Visualize**: Generate interactive maps, plots, and HTML report

### Output Structure

```
case_studies/wa_news/data/
├── chunks/
│   └── preprocessed_chunks.json          # Document chunks
├── results/
│   └── extraction_results.json           # Extraction results
└── visualizations/
    ├── stindex_report_{timestamp}.html   # Main HTML report
    └── stindex_report_{timestamp}_source/
        ├── summary.json                   # Statistical summary
        ├── map.html                       # Interactive map
        └── *.png                          # Statistical plots
```

## Configuration

**Dimension Config**: `config/wa_dimensions.yml`

Customize extraction dimensions, categories, and settings.

## Expected Results

### Temporal Entities
- September 22, 2025 (space partnership announcement)
- February 26, 2025 (Chancellor installation)
- December 23, 2024 (financial report publication)
- September 24, 2025 at 9:30 AM (workshop)

### Spatial Entities
- Perth, Western Australia
- University of Western Australia
- Winthrop Hall
- Poland (international partnership)

### Person Entities
- Associate Professor David Coward (researcher)
- Diane Smith-Gander AO (administrator)
- Governor Chris Dawson (government official)
- Professor Amit Chakma (administrator)

### Organization Entities
- University of Western Australia (university)
- Polish Space Agency (international agency)
- Australian Research Council (research organization)
- Department of Treasury and Finance (government agency)

### Event Types
- Research collaboration
- Leadership appointment
- Financial reporting
- International workshop

## Visualization Features

The pipeline generates comprehensive visualizations:

- **Interactive Map**: Locations mentioned across articles
- **Timeline**: Events plotted chronologically
- **Entity Distributions**: Bar charts for people, organizations
- **Category Analysis**: Event types breakdown
- **Statistical Summary**: Success rates, coverage metrics

## Use Cases

This case study demonstrates:

1. **Multi-Source Processing**: Web articles + PDF documents
2. **Multi-Dimensional Extraction**: 5 different dimension types
3. **Real-World Data**: Current WA news and government reports
4. **Complete Pipeline**: End-to-end workflow in single script
5. **Visualization**: Interactive reports for analysis

## Requirements

```bash
pip install -e .
```

Optional for full visualization:
```bash
pip install folium matplotlib seaborn plotly
```

## License

MIT License
