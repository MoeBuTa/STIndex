# Public Health Surveillance Case Study

## Overview

This case study demonstrates STIndex's multi-dimensional extraction capabilities for public health surveillance, specifically tracking the 2025 measles and influenza outbreaks in Australia.

## Key Contributions

### 1. Novel Multi-Dimensional Extraction
Unlike traditional pipelines that extract spatial and temporal information separately, STIndex extracts **multiple configurable dimensions** in a single LLM call:

- **Temporal**: When did the exposure occur? (absolute and relative time expressions)
- **Spatial**: Where did it happen? (with geocoding and disambiguation)
- **Event Type**: What kind of event? (exposure site, case report, outbreak alert)
- **Venue Type**: What type of location? (hospital, clinic, emergency department, public venue)
- **Disease**: Which disease? (measles, influenza, etc.)
- **Patient Demographics**: Age group, vaccination status (if mentioned)

### 2. "Killer Feature": WA Disambiguation
Demonstrates context-aware disambiguation of "WA Health" alerts:
- **Western Australia** (Pilbara, Margaret River, Broome)
- **Washington State, USA** (King County, Spokane, Snohomish)

STIndex uses document-level context (co-occurring toponyms) to correctly identify the source.

### 3. Relative Temporal Resolution
Handles relative temporal expressions grounded in document metadata:
- "Monday from 11:00am to 7:00pm" → "2025-10-27T11:00:00/2025-10-27T19:00:00"
- Uses document publication date as anchor point

## Data Sources

### Primary Sources (Live Scraping)
1. **WA Health Measles Alerts**: https://www.health.wa.gov.au/news/2025/measles-alert
2. **WA Infectious Disease Alerts**: https://www.health.wa.gov.au/Articles/F_I/Health-alerts-infectious-diseases
3. **Washington State Measles Data**: https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025
4. **Australian Influenza Stats**: https://immunisationcoalition.org.au/influenza-statistics/

### Secondary Sources (News Articles)
- Guardian, PerthNow, and other Australian news outlets for narrative descriptions

## Use Cases

### Use Case 1: Measles Outbreak Timeline
**Input**: WA Health alert describing exposure sites with dates and times
**Output**: Animated map showing exposure locations over time with venue types

**Dimensions Extracted**:
```yaml
- temporal: "Monday 27/10/2025 from 11:00 am to 7:00 pm"
- spatial: "Margaret River Emergency Department, Western Australia"
- event_type: "exposure_site"
- venue_type: "hospital_emergency"
- disease: "measles"
```

### Use Case 2: Cross-Country Disambiguation
**Input**: Two "WA Health" documents (Australia vs USA)
**Output**: Correctly attributed locations based on context

**Australia Document**:
- Context: "Pilbara", "Perth", "Western Australia"
- WA → Western Australia

**USA Document**:
- Context: "King County", "Spokane", "Washington"
- WA → Washington State

### Use Case 3: Multi-Disease Surveillance
**Input**: Mixed health alerts (measles, influenza, COVID)
**Output**: Disease-specific timelines and geographic distributions

## Folder Structure

```
case_studies/public_health/
├── README.md                         # This file
├── data/                             # Data directory (gitignored)
│   ├── chunks/                       # Preprocessed document chunks
│   ├── results/                      # Extraction results
│   └── visualizations/               # Generated visualizations
├── extraction/
│   └── config/
│       └── health_dimensions.yml    # Health-specific dimension config
└── scripts/
    ├── run_case_study.py             # Main case study script
    └── run_complete_pipeline.sh      # Shell script wrapper
```

**Note**: Preprocessing and visualization are handled by STIndex core modules (`stindex.preprocessing` and `stindex.visualization`). No case-specific code needed.

## Installation

```bash
# Install STIndex with dependencies
pip install -e .

# Install optional dependencies for advanced parsing
pip install 'unstructured[local-inference]'
```

## Quick Start

### Run Full Pipeline

```bash
# From STIndex root directory
python case_studies/public_health/scripts/run_case_study.py
```

This will:
1. **Preprocess**: Scrape health surveillance pages, parse HTML, chunk documents
2. **Extract**: Extract temporal, spatial, and health-specific dimensions
3. **Visualize**: Generate interactive maps, plots, and HTML report

### Output Structure

```
case_studies/public_health/data/
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

### Python API Example

```python
from stindex import InputDocument, STIndexPipeline

# Define input documents
docs = [
    InputDocument.from_url(
        url="https://www.health.wa.gov.au/news/2025/measles-alert",
        metadata={"source": "WA Health", "year": 2025}
    ),
    InputDocument.from_url(
        url="https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025",
        metadata={"source": "WA State DOH", "year": 2025}
    ),
]

# Run pipeline
pipeline = STIndexPipeline(
    dimension_config="case_studies/public_health/extraction/config/health_dimensions",
    output_dir="case_studies/public_health/data"
)

results = pipeline.run_pipeline(docs, save_results=True, visualize=True)
```

## Evaluation Metrics

### Dimensional Extraction (Per Dimension)
- **Precision, Recall, F1**: Entity recognition accuracy
- **Normalization Accuracy**: Correct ISO 8601 conversion (temporal)
- **Geocoding Success**: Percentage successfully geocoded (spatial)
- **Distance Error**: Mean error distance for geocoded locations
- **Classification Accuracy**: Correct category assignment (event_type, venue_type, disease)

### Cross-Dimensional Linking
- **Event F1**: Correctly linked (location + time + event_type)
- **Co-reference Resolution**: Same location mentioned multiple times

### Disambiguation
- **WA Accuracy**: % correctly attributed to Australia vs USA

## Comparison to Baselines

We benchmark against:
- **TopoBERT** (spatial NER)
- **pyTLEX** (temporal extraction)
- **Pipeline approach**: TopoBERT + pyTLEX + manual linking

**Hypothesis**: STIndex's unified extraction will have:
1. Higher F1 for cross-dimensional linking
2. Better disambiguation (uses full document context)
3. Fewer errors from pipeline composition

## Demo Paper Sections

1. **Introduction**: Public health surveillance challenge
2. **System Architecture**: Multi-dimensional extraction framework
3. **Use Cases**: 3 use cases (measles timeline, WA disambiguation, multi-disease)
4. **Evaluation**: Metrics comparison against baselines
5. **Live Demo**: Interactive Folium map with animated timeline
6. **Conclusion**: Generalizability to other domains

## Future Extensions

- Real-time monitoring dashboard
- Integration with official health APIs
- Predictive modeling of outbreak spread
- Multi-language support (non-English health alerts)
