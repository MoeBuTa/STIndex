# STIndex Report Package

This folder contains a **standalone, self-contained report** for the STIndex project with all visualizations and data.

## Contents

### Main Report
- **REPORT.html** - Complete HTML report with embedded visualizations (open this in any web browser)

### Visualization Files

#### Static Plots (PNG images)
1. **temporal_distribution.png** - Events over time distribution
2. **disease_distribution.png** - Disease breakdown (pie + bar charts)
3. **spatial_distribution.png** - Top 15 locations by event count
4. **event_type_distribution.png** - Event types distribution
5. **venue_distribution.png** - Venue types distribution
6. **disease_by_source.png** - Cross-dimensional analysis (disease by data source)
7. **extraction_metrics.png** - Extraction success rate & entity counts
8. **geocoding_success.png** - Geocoding performance (success rate)

#### Interactive Visualizations (HTML)
9. **interactive_timeline.html** - Plotly interactive cumulative timeline
10. **health_events_map.html** - Folium animated map with timeline slider (132 events)

## How to Use

### View the Complete Report
1. Open `REPORT.html` in any web browser (Chrome, Firefox, Safari, Edge)
2. All images and interactive visualizations are embedded/referenced locally
3. No internet connection required (fully standalone)

### View Individual Visualizations
- **Static plots**: Open any .png file in an image viewer
- **Interactive timeline**: Open `interactive_timeline.html` in a browser
- **Animated map**: Open `health_events_map.html` in a browser

## Report Sections

The REPORT.html includes:
1. **Overview** - What is STIndex
2. **Core Workflow** - Architecture diagram with modules and descriptions
3. **Generic Dimension Framework** - YAML-configurable system
4. **Case Study** - Public Health Surveillance pipeline (3 phases)
5. **Results** - Statistics and performance metrics
6. **Interactive Map** - Embedded animated timeline map
7. **Statistical Analysis** - 9 comprehensive plots with analysis
8. **Technical Contributions** - 4 key innovations
9. **Next Steps** - Evaluation roadmap and future work
10. **Summary** - Achievements and capabilities
11. **Quick Start** - Installation and execution commands

## Technical Details

- **Total Events Extracted**: 132 health events
- **Documents Processed**: 67 chunks from 3 data sources
- **Dimensions Extracted**: 5 (temporal, spatial, event_type, venue_type, disease)
- **Extraction Success Rate**: 95%
- **Geocoding Accuracy**: 95-98% (with multi-level fallback)

## Data Sources

1. **WA Health Australia** (45 chunks)
   - Region: Western Australia
   - Type: Infectious disease alerts
   - URL: https://www.health.wa.gov.au/~/media/Files/Corporate/general-documents/Medical-notifications/PDF/Measles-quick-guide-for-primary-healthcare-workers.pdf

2. **WA DOH USA** (12 chunks)
   - Region: Washington State, USA
   - Type: Measles case reports
   - URL: https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025

3. **Australian Flu Stats** (10 chunks)
   - Region: Australia
   - Type: Influenza surveillance
   - URL: https://immunisationcoalition.org.au/influenza-statistics/

## File Size
- Total package: ~5.5 MB
- REPORT.html: ~200 KB
- Images: ~1.3 MB
- Interactive HTML: ~3.7 MB

## Platform Compatibility
- Works on Windows, macOS, Linux
- Compatible with all modern web browsers
- No dependencies or installation required

## For Presentation
This package is ready for:
- 5-10 minute presentations
- Research demonstrations
- Sharing via email/USB/download
- Offline viewing during presentations

---

**Generated**: November 2025
**Project**: STIndex - Multi-Dimensional Spatiotemporal Information Extraction
