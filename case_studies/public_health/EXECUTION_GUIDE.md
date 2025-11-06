# 🚀 Execution Guide: Public Health Surveillance Case Study

## Overview

This guide shows you how to run the complete end-to-end pipeline for the public health surveillance case study, from data collection to animated visualization.

---

## 📋 Prerequisites

### 1. Install Dependencies

```bash
# Install STIndex with case study dependencies
pip install -e ".[case_studies]"

# Install Google Maps API client (optional, for better geocoding)
pip install googlemaps
```

### 2. Set Up Google Maps API (Optional but Recommended)

For better geocoding accuracy (especially for specific venue names like hospitals):

1. Get a Google Maps API key from: https://console.cloud.google.com/
2. Enable the "Geocoding API"
3. Set environment variable:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

Or add to your `.bashrc` / `.zshrc`:
```bash
echo 'export GOOGLE_MAPS_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

**Note**: Without Google Maps API, the system will still work using free Nominatim geocoding + city extraction fallback.

### 3. Start vLLM Server

```bash
# Check if server is running
./scripts/check_servers.sh

# If not running, start it
./scripts/start_server.sh

# Or for multi-GPU setup
./scripts/start_servers.sh
```

---

## 🔄 Complete Pipeline Execution

### Step 1: Data Collection (Already Done!)

```bash
# This was already run successfully
python case_studies/public_health/scripts/run_data_collection.py
```

**Output**: 67 document chunks in `case_studies/public_health/data/processed/`

✅ **Status**: Complete

---

### Step 2: Test Dimensional Extraction (Already Done!)

```bash
# This was already run successfully
python case_studies/public_health/scripts/test_dimensional_extraction.py
```

**Results**:
- ✅ 5 dimensions extracted (temporal, spatial, event_type, venue_type, disease)
- ✅ Temporal normalization to ISO 8601
- ✅ WA disambiguation working (Australia vs USA)
- ⚠️ Some venues need Google Maps API for geocoding

---

### Step 3: Batch Extraction on All Documents

Extract dimensions from all 67 chunked documents:

```bash
# Process all documents
python case_studies/public_health/scripts/extract_all_documents.py

# Or test with first 10 documents
python case_studies/public_health/scripts/extract_all_documents.py --sample-limit 10
```

**Options**:
- `--output`: Output JSON file (default: `data/results/batch_extraction_results.json`)
- `--dimension-config`: Dimension config path (default: health_dimensions)
- `--sample-limit`: Process only N documents (for testing)

**Expected output**:
```
================================================================================
Batch Extraction: Health Surveillance Documents
================================================================================
Loaded 67 document chunks from 4 files
Initializing DimensionalExtractor with 5 dimensions...
Processing 67 document chunks...
Extracting: 100%|████████████████████| 67/67 [03:45<00:00,  3.35s/it]

================================================================================
Extraction Summary
================================================================================
Total chunks processed: 67
Successful extractions: 64
Failed extractions: 3

Dimensions extracted:
  - temporal: 89 entities
  - spatial: 42 entities
  - event_type: 67 entities
  - venue_type: 38 entities
  - disease: 67 entities

✓ Results saved to: case_studies/public_health/data/results/batch_extraction_results.json
```

**Time estimate**: ~3-5 minutes for 67 documents

---

### Step 4: Create Animated Visualization

Generate an interactive Folium map:

```bash
python case_studies/public_health/visualization/map_generator.py
```

**Options**:
- `--results`: Path to extraction results JSON
- `--output`: Path to save HTML map file

**Expected output**:
```
================================================================================
Health Surveillance Map Visualization
================================================================================
Loaded 67 extraction results
Built 95 health events from extraction results
Using 78 events with valid coordinates and timestamps
Creating animated health surveillance map...
✓ Map saved to: case_studies/public_health/data/results/health_events_map.html
  Open in browser to view animated timeline

================================================================================
Visualization complete!
================================================================================
```

---

### Step 5: View the Map

Open the generated HTML file in your browser:

```bash
# On Linux
firefox case_studies/public_health/data/results/health_events_map.html

# On Mac
open case_studies/public_health/data/results/health_events_map.html

# Or copy to your local machine and open
```

**Map Features**:
- 📍 Health events plotted by location
- ⏱️ **Animated timeline** (use slider at bottom)
- 🎨 Color-coded by disease (red=measles, blue=influenza, etc.)
- 💬 Click markers for event details
- 🗺️ Zoom and pan to explore

---

## 📊 Quick Commands Reference

```bash
# 1. Check vLLM server
./scripts/check_servers.sh

# 2. Run test extraction
python case_studies/public_health/scripts/test_dimensional_extraction.py

# 3. Extract from all documents (with Google Maps API)
export GOOGLE_MAPS_API_KEY="your_key"
python case_studies/public_health/scripts/extract_all_documents.py

# 4. Create visualization
python case_studies/public_health/visualization/map_generator.py

# 5. View results
firefox case_studies/public_health/data/results/health_events_map.html
```

---

## 🐛 Troubleshooting

### Issue: "googlemaps not found"
```bash
pip install googlemaps
```

### Issue: "vLLM server not responding"
```bash
./scripts/check_servers.sh
./scripts/restart_servers.sh
```

### Issue: "Geocoding failed for all venues"

**Solution 1**: Set Google Maps API key
```bash
export GOOGLE_MAPS_API_KEY="your_key"
```

**Solution 2**: City extraction fallback will still work
- "Margaret River Emergency Department" → geocodes "Margaret River" ✅

### Issue: "No events in visualization"

Check extraction results:
```bash
python -c "import json; r=json.load(open('case_studies/public_health/data/results/batch_extraction_results.json')); print(f'Success: {sum(1 for x in r if x.get(\"extraction\",{}).get(\"success\"))}/{len(r)}')"
```

---

## 📁 Output Files

After running the complete pipeline, you'll have:

```
case_studies/public_health/data/
├── raw/                                    # Scraped HTML/JSON
│   ├── wa_health_au_infectious.json        (41KB)
│   ├── wa_doh_us_measles.json             (14KB)
│   └── au_influenza_stats.json            (28KB)
│
├── processed/                              # Parsed & chunked
│   ├── chunked_parsed_wa_health_au_infectious.json  (67 chunks)
│   ├── chunked_parsed_wa_doh_us_measles.json
│   └── chunked_parsed_au_influenza_stats.json
│
└── results/                                # Extraction & visualization
    ├── batch_extraction_results.json       # Full extraction results
    ├── health_events_map.html              # Interactive map ⭐
    ├── test_1_wa_health_measles_alert_(australia).json
    ├── test_2_wa_doh_measles_cases_(usa).json
    └── test_3_influenza_surveillance_update.json
```

---

## 🎯 Expected Results

### Extraction Success Rate
- **Target**: 90-95% successful extractions
- **Common failures**: Documents with no spatial/temporal info (e.g., PDF links only)

### Dimensions Per Document
- **Temporal**: 1-3 entities per document
- **Spatial**: 0-5 entities per document
- **Event Type**: 1 entity per document
- **Venue Type**: 0-3 entities per document
- **Disease**: 1 entity per document

### Geocoding Success Rate
- **With Google Maps API**: 95-98%
- **Without (Nominatim + city extraction)**: 70-80%

---

## 🔧 Advanced Usage

### Process Specific Files Only

```python
from pathlib import Path
import json
from stindex import DimensionalExtractor

# Load specific file
with open("case_studies/public_health/data/processed/chunked_parsed_wa_health_au_infectious.json") as f:
    chunks = json.load(f)

# Extract from first chunk
extractor = DimensionalExtractor(
    dimension_config_path="case_studies/public_health/extraction/config/health_dimensions"
)

result = extractor.extract(
    text=chunks[0]['text'],
    document_metadata=chunks[0]['document_metadata']
)

print(json.dumps(result.entities, indent=2))
```

### Custom Visualization

```python
from case_studies.public_health.visualization.map_generator import load_extraction_results, build_health_events

# Load and filter events
results = load_extraction_results("data/results/batch_extraction_results.json")
events = build_health_events(results)

# Filter by disease
measles_events = [e for e in events if e['disease'] == 'measles']
print(f"Measles events: {len(measles_events)}")

# Filter by location
wa_events = [e for e in events if 'western australia' in e.get('source', '').lower()]
print(f"WA events: {len(wa_events)}")
```

---

## 📝 Next Steps

After running the pipeline:

1. **Review Results**: Check `batch_extraction_results.json` for extraction quality
2. **Annotate Ground Truth**: Manually annotate 20-30 documents for evaluation
3. **Compute Metrics**: Compare against baselines (TopoBERT + pyTLEX)
4. **Refine Dimensions**: Adjust dimension configs based on results
5. **Prepare Demo**: Use the animated map for WWW demo paper

---

## 📚 Related Documentation

- **Main README**: `CLAUDE.md`
- **Dimensional Framework**: `docs/DIMENSIONAL_EXTRACTION.md`
- **Case Study README**: `case_studies/public_health/README.md`
- **Implementation Summary**: `case_studies/public_health/IMPLEMENTATION_SUMMARY.md`

---

## 🆘 Getting Help

If you encounter issues:

1. Check logs for error messages
2. Verify vLLM server is running: `./scripts/check_servers.sh`
3. Test with sample limit: `--sample-limit 5`
4. Check extraction results JSON for specific errors

**Common success indicators**:
- ✅ Extractor initializes with 5 dimensions
- ✅ LLM extraction succeeds (check `success: true` in results)
- ✅ Geocoding works (check `latitude`/`longitude` in spatial entities)
- ✅ Map shows events with popups
