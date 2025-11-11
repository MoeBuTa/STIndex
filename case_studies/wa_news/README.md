# Western Australia News Case Study

Demonstrates STIndex's end-to-end pipeline using Western Australia news articles and government documents to extract temporal, spatial, and entity information.

## Overview

This case study showcases STIndex's ability to process diverse document types (web articles and PDFs) from Western Australian sources, extracting multi-dimensional information including:
- **Temporal entities**: Dates, time periods, and temporal expressions
- **Spatial entities**: Locations with geocoding and disambiguation
- **Entity information**: Organizations, people, and events

The pipeline demonstrates preprocessing, context-aware extraction, post-processing (geocoding, temporal normalization), and interactive visualizations.

## Data Sources (10 Documents)

This case study processes **10 documents** from Western Australian and broader Australian news sources, demonstrating STIndex's end-to-end pipeline capabilities for multi-source news aggregation and entity extraction.

### 1. ABC News - Western Australia
**Source**: ABC News
**URL**: https://www.abc.net.au/news/wa
**Type**: Regional news feed
**Category**: WA News

Aggregated news feed covering Western Australia, including breaking news, local stories, and regional developments.

**Demonstrates extraction of**:
- Regional news events and developments
- WA locations (Perth, regional centers)
- Current affairs and breaking news
- Multiple story types and temporal references

---

### 2. Curtin University News
**Source**: Curtin University
**URL**: https://www.curtin.edu.au/news/
**Type**: University news feed
**Category**: University News

Official news from Curtin University covering research, campus developments, and institutional announcements.

**Demonstrates extraction of**:
- Research milestones and discoveries
- Campus locations and facilities
- Academic events and conferences
- University organizational entities

---

### 3. Edith Cowan University News
**Source**: Edith Cowan University (ECU)
**URL**: https://www.ecu.edu.au/news
**Type**: University news feed
**Category**: University News

Official ECU news covering research achievements, student success stories, and university developments.

**Demonstrates extraction of**:
- Research outputs and collaborations
- Multiple ECU campus locations
- Academic programs and initiatives
- Educational institution entities

---

### 4. Perth Now - WA News
**Source**: PerthNow
**URL**: https://www.perthnow.com.au/news/wa
**Type**: News website
**Category**: WA News

Major WA news outlet covering local news, politics, crime, and community stories across Western Australia.

**Demonstrates extraction of**:
- Local news events
- Perth and regional WA locations
- Crime and court reporting
- Political developments

---

### 5. The West - WA News
**Source**: The West Australian
**URL**: https://www.thewest.com.au/news/wa
**Type**: News website
**Category**: WA News

The West Australian's WA news section, covering state politics, business, and local news.

**Demonstrates extraction of**:
- State-level news and politics
- Business and economic stories
- Regional WA coverage
- Government and policy entities

---

### 6. Australian Geographic
**Source**: Australian Geographic
**URL**: https://www.australiangeographic.com.au/
**Type**: Geographic and nature magazine
**Category**: Geography & Nature

National publication covering Australian geography, environment, wildlife, and natural sciences.

**Demonstrates extraction of**:
- Australian locations and geographic features
- Environmental and conservation topics
- Wildlife and biodiversity entities
- Natural phenomena and events

---

### 7. ABC Science News
**Source**: ABC News
**URL**: https://www.abc.net.au/news/science/
**Type**: Science news feed
**Category**: Science News

National science news covering research, discoveries, and scientific developments across Australia.

**Demonstrates extraction of**:
- Scientific research and discoveries
- Research institutions and universities
- Science events and announcements
- Technology and innovation topics

---

### 8. ABC Rural News
**Source**: ABC News
**URL**: https://www.abc.net.au/news/rural/
**Type**: Rural news feed
**Category**: Rural & Agriculture News

National rural and agricultural news covering farming, regional communities, and rural industries.

**Demonstrates extraction of**:
- Agricultural regions and farming locations
- Rural communities and industries
- Weather and climate impacts
- Agricultural organizations and commodities

---

### 9. WAtoday Homepage
**Source**: WAtoday
**URL**: https://www.watoday.com.au/
**Type**: News website homepage
**Category**: WA News & Current Affairs

Major WA news portal covering breaking news, politics, lifestyle, and entertainment for Western Australia.

**Demonstrates extraction of**:
- Diverse news topics (politics, lifestyle, entertainment)
- Perth metropolitan and regional WA locations
- Current affairs and opinion pieces
- Cultural and community events

---

### 10. PerthNow Homepage
**Source**: PerthNow
**URL**: https://www.perthnow.com.au/
**Type**: News website homepage
**Category**: Perth News & Current Affairs

Leading Perth news site covering local news, sport, entertainment, and lifestyle for Perth and WA.

**Demonstrates extraction of**:
- Perth-centric news and events
- Sport and entertainment coverage
- Lifestyle and community stories
- Local business and property news

---

## Run

```bash
python case_studies/wa_news/scripts/run_wa_news_case_study.py
```

## Expected Output

The pipeline generates:
- **Preprocessed chunks**: `data/chunks/preprocessed_chunks.json`
- **Extraction results**: `data/results/extraction_results.json`
- **Interactive visualizations**: `data/visualizations/*.html`
  - Temporal timelines showing event sequences
  - Interactive maps with geocoded locations
  - Entity relationship networks
  - Statistical summaries and distributions

## Key Features Demonstrated

- **Multi-source news aggregation**: Processes news feeds from multiple publishers (ABC, PerthNow, WAtoday, The West)
- **Diverse content types**: Regional news, university news, science, rural, and geographic content
- **Context-aware extraction**: Maintains context across document chunks and multiple stories
- **Geocoding**: Automatic geocoding of Australian locations (Perth, WA regions, national locations)
- **Temporal normalization**: ISO 8601 standardization of dates from various news formats
- **Entity extraction**: Organizations, institutions, locations, events, and topics
- **Visualization**: Interactive maps and timelines showing news patterns and geographic distributions
