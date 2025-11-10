# STIndex Workflow Diagrams

Comprehensive workflow diagrams for STIndex v0.5.0 architecture with context-aware extraction.

---

## 1. Full End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STIndex Pipeline                                │
│                     (STIndexPipeline Orchestrator)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │          INPUT DOCUMENTS                       │
        ├───────────────────────────────────────────────┤
        │  • Web URLs (HTTP/HTTPS)                      │
        │  • Local Files (PDF, HTML, DOCX, TXT)         │
        │  • Raw Text Strings                           │
        │                                                │
        │  Created via InputDocument.from_*()           │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING STAGE                                 │
│                      (stindex/preprocessing/)                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│  │   SCRAPING  │      │   PARSING   │      │  CHUNKING   │               │
│  │  (for URLs) │ ───> │ (all types) │ ───> │ (long docs) │               │
│  └─────────────┘      └─────────────┘      └─────────────┘               │
│       │                     │                     │                        │
│       │ WebScraper          │ DocumentParser      │ DocumentChunker       │
│       │ • Rate limiting     │ • HTML parsing      │ • Sliding window      │
│       │ • User agent        │ • PDF parsing       │ • Paragraph-based     │
│       │ • Error handling    │ • DOCX parsing      │ • Element-based (NEW) │
│       │                     │ • Text extraction   │ • Semantic (future)   │
│       │                     │ • Structured        │ • Context overlap     │
│       │                     │   sections          │                        │
│       │                     │                     │                        │
│       └─────────────────────┴─────────────────────┘                       │
│                                  │                                          │
│                         ParsedDocument                                      │
│                         (structured text)                                   │
│                         • sections (NEW)                                    │
│                         • tables (NEW)                                      │
│                                  │                                          │
│                                  ▼                                          │
│                         DocumentChunk[]                                     │
│                         • chunk_id                                          │
│                         • text                                              │
│                         • metadata                                          │
│                         • section_hierarchy (NEW)                           │
│                         • keywords (NEW)                                    │
│                         • preview (NEW)                                     │
│                         • position info                                     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                        Save to: data/chunks/
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION STAGE                                    │
│                   (stindex/extraction/dimensional_extraction.py)           │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │              DimensionalExtractor                              │        │
│  │  • Load dimension config (YAML)                                │        │
│  │  • Build JSON schema for all dimensions                        │        │
│  │  • Initialize ExtractionContext (NEW)                          │        │
│  │  • Prepare context-aware LLM prompts                           │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │              ExtractionContext (NEW v0.5.0)                    │        │
│  │         (stindex/extraction/context_manager.py)                │        │
│  ├───────────────────────────────────────────────────────────────┤        │
│  │  Context Engineering Best Practices:                           │        │
│  │                                                                 │        │
│  │  • cinstr: Instruction context (schemas, tasks)                │        │
│  │  • ctools: Tool context (geocoding, normalization)             │        │
│  │  • cmem: Memory context (prior extractions)                    │        │
│  │  • cstate: State context (metadata, position)                  │        │
│  │                                                                 │        │
│  │  Prior References (Sliding Window):                            │        │
│  │  ├─ prior_temporal_refs (last N temporal mentions)             │        │
│  │  ├─ prior_spatial_refs (last N spatial mentions)               │        │
│  │  └─ prior_events (last N events)                               │        │
│  │                                                                 │        │
│  │  Document State:                                               │        │
│  │  ├─ publication_date (anchor for relative dates)               │        │
│  │  ├─ source_location (spatial disambiguation)                   │        │
│  │  ├─ chunk_position (current/total)                             │        │
│  │  └─ section_hierarchy (document structure)                     │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │                    LLMManager                                  │        │
│  │         (stindex/llm/manager.py)                               │        │
│  ├───────────────────────────────────────────────────────────────┤        │
│  │  Factory creates provider based on config:                     │        │
│  │                                                                 │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │        │
│  │  │  OpenAILLM   │  │ AnthropicLLM │  │ HuggingFace  │        │        │
│  │  │              │  │              │  │   LLM        │        │        │
│  │  │ gpt-4o-mini  │  │ claude-3.5   │  │ Qwen3-8B     │        │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │        │
│  │                                                                 │        │
│  │  All expose: generate(messages) → LLMResponse                  │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                         LLM generates JSON                                  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │              JSON Extraction & Validation                      │        │
│  │         (stindex/core/utils.py)                                │        │
│  │                                                                 │        │
│  │  • Find all JSON objects in output                             │        │
│  │  • Parse from last to first (handles corrections)              │        │
│  │  • Validate against Pydantic models                            │        │
│  │  • Extract: {dimension_name: [mentions]}                       │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │           Dimension-Specific Post-Processing                   │        │
│  ├───────────────────────────────────────────────────────────────┤        │
│  │                                                                 │        │
│  │  For each dimension type:                                      │        │
│  │                                                                 │        │
│  │  NORMALIZED (temporal):                                        │        │
│  │  └─> Validate ISO 8601 format                                 │        │
│  │  └─> Resolve relative expressions with context (NEW)          │        │
│  │  └─> Add character positions                                  │        │
│  │                                                                 │        │
│  │  GEOCODED (spatial):                                           │        │
│  │  └─> GeocoderService                                           │        │
│  │      • Nominatim API                                           │        │
│  │      • Parent region hints                                     │        │
│  │      • OSM nearby locations (NEW)                              │        │
│  │      • spaCy NER extraction                                    │        │
│  │      • Cache results                                           │        │
│  │                                                                 │        │
│  │  CATEGORICAL (disease, event_type):                            │        │
│  │  └─> Validate category against allowed values                 │        │
│  │                                                                 │        │
│  │  STRUCTURED:                                                   │        │
│  │  └─> Return as-is with field validation                       │        │
│  │                                                                 │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │        Update ExtractionContext Memory (NEW)                   │        │
│  │        • Add new temporal refs to sliding window               │        │
│  │        • Add new spatial refs to sliding window                │        │
│  │        • Update chunk position                                 │        │
│  │        • Keep last N references only                           │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │        Two-Pass Verification (Optional, NEW)                   │        │
│  │        (stindex/postprocess/verification.py)                   │        │
│  ├───────────────────────────────────────────────────────────────┤        │
│  │  Second LLM pass for quality assurance:                        │        │
│  │                                                                 │        │
│  │  Score each entity on:                                         │        │
│  │  • Relevance (0-1): Is it in the text?                         │        │
│  │  • Accuracy (0-1): Does it match exactly?                      │        │
│  │  • Completeness (0-1): Is it complete?                         │        │
│  │                                                                 │        │
│  │  Filter entities:                                              │        │
│  │  └─> Keep only relevance ≥ 0.7 AND accuracy ≥ 0.7             │        │
│  │                                                                 │        │
│  │  Expected impact: -40-60% false positive rate                  │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│                     MultiDimensionalResult                                  │
│                     • entities: {dim: [entity]}                             │
│                     • success: bool                                         │
│                     • processing_time                                       │
│                     • extraction_config                                     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                       Save to: data/results/
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION STAGE (Optional)                        │
│                      (stindex/visualization/)                              │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │              STIndexVisualizer                                 │        │
│  │         (stindex/visualization/visualizer.py)                  │        │
│  │                                                                 │        │
│  │  Orchestrates all visualization components                     │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    STEP 1: Statistical Summary                       │  │
│  │               (StatisticalSummary.generate_summary)                  │  │
│  ├─────────────────────────────────────────────────────────────────────┤  │
│  │  • Overview stats (success rate, total chunks)                       │  │
│  │  • Dimension stats (entity counts, unique values)                    │  │
│  │  • Source stats (documents processed)                                │  │
│  │  • Performance stats (processing times)                              │  │
│  │  • Temporal coverage (date ranges)                                   │  │
│  │  • Spatial coverage (geocoding rate, bounds)                         │  │
│  │                                                                        │  │
│  │  Output: summary.json                                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    STEP 2: Statistical Plots                         │  │
│  │                (PlotGenerator.generate_plots)                        │  │
│  ├─────────────────────────────────────────────────────────────────────┤  │
│  │  For each dimension:                                                 │  │
│  │                                                                        │  │
│  │  TEMPORAL dimensions:                                                │  │
│  │  └─> Temporal distribution over time (bar chart)                    │  │
│  │  └─> Interactive cumulative timeline (plotly)                       │  │
│  │                                                                        │  │
│  │  SPATIAL dimensions:                                                 │  │
│  │  └─> Top 15 locations by count (horizontal bar)                     │  │
│  │                                                                        │  │
│  │  CATEGORICAL dimensions:                                             │  │
│  │  └─> Category distribution (bar chart)                              │  │
│  │                                                                        │  │
│  │  CROSS-DIMENSIONAL:                                                  │  │
│  │  └─> Category by category (stacked bar)                             │  │
│  │                                                                        │  │
│  │  EXTRACTION METRICS:                                                 │  │
│  │  └─> Entity count by dimension (bar chart)                          │  │
│  │                                                                        │  │
│  │  Output: *.png, *.html (interactive)                                │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    STEP 3: Interactive Map                           │  │
│  │                  (MapGenerator.generate_map)                         │  │
│  ├─────────────────────────────────────────────────────────────────────┤  │
│  │  Extract events (link temporal + spatial + category):               │  │
│  │  • Location (latitude, longitude)                                   │  │
│  │  • Timestamp (normalized temporal value)                            │  │
│  │  • Category (for color coding)                                      │  │
│  │  • All dimension entities (for popup)                               │  │
│  │                                                                        │  │
│  │  Generate map mode:                                                  │  │
│  │                                                                        │  │
│  │  STATIC MAP:                                                         │  │
│  │  └─> MarkerCluster for all events                                   │  │
│  │  └─> Color-coded by category                                        │  │
│  │  └─> Popups with entity details                                     │  │
│  │                                                                        │  │
│  │  ANIMATED MAP (if temporal data):                                   │  │
│  │  └─> TimestampedGeoJson with timeline slider                        │  │
│  │  └─> Events appear chronologically                                  │  │
│  │  └─> Play/pause controls                                            │  │
│  │                                                                        │  │
│  │  Output: map.html (Folium)                                          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    STEP 4: HTML Report                               │  │
│  │              (HTMLReportGenerator.generate_report)                   │  │
│  ├─────────────────────────────────────────────────────────────────────┤  │
│  │  Generate comprehensive HTML with:                                   │  │
│  │                                                                        │  │
│  │  • Header (title, timestamp)                                         │  │
│  │  • Table of Contents (navigation)                                    │  │
│  │  • Overview Section (stat cards)                                     │  │
│  │  • Dimensional Analysis (table)                                      │  │
│  │  • Interactive Map (iframe embed)                                    │  │
│  │  • Statistical Visualizations (embedded plots)                       │  │
│  │  • Performance Metrics (stat cards)                                  │  │
│  │  • Data Sources (table)                                              │  │
│  │  • Footer (version info)                                             │  │
│  │                                                                        │  │
│  │  Output: stindex_report_{timestamp}.html                            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                    Save to: data/output/visualizations/
                             {timestamp}.html
                             {timestamp}_source/*.png
                             {timestamp}_source/map.html
                             {timestamp}_source/summary.json
                                    │
                                    ▼
                              ┌──────────┐
                              │  DONE!   │
                              └──────────┘
```

---

## 2. Preprocessing Module (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Preprocessor.process()                        │
│               (stindex/preprocessing/processor.py)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  InputDocument   │
                    │  • input_type    │
                    │  • content       │
                    │  • metadata      │
                    └──────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Route by Type  │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  INPUT: URL  │      │ INPUT: FILE  │      │ INPUT: TEXT  │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        ▼                     ▼                     │
┌──────────────────────────────────────┐            │
│         WebScraper                    │            │
│  (stindex/preprocessing/scraping.py)  │            │
├──────────────────────────────────────┤            │
│  1. Rate limit check (2s between)    │            │
│  2. HTTP GET with user agent         │            │
│  3. Error handling                   │            │
│  4. Return HTML content               │            │
└──────────────────────────────────────┘            │
        │                     │                     │
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                     HTML / File / Text
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         DocumentParser                       │
        │  (stindex/preprocessing/parsing.py)          │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Parsing Method: "unstructured" or "simple"  │
        │                                              │
        │  UNSTRUCTURED:                               │
        │  ├─ partition_html() or partition()          │
        │  ├─ Extract elements:                        │
        │  │  • Title                                  │
        │  │  • NarrativeText                          │
        │  │  • Table                                  │
        │  ├─ Clean whitespace                         │
        │  └─ Combine to full text                     │
        │                                              │
        │  SIMPLE:                                     │
        │  ├─ BeautifulSoup for HTML                   │
        │  ├─ Plain text read for files                │
        │  └─ Basic text extraction                    │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ ParsedDocument   │
                    │ • title          │
                    │ • content        │
                    │ • sections       │
                    │ • tables         │
                    │ • metadata       │
                    └──────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         DocumentChunker                      │
        │  (stindex/preprocessing/chunking.py)         │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Check: len(text) <= max_chunk_size?         │
        │                                              │
        │  NO  → Apply chunking strategy:              │
        │                                              │
        │  SLIDING_WINDOW:                             │
        │  ├─ Fixed size chunks                        │
        │  ├─ Overlap for context                      │
        │  └─ Break at sentence boundaries             │
        │                                              │
        │  PARAGRAPH:                                  │
        │  ├─ Split by \n\n                            │
        │  ├─ Group paragraphs to max size             │
        │  └─ Handle oversized paragraphs              │
        │                                              │
        │  ELEMENT_BASED (NEW):                        │
        │  ├─ Chunk by structural elements             │
        │  ├─ Start new chunks at titles/tables        │
        │  ├─ Keep tables intact (never fragment)      │
        │  ├─ Track section hierarchy                  │
        │  └─ Add enriched metadata:                   │
        │      • section_hierarchy                     │
        │      • keywords (frequency-based)            │
        │      • preview (first 2 sentences)           │
        │      • element_types                         │
        │                                              │
        │  SEMANTIC (future):                          │
        │  └─ Embedding-based chunking                 │
        │                                              │
        │  YES → Single chunk (no splitting)           │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ DocumentChunk[]  │
                    │ • chunk_id       │
                    │ • chunk_index    │
                    │ • text           │
                    │ • word_count     │
                    │ • metadata       │
                    │ • start_char     │
                    │ • end_char       │
                    │ • section_hierarchy (NEW) │
                    │ • keywords (NEW)          │
                    │ • preview (NEW)           │
                    │ • element_types (NEW)     │
                    └──────────────────┘
                              │
                              ▼
                         OUTPUT READY
```

---

## 3. Extraction Module (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│              DimensionalExtractor.extract()                      │
│         (stindex/core/dimensional_extraction.py)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                   Input: text, metadata
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  STEP 1: Load Dimension Configuration       │
        │  (stindex/utils/dimension_loader.py)         │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Load YAML config:                           │
        │  cfg/dimensions.yml or custom config         │
        │                                              │
        │  Example dimensions:                         │
        │  • temporal (normalized)                     │
        │  • spatial (geocoded)                        │
        │  • disease (categorical)                     │
        │  • event_type (categorical)                  │
        │                                              │
        │  Build JSON schema for all enabled dims      │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  STEP 2: Build LLM Prompt                   │
        │  (stindex/llm/prompts/dimensional_*.py)      │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  System Prompt:                              │
        │  "You are an expert information extractor"  │
        │                                              │
        │  User Prompt:                                │
        │  ├─ Input text                               │
        │  ├─ Document metadata (context)              │
        │  ├─ Dimension descriptions                   │
        │  ├─ JSON schema                              │
        │  └─ Few-shot examples (optional)             │
        │                                              │
        │  Format: ChatML messages                     │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  STEP 3: LLM Generation                     │
        │  (stindex/llm/manager.py)                    │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  LLMManager.generate(messages)               │
        │         │                                    │
        │         └─> Select provider from config:     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │  OpenAI (openai.py)                │     │
        │  │  • API key auth                    │     │
        │  │  • client.chat.completions.create  │     │
        │  │  • Async batch support             │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │  Anthropic (anthropic.py)          │     │
        │  │  • API key auth                    │     │
        │  │  • client.messages.create          │     │
        │  │  • Async batch support             │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │  HuggingFace (hf.py)               │     │
        │  │  • Connect to vLLM server(s)       │     │
        │  │  • Load balancing (multi-GPU)      │     │
        │  │  • Health checking                 │     │
        │  │  • Qwen3 chat template support     │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  Returns: LLMResponse                        │
        │  • content: raw text output                  │
        │  • success: bool                             │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  STEP 4: JSON Extraction                    │
        │  (stindex/core/utils.py)                     │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  extract_json_from_text(raw_output)          │
        │                                              │
        │  1. Find all complete JSON objects:          │
        │     • Track brace depth                      │
        │     • Ignore braces in strings               │
        │     • Extract complete {...} blocks          │
        │                                              │
        │  2. Parse from LAST to FIRST:                │
        │     • Handles model corrections              │
        │     • Prefers final output                   │
        │                                              │
        │  3. Validate with Pydantic:                  │
        │     • Try each candidate                     │
        │     • Return first valid                     │
        │                                              │
        │  Output: dict                                │
        │  {                                           │
        │    "temporal": [...],                        │
        │    "spatial": [...],                         │
        │    "disease": [...]                          │
        │  }                                           │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  STEP 5: Post-Processing (Per Dimension)    │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  For dimension_name, mentions in dict:       │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │ NORMALIZED (temporal)              │     │
        │  │ • Validate ISO 8601 format         │     │
        │  │ • Add character positions          │     │
        │  │ → NormalizedDimensionEntity        │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │ GEOCODED (spatial)                 │     │
        │  │ • Call GeocoderService             │     │
        │  │ • Use parent_region hint           │     │
        │  │ • Extract coords (lat, lon)        │     │
        │  │ • Cache results                    │     │
        │  │ → GeocodedDimensionEntity          │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │ CATEGORICAL (disease, event_type)  │     │
        │  │ • Validate category in allowed     │     │
        │  │ • Store category + confidence      │     │
        │  │ → CategoricalDimensionEntity       │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        │  ┌────────────────────────────────────┐     │
        │  │ STRUCTURED                         │     │
        │  │ • Validate fields                  │     │
        │  │ • Return as dict                   │     │
        │  └────────────────────────────────────┘     │
        │                                              │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ MultiDimensional │
                    │     Result       │
                    │                  │
                    │ entities: {      │
                    │   temporal: [...] │
                    │   spatial: [...]  │
                    │   disease: [...]  │
                    │ }                │
                    │ success: true    │
                    │ processing_time  │
                    └──────────────────┘
                              │
                              ▼
                         OUTPUT READY
```

---

## 4. Pipeline Execution Modes

```
┌─────────────────────────────────────────────────────────────────┐
│                   STIndexPipeline Modes                          │
└─────────────────────────────────────────────────────────────────┘

MODE 1: FULL PIPELINE
═══════════════════════════════════════════════════════════════════
Input: InputDocument[]
        │
        ▼
  Preprocessing → DocumentChunk[]
        │
        │ (save to chunks/)
        ▼
  Extraction → Results[]
        │
        │ (save to results/)
        ▼
  Visualization → Maps, Plots
        │
        │ (save to visualizations/)
        ▼
Output: Results[] + Visualizations


MODE 2: PREPROCESSING ONLY
═══════════════════════════════════════════════════════════════════
Input: InputDocument[]
        │
        ▼
  Preprocessing → DocumentChunk[]
        │
        │ (save to chunks/)
        ▼
Output: DocumentChunk[]


MODE 3: EXTRACTION ONLY
═══════════════════════════════════════════════════════════════════
Input: DocumentChunk[] (load from file)
        │
        ▼
  Extraction → Results[]
        │
        │ (save to results/)
        ▼
Output: Results[]


MODE 4: VISUALIZATION ONLY
═══════════════════════════════════════════════════════════════════
Input: Results[] (load from file)
        │
        ▼
  Visualization → Maps, Plots
        │
        │ (save to visualizations/)
        ▼
Output: Visualizations
```

---

## 5. Data Flow & Models

```
┌────────────────────────────────────────────────────────────────┐
│                    Data Models Flow                             │
└────────────────────────────────────────────────────────────────┘

INPUT STAGE
═══════════════════════════════════════════════════════════════════
InputDocument
├─ input_type: URL | FILE | TEXT
├─ content: str (URL/path/text)
├─ metadata: dict
├─ document_id: str
└─ title: str

    Factory methods:
    • InputDocument.from_url(url, metadata)
    • InputDocument.from_file(file_path, metadata)
    • InputDocument.from_text(text, metadata)


PREPROCESSING STAGE
═══════════════════════════════════════════════════════════════════
ParsedDocument (intermediate)
├─ document_id: str
├─ title: str
├─ content: str (full text)
├─ sections: List[dict]
├─ tables: List[dict]
├─ metadata: dict
└─ parsing_method: str

            ↓ (chunking)

DocumentChunk[]
├─ chunk_id: str
├─ chunk_index: int
├─ total_chunks: int
├─ text: str
├─ word_count: int
├─ char_count: int
├─ document_id: str
├─ document_title: str
├─ document_metadata: dict
├─ start_char: int
├─ end_char: int
├─ previous_chunk_summary: str?
├─ section_hierarchy: str? (NEW v0.5.0)
├─ keywords: List[str]? (NEW v0.5.0)
├─ preview: str? (NEW v0.5.0)
└─ element_types: List[str]? (NEW v0.5.0)


EXTRACTION STAGE
═══════════════════════════════════════════════════════════════════
LLM Output (raw JSON)
{
  "temporal": [
    {
      "text": "March 15, 2025",
      "normalized": "2025-03-15",
      "type": "date"
    }
  ],
  "spatial": [
    {
      "text": "Perth",
      "parent_region": "Western Australia",
      "location_type": "city"
    }
  ],
  "disease": [
    {
      "text": "measles",
      "category": "infectious_disease",
      "confidence": 0.95
    }
  ]
}

            ↓ (post-processing)

MultiDimensionalResult
├─ input_text: str
├─ entities: dict
│   ├─ "temporal": [NormalizedDimensionEntity]
│   │   ├─ text: str
│   │   ├─ normalized: str (ISO 8601)
│   │   ├─ dimension_name: str
│   │   └─ confidence: float
│   │
│   ├─ "spatial": [GeocodedDimensionEntity]
│   │   ├─ text: str
│   │   ├─ latitude: float
│   │   ├─ longitude: float
│   │   ├─ dimension_name: str
│   │   └─ confidence: float
│   │
│   └─ "disease": [CategoricalDimensionEntity]
│       ├─ text: str
│       ├─ category: str
│       ├─ dimension_name: str
│       └─ confidence: float
│
├─ success: bool
├─ error: str?
├─ processing_time: float
├─ document_metadata: dict
└─ extraction_config: dict


OUTPUT STAGE
═══════════════════════════════════════════════════════════════════
Saved Files:
• data/chunks/preprocessed_chunks.json
• data/results/extraction_results.json
• data/visualizations/extraction_summary.json
• data/visualizations/*.html (maps, plots)
```

---

## 6. LLM Provider Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     LLM Provider System                         │
└────────────────────────────────────────────────────────────────┘

                        LLMManager
                            │
                    (Factory Pattern)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  OpenAILLM   │    │AnthropicLLM  │    │HuggingFaceLLM│
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ OpenAI API   │    │ Anthropic API│    │  vLLM Server │
│              │    │              │    │              │
│ • gpt-4o     │    │ • claude-3.5 │    │ • Qwen3-8B   │
│ • gpt-4o-mini│    │ • claude-3   │    │ • Custom     │
│              │    │              │    │   models     │
│ HTTPS REST   │    │ HTTPS REST   │    │ HTTP REST    │
└──────────────┘    └──────────────┘    └──────────────┘

Common Interface:
══════════════════════════════════════════════════════════
All providers implement:

generate(messages: List[dict]) → LLMResponse
    • Synchronous single generation
    • Returns: content, success, error_msg

generate_batch(message_list: List[List[dict]]) → List[LLMResponse]
    • Async batch generation
    • Parallel processing
    • Rate limiting

HuggingFaceLLM Special Features:
══════════════════════════════════════════════════════════
• Multi-server support (load balancing)
• Health checking
• Automatic failover
• Qwen3 chat template
• enable_thinking parameter (for Qwen3-Thinking models)
```

---

## 7. Geocoding Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│              GeocoderService.get_coordinates()                  │
│            (stindex/spatial/geocoder.py)                        │
└────────────────────────────────────────────────────────────────┘

Input: location, context, parent_region
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 1: Check Cache                │
│  • Load from geocode_cache.json     │
│  • Return if found                  │
└─────────────────────────────────────┘
    │ (not found)
    ▼
┌─────────────────────────────────────┐
│  STEP 2: Build Search Query         │
│  • Use parent_region if provided    │
│  • Build: "location, parent_region" │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 3: Nominatim Query            │
│  • geopy.geocoders.Nominatim        │
│  • Rate limit: 1 second             │
│  • Structured query                 │
└─────────────────────────────────────┘
    │
    ├─> SUCCESS → coords found
    │       │
    │       ▼
    │   ┌─────────────────────────┐
    │   │  STEP 4A: Validate      │
    │   │  • Check lat/lon valid  │
    │   │  • Cache result         │
    │   │  • Return coordinates   │
    │   └─────────────────────────┘
    │
    └─> FAILED → try fallbacks
            │
            ▼
        ┌─────────────────────────────────┐
        │  STEP 4B: Fallback Strategies   │
        ├─────────────────────────────────┤
        │                                  │
        │  1. Extract parent from context: │
        │     • Use spaCy NER              │
        │     • Find GPE entities          │
        │     • Retry with found region    │
        │                                  │
        │  2. Try broader search:          │
        │     • Remove specifics           │
        │     • Query just location name   │
        │                                  │
        │  3. Nearby locations:            │
        │     • Score candidate results    │
        │     • Use context co-occurrence  │
        │                                  │
        └─────────────────────────────────┘
            │
            ├─> SUCCESS → return coords
            └─> FAILED → return None

Cache Structure:
═══════════════════════════════════════════════════════════
{
  "location|parent_region": {
    "latitude": float,
    "longitude": float,
    "timestamp": str
  }
}
```

---

## 7.5. OpenStreetMap Nearby Locations (NEW v0.5.0)

```
┌────────────────────────────────────────────────────────────────┐
│        OSMContextProvider.get_nearby_locations()                │
│        (stindex/postprocess/spatial/osm_context.py)             │
└────────────────────────────────────────────────────────────────┘

Input: location (lat, lon), radius_km
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 1: Build Overpass API Query  │
│  • Query for named features         │
│  • Within radius (in meters)        │
│  • Both nodes and ways              │
│  • Limit results (default: 10)     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 2: Query Overpass API         │
│  • POST to overpass-api.de          │
│  • Timeout: 30 seconds              │
│  • Parse JSON response              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 3: Process Results            │
│  For each element:                  │
│  • Extract coordinates              │
│  • Calculate distance (geodesic)    │
│  • Calculate bearing                │
│  • Convert bearing to direction     │
│  • Extract feature type from tags   │
│  • Skip if too close (<0.1km)       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 4: Sort and Format            │
│  • Sort by distance (ascending)     │
│  • Take top N results               │
│  • Return list of POIs              │
└─────────────────────────────────────┘
    │
    ▼
Output: List[Dict]
[
  {
    "name": "Roebuck Bay",
    "distance_km": 5.2,
    "direction": "SE",
    "type": "bay",
    "osm_type": "way",
    "osm_id": 12345
  },
  ...
]

Usage in Extraction:
═══════════════════════════════════════════════════════════
• Included in ExtractionContext when enable_nearby_locations=True
• Added to spatial dimension prompts
• Helps LLM disambiguate location mentions
• Expected: 3.3x improvement (GeoLLM research)

Example Prompt Addition:
═══════════════════════════════════════════════════════════
Nearby geographic features (within 100km):
  - Roebuck Bay (bay): 5.2km SE
  - Derby (town): 220km NE
  - Port Hedland (city): 600km SW
```

---

## 8. Configuration System

```
┌────────────────────────────────────────────────────────────────┐
│               Configuration Loading System                      │
│            (stindex/utils/config.py)                            │
└────────────────────────────────────────────────────────────────┘

load_config_from_file(config_path: str)
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 1: Load Main Config           │
│  • cfg/{config_path}.yml             │
│  • Get llm_provider field            │
│                                      │
│  Example (cfg/extract.yml):          │
│  llm_provider: "hf"                  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 2: Load Provider Config       │
│  • cfg/{llm_provider}.yml            │
│                                      │
│  Examples:                           │
│  • cfg/openai.yml                    │
│  • cfg/anthropic.yml                 │
│  • cfg/hf.yml                        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 3: Merge Configurations       │
│  • Provider config overrides main   │
│  • Nested dict merge                │
│  • Keep both llm and other sections │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STEP 4: Environment Variables      │
│  • Only for API keys:                │
│    - OPENAI_API_KEY                  │
│    - ANTHROPIC_API_KEY               │
│  • Not for config values             │
└─────────────────────────────────────┘
    │
    ▼
    Merged Config Dict


Config File Hierarchy:
═══════════════════════════════════════════════════════════
cfg/
├── extract.yml          # Main config
│   └─ llm_provider: "hf"
│
├── dimensions.yml       # Dimension definitions
│   └─ dimensions: {...}
│
├── openai.yml          # OpenAI settings
│   ├─ model_name: "gpt-4o-mini"
│   ├─ temperature: 0.0
│   └─ max_tokens: 2048
│
├── anthropic.yml       # Anthropic settings
│   ├─ model_name: "claude-3-5-sonnet"
│   ├─ temperature: 0.0
│   └─ max_tokens: 2048
│
└── hf.yml              # HuggingFace settings
    ├─ model_name: "Qwen/Qwen3-8B"
    ├─ device: "auto"
    ├─ temperature: 0.0
    ├─ enable_thinking: true
    └─ server_urls: [...]
```

---

## 9. Component Interactions

```
┌────────────────────────────────────────────────────────────────┐
│                    Component Interaction Map                    │
└────────────────────────────────────────────────────────────────┘

         USER
           │
           │ (calls)
           ▼
    ┌─────────────┐
    │ STIndexPipe │
    │   line      │
    └─────────────┘
           │
           │ (creates & orchestrates)
           │
    ┌──────┴──────┬─────────────┬─────────────┐
    │             │             │             │
    ▼             ▼             ▼             ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Preprocessor│ │Dimensional│ │Geocoder  │ │Visualizer│
│          │ │ Extractor │ │ Service  │ │ (future) │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
    │             │             │
    │             │             │
    │             ▼             │
    │       ┌──────────┐        │
    │       │   LLM    │        │
    │       │ Manager  │        │
    │       └──────────┘        │
    │             │             │
    │             │             │
    │    ┌────────┴────────┐    │
    │    │                 │    │
    │    ▼                 ▼    ▼
    │ ┌──────┐  ┌──────┐  ┌──────┐
    │ │OpenAI│  │Anthro│  │ HF   │
    │ │ LLM  │  │ pic  │  │ LLM  │
    │ └──────┘  │ LLM  │  └──────┘
    │           └──────┘      │
    │                         │
    │                         ▼
    │                   ┌──────────┐
    │                   │ vLLM     │
    │                   │ Server   │
    │                   └──────────┘
    │
    │ (uses)
    ▼
┌───────────────────────────────────────┐
│  Utility Modules                      │
├───────────────────────────────────────┤
│  • Config Loader                      │
│  • JSON Extractor                     │
│  • Dimension Loader                   │
│  • Tokenizer (Qwen3 chat template)    │
└───────────────────────────────────────┘

Data Flow:
══════════════════════════════════════════════════════════
User → Pipeline → Preprocessor → DocumentChunk[]
                      ↓
                 Save to disk
                      ↓
       Pipeline → DimensionalExtractor → LLM → JSON
                      ↓
              Post-processing (per dim)
                      ↓
                 GeocoderService (for spatial)
                      ↓
              MultiDimensionalResult[]
                      ↓
                 Save to disk
                      ↓
       Pipeline → Visualizer → Maps/Plots
```

---

## 10. Visualization Module (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│            STIndexVisualizer.visualize()                         │
│         (stindex/visualization/visualizer.py)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                Input: results (list or file path)
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  Create output directory structure          │
        ├─────────────────────────────────────────────┤
        │  • data/output/visualizations/              │
        │    ├─ stindex_report_{timestamp}.html       │
        │    └─ stindex_report_{timestamp}_source/    │
        │       ├─ summary.json                        │
        │       ├─ map.html                            │
        │       └─ *.png (plots)                       │
        └─────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 1: Generate Statistical Summary                             │
│  (stindex/visualization/statistical_summary.py)                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  StatisticalSummary.generate_summary(results)                     │
│                                                                     │
│  For all results:                                                 │
│  ├─> _overview_stats()                                            │
│  │   • total_chunks, successful, failed                           │
│  │   • success_rate                                               │
│  │                                                                 │
│  ├─> _dimension_stats()                                           │
│  │   For each dimension:                                          │
│  │   • chunks_with_entities                                       │
│  │   • total_entities                                             │
│  │   • unique_count (unique values)                               │
│  │                                                                 │
│  ├─> _source_stats()                                              │
│  │   • Count by data source                                       │
│  │                                                                 │
│  ├─> _performance_stats()                                         │
│  │   • mean_time, median_time                                     │
│  │   • min_time, max_time                                         │
│  │   • total_time                                                 │
│  │                                                                 │
│  ├─> _temporal_coverage()                                         │
│  │   • earliest_date, latest_date                                 │
│  │   • date_range_days                                            │
│  │                                                                 │
│  └─> _spatial_coverage()                                          │
│      • geocoded_count, geocoding_rate                             │
│      • bounds (min/max lat/lon)                                   │
│                                                                     │
│  Output: Dict[str, Any] (saved as summary.json)                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate Statistical Plots                               │
│  (stindex/visualization/plot_generator.py)                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PlotGenerator.generate_plots(results, output_dir)                │
│                                                                     │
│  Convert results to DataFrame:                                    │
│  • _results_to_dataframe()                                        │
│    └─> Extract first entity per dimension per chunk               │
│    └─> Create columns: {dim}_text, {dim}_normalized, etc.         │
│                                                                     │
│  For each dimension in DataFrame:                                 │
│  ├─> Check dimension type                                         │
│  │                                                                 │
│  ├─> _plot_temporal_dimension()                                   │
│  │   • Parse normalized dates                                     │
│  │   • Group by year_month                                        │
│  │   • Generate bar chart (matplotlib)                            │
│  │   • Output: {dim}_temporal_distribution.png                    │
│  │                                                                 │
│  ├─> _plot_text_dimension()                                       │
│  │   • Count value occurrences                                    │
│  │   • Get top 15                                                 │
│  │   • Generate horizontal bar chart                              │
│  │   • Output: {dim}_distribution.png                             │
│  │                                                                 │
│  ├─> _plot_categorical_dimension()                                │
│  │   • Count category occurrences                                 │
│  │   • Generate bar chart                                         │
│  │   • Output: {dim}_distribution.png                             │
│  │                                                                 │
│  ├─> _plot_cross_dimensional()                                    │
│  │   • Find categorical dimensions                                │
│  │   • Create cross-tabulation                                    │
│  │   • Generate stacked bar chart                                 │
│  │   • Output: cross_{dim1}_{dim2}.png                            │
│  │                                                                 │
│  ├─> _plot_extraction_metrics()                                   │
│  │   • Count entities by dimension                                │
│  │   • Generate bar chart                                         │
│  │   • Output: extraction_metrics.png                             │
│  │                                                                 │
│  └─> _create_interactive_plots() (if plotly available)            │
│      • Parse temporal data                                        │
│      • Calculate cumulative counts                                │
│      • Generate interactive timeline                              │
│      • Output: interactive_timeline.html                          │
│                                                                     │
│  Libraries: matplotlib, seaborn, plotly                           │
│  Output: List[str] (plot file paths)                              │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 3: Generate Interactive Map                                 │
│  (stindex/visualization/map_generator.py)                         │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MapGenerator.generate_map(results, output_file)                  │
│                                                                     │
│  Extract events with spatial coordinates:                         │
│  • _extract_events()                                              │
│    For each result:                                               │
│    ├─> Get spatial entities (lat/lon)                             │
│    ├─> Link to temporal entities (timestamp)                      │
│    ├─> Link to category entities (for color)                      │
│    └─> Create event dict with all data                            │
│                                                                     │
│  Determine map mode:                                              │
│  ├─> Has temporal data? → Animated map                            │
│  └─> No temporal data? → Static map                               │
│                                                                     │
│  _create_static_map():                                            │
│  ├─> Calculate center (avg lat/lon)                               │
│  ├─> Create Folium base map                                       │
│  ├─> Add MarkerCluster                                            │
│  ├─> For each event:                                              │
│  │   • Create popup with entity details                           │
│  │   • Get color from category                                    │
│  │   • Add marker to cluster                                      │
│  └─> Add legend (if categories)                                   │
│                                                                     │
│  _create_animated_map():                                          │
│  ├─> Filter events with datetime                                  │
│  ├─> Sort events by timestamp                                     │
│  ├─> Calculate center (avg lat/lon)                               │
│  ├─> Create Folium base map                                       │
│  ├─> Build GeoJSON features:                                      │
│  │   For each event:                                              │
│  │   • Create Feature with geometry (Point)                       │
│  │   • Add time property (ISO format)                             │
│  │   • Add style (color from category)                            │
│  │   • Add popup content                                          │
│  ├─> Create TimestampedGeoJson layer                              │
│  │   • period: P1D (1 day)                                        │
│  │   • auto_play: False                                           │
│  │   • loop_button: True                                          │
│  │   • time_slider_drag_update: True                              │
│  └─> Add legend (if categories)                                   │
│                                                                     │
│  Library: folium, folium.plugins.TimestampedGeoJson               │
│  Output: str (map HTML file path)                                 │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  STEP 4: Generate HTML Report                                     │
│  (stindex/visualization/html_report.py)                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HTMLReportGenerator.generate_report()                            │
│                                                                     │
│  Prepare assets:                                                  │
│  ├─> Copy plots to source directory (if needed)                   │
│  └─> Copy map to source directory (if needed)                     │
│                                                                     │
│  Generate HTML sections:                                          │
│  ├─> _get_css()                                                   │
│  │   • Responsive layout styles                                   │
│  │   • Gradient backgrounds                                       │
│  │   • Card styles for stats                                      │
│  │   • Table styles                                               │
│  │   • Map/plot embedding styles                                  │
│  │                                                                 │
│  ├─> _overview_section(summary)                                   │
│  │   • Total chunks card                                          │
│  │   • Success rate card                                          │
│  │   • Failed extractions card                                    │
│  │                                                                 │
│  ├─> _dimensions_section(summary)                                 │
│  │   • Table with dimension stats                                 │
│  │   • Columns: Dimension, Chunks, Entities, Unique              │
│  │                                                                 │
│  ├─> _map_section(map_filename)                                   │
│  │   • Iframe embed of map.html                                   │
│  │   • 600px height                                               │
│  │                                                                 │
│  ├─> _plots_section(plot_files)                                   │
│  │   • Static plots grid (PNG images)                             │
│  │   • Interactive plots (HTML iframes)                           │
│  │                                                                 │
│  ├─> _performance_section(summary)                                │
│  │   • Mean time card                                             │
│  │   • Total time card                                            │
│  │   • Min/Max time card                                          │
│  │                                                                 │
│  └─> _data_sources_section(summary)                               │
│      • Table with source names and chunk counts                   │
│                                                                     │
│  Combine into complete HTML:                                      │
│  • Header with title and timestamp                                │
│  • Table of contents with anchors                                 │
│  • All sections in order                                          │
│  • Footer with version info                                       │
│                                                                     │
│  Save to: stindex_report_{timestamp}.html                         │
│                                                                     │
│  Output: str (report HTML file path)                              │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         OUTPUT READY
                              │
                Log summary to console:
                • Total chunks processed
                • Success rate
                • Number of dimensions
                • Number of plots
                • Map generated (yes/no)
                • Report path
```

---

## Summary

### Key Workflows

1. **Full Pipeline**: InputDocument → Preprocessing → Extraction → Visualization
2. **Preprocessing**: URL/File/Text → Scraping/Parsing → Chunking → DocumentChunk[]
3. **Extraction**: DocumentChunk → LLM → JSON → Post-processing → MultiDimensionalResult
4. **Visualization**: MultiDimensionalResult[] → Statistics → Plots → Map → HTML Report
5. **LLM Flow**: Prompt → Provider → API → JSON → Validation

### Critical Components

- **InputDocument**: Unified input model for all types
- **Preprocessor**: Orchestrates scraping, parsing, chunking
- **DocumentChunker**: Multiple strategies including element-based (NEW v0.5.0)
- **DimensionalExtractor**: Multi-dimensional extraction with LLM
- **ExtractionContext**: Context management for multi-chunk extraction (NEW v0.5.0)
- **LLMManager**: Factory for provider-specific LLM clients
- **GeocoderService**: Context-aware geocoding with caching
- **OSMContextProvider**: Nearby location queries for disambiguation (NEW v0.5.0)
- **ExtractionVerifier**: Two-pass verification for quality (NEW v0.5.0)
- **STIndexPipeline**: End-to-end orchestrator with 4 modes
- **STIndexVisualizer**: Comprehensive visualization orchestrator

### Data Models

- **InputDocument** → **ParsedDocument** → **DocumentChunk**
- **DocumentChunk** → **MultiDimensionalResult** (with typed entities)
- All stages use Pydantic models for validation

---

## 11. Context-Aware Extraction Features (NEW v0.5.0)

```
┌────────────────────────────────────────────────────────────────┐
│          Context-Aware Extraction Workflow                      │
│          (Multi-Chunk Document Processing)                      │
└────────────────────────────────────────────────────────────────┘

INITIALIZATION
═══════════════════════════════════════════════════════════
┌─────────────────────────────────────┐
│  Create ExtractionContext          │
│  • document_metadata                │
│  • enable_nearby_locations          │
│  • max_memory_refs (default: 10)   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Initialize DimensionalExtractor    │
│  • Pass extraction_context          │
│  • Load dimension configs           │
└─────────────────────────────────────┘

PROCESSING LOOP (for each chunk)
═══════════════════════════════════════════════════════════
        │
        ▼
┌─────────────────────────────────────┐
│  Update Context State               │
│  • set_chunk_position(i, total)     │
│  • section_hierarchy from chunk     │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Build Context-Aware Prompt         │
│  • Document metadata                │
│  • Prior temporal references        │
│  • Prior spatial references         │
│  • Nearby locations (if enabled)    │
│  • Section hierarchy                │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  LLM Extraction                     │
│  • With full context                │
│  • Resolves relative expressions    │
│  • Disambiguates locations          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Post-Processing                    │
│  • Temporal normalization           │
│  • Geocoding with context           │
│  • OSM nearby locations (optional)  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Update Context Memory              │
│  • Add temporal refs to window      │
│  • Add spatial refs to window       │
│  • Keep last N only (sliding)       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Two-Pass Verification (optional)   │
│  • Score relevance, accuracy        │
│  • Filter low-confidence entities   │
└─────────────────────────────────────┘
        │
        └───> Next chunk (with updated context)

BENEFITS
═══════════════════════════════════════════════════════════
✓ Element-Based Chunking:
  • +70% chunk quality improvement
  • Preserves document structure
  • Never fragments tables

✓ Temporal Context Window:
  • +50-70% accuracy on relative dates
  • "yesterday" → resolves to actual date
  • "next day" → uses prior reference

✓ Spatial Context:
  • +200-300% disambiguation accuracy (3.3x)
  • Nearby locations disambiguate mentions
  • Parent region hints improve geocoding

✓ Two-Pass Verification:
  • -40-60% false positive rate
  • Quality scores for each entity
  • Filters hallucinations

✓ Context Propagation:
  • Maintains consistency across chunks
  • References carry forward
  • Sliding window prevents overflow
```

---

## Summary

### Key Workflows

1. **Full Pipeline**: InputDocument → Preprocessing → Extraction → Visualization
2. **Preprocessing**: URL/File/Text → Scraping/Parsing → Element-Based Chunking → DocumentChunk[]
3. **Extraction**: DocumentChunk → Context-Aware LLM → JSON → Post-processing → Verification → MultiDimensionalResult
4. **Visualization**: MultiDimensionalResult[] → Statistics → Plots → Map → HTML Report
5. **LLM Flow**: Prompt → Provider → API → JSON → Validation
