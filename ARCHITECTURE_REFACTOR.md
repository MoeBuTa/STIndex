# STIndex Refactored Architecture Design

## Overview

Refactored STIndex to follow MetacogRAG's **Observe-Reason-Act** agentic pattern with:
- Unified spatiotemporal extraction (single LLM call)
- Tool calling for postprocessing (geocoding, normalization)
- LangChain integration for simplicity
- Research-based best practices from 2025

## Architecture Comparison

### Before (Old)
```
STIndexExtractor
├── TemporalExtractor (separate LLM call)
│   └── EnhancedTimeNormalizer
└── SpatialExtractor (spaCy NER)
    └── EnhancedGeocoderService
```

### After (New - MetacogRAG-inspired)
```
stindex/
├── agents/
│   ├── base.py                    # BaseAgent (observe-reason-act)
│   ├── extractor.py               # SpatioTemporalExtractorAgent
│   ├── llm/
│   │   ├── local.py               # LocalLLM wrapper
│   │   └── api.py                 # OpenAI/Anthropic wrapper
│   ├── prompts/
│   │   ├── base.py                # Base prompt templates
│   │   └── extraction.py          # Extraction prompts
│   └── response/
│       └── models.py              # Response models (Pydantic)
├── pipeline/
│   ├── models.py                  # Pipeline data models
│   └── extraction_pipeline.py    # Main pipeline
├── tools/
│   ├── temporal.py                # Temporal normalization tool
│   ├── spatial.py                 # Geocoding tool
│   └── registry.py                # Tool registry
├── utils/
│   ├── preprocessing.py           # Text preprocessing
│   └── config.py                  # Configuration
└── cli/
    └── run.py                     # CLI interface
```

## Key Design Patterns

### 1. Observe-Reason-Act Pattern

Based on ReAct (Yao et al., 2023) and MetacogRAG:

```python
class SpatioTemporalExtractorAgent(BaseAgent):
    def observe(self, environment: Dict) -> Observation:
        """
        OBSERVE: Preprocess text and extract context
        - Clean text
        - Extract document-level temporal context (years)
        - Extract document-level spatial context (regions)
        """

    def reason(self, observations: Observation) -> Reasoning:
        """
        REASON: Single LLM call to extract both temporal and spatial
        - Input: cleaned text + context
        - Output: {temporal_mentions: [...], spatial_mentions: [...]}
        """

    def act(self, reasoning: Reasoning) -> ActionResponse:
        """
        ACT: Postprocess using tools
        - normalize_temporal: ISO 8601 conversion
        - geocode_location: Get lat/lon from Nominatim
        - Return structured TemporalEntity and SpatialEntity objects
        """
```

### 2. Tool Calling Best Practices (2025 Research)

#### Structured Output First
```python
# Use with_structured_output() for API models (OpenAI/Anthropic)
structured_llm = self.llm.with_structured_output(UnifiedExtractionResult)

# For local models, use prompt engineering + JSON parsing
result = self.llm.generate_structured(prompt)
```

#### Tool Design Principles
1. **Simple, narrowly scoped tools** - Each tool does ONE thing well
2. **Well-chosen names and descriptions** - LLM can understand when to use
3. **Explicit parameters** - Clear type hints and descriptions

```python
Tool(
    name="normalize_temporal",
    description="Normalize temporal expression to ISO 8601. Use for dates/times.",
    parameters=[
        ToolParameter(name="temporal_expression", type="string", required=True),
        ToolParameter(name="context", type="string", required=False)
    ]
)
```

### 3. Context Engineering

Based on recent spatiotemporal extraction research (2024-2025):

#### Document-Level Context Extraction
```python
# PREPROCESSING: Extract context BEFORE LLM call
temporal_context = {
    "mentioned_years": [2022, 2021],  # For year inference
    "has_relative_dates": True
}

spatial_context = {
    "regions": ["Western Australia"],  # For disambiguation
    "countries": ["Australia"]
}
```

#### Context-Aware Postprocessing
```python
# ACT phase: Use context for normalization
normalize_temporal(
    "March 15",
    document_years=[2022],  # Infer 2022-03-15
)

geocode_location(
    "Broome",
    parent_region="Western Australia"  # Disambiguate to correct Broome
)
```

### 4. Response Models (Pydantic)

Following LangChain best practices:

```python
class Observation(BaseModel):
    """What the agent observes."""
    cleaned_text: str
    temporal_context: Dict
    spatial_context: Dict

class Reasoning(BaseModel):
    """LLM reasoning output."""
    temporal_mentions: List[TemporalMention]
    spatial_mentions: List[SpatialMention]
    raw_output: str

class ActionResponse(BaseModel):
    """Final structured output."""
    temporal_entities: List[TemporalEntity]
    spatial_entities: List[SpatialEntity]
    success: bool
    metadata: Dict
```

## Workflow

```
┌─────────────────────────────────────────┐
│  1. OBSERVE (Preprocessing)              │
│  - TextPreprocessor.preprocess()         │
│  - Extract years, regions from document  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  2. REASON (Unified LLM Extraction)      │
│  - Single LLM call                       │
│  - Extract temporal + spatial together   │
│  - Structured output                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  3. ACT (Tool-based Postprocessing)      │
│  ┌─────────────────────────────────────┐│
│  │ Temporal: normalize_temporal tool   ││
│  │ - Context-aware year inference      ││
│  │ - ISO 8601 formatting               ││
│  └─────────────────────────────────────┘│
│  ┌─────────────────────────────────────┐│
│  │ Spatial: geocode_location tool      ││
│  │ - Nominatim API                     ││
│  │ - Context-aware disambiguation      ││
│  └─────────────────────────────────────┘│
└──────────────┬──────────────────────────┘
               │
               ▼
         ActionResponse
         (TemporalEntity + SpatialEntity)
```

## Benefits of New Architecture

### 1. Single LLM Call (Efficiency)
- **Before**: 2 separate LLM calls (temporal + spatial)
- **After**: 1 unified call
- **Savings**: ~50% LLM cost + faster processing

### 2. Agentic Pattern (Flexibility)
- Follows proven observe-reason-act pattern
- Easy to extend with new agents (e.g., EventExtractor, RelationExtractor)
- State management for complex workflows

### 3. Tool Calling (Accuracy)
- LLM for entity **detection** (what it's good at)
- APIs for **normalization/geocoding** (authoritative sources)
- Best of both worlds

### 4. Research-Based (2025)
- ReAct pattern (Yao et al., 2023)
- Structured outputs (LangChain 2025)
- Context engineering (spatiotemporal extraction research 2024-2025)
- Process calling for stateful operations

### 5. Modular Design (Maintainability)
- Clear separation of concerns
- Each module has single responsibility
- Easy to test and debug
- Mimics MetacogRAG's proven structure

## Implementation Priority

1. ✅ **Core modules** (preprocessing, tools)
2. ⏳ **Base agent + response models**
3. ⏳ **SpatioTemporalExtractorAgent**
4. ⏳ **ExtractionPipeline**
5. ⏳ **CLI updates**
6. ⏳ **Tests**

## Migration Path

### Phase 1: New architecture (parallel to old)
- Keep old extractors working
- Implement new architecture in `stindex/agents/`
- Add feature flag: `STINDEX_USE_AGENTIC=true`

### Phase 2: Migration
- Update tests to use new architecture
- Deprecate old extractors
- Update documentation

### Phase 3: Cleanup
- Remove old extractors
- Clean up imports
- Final optimization

## Research Citations

1. **ReAct Pattern**: Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"
2. **Tool Calling**: LangChain (2025) "How to do tool/function calling"
3. **Spatiotemporal Extraction**: Tian et al. (2025) "Advancing Large Language Models for Spatiotemporal and Semantic Association Mining"
4. **Process Calling**: Rasa (2025) "Process Calling: Agentic Tools Need State"
5. **Structured Output**: OpenAI/LangChain (2025) "Structured Outputs Documentation"
