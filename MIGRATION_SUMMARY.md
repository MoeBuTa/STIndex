# Legacy Code Removal - Migration Summary

## Goal
Migrate from old separate temporal/spatial extractors to unified agentic spatiotemporal extraction architecture.

## Completed ✅

### 1. Updated Core Package Interface
- **File**: `stindex/__init__.py`
- **Changes**:
  - Removed: `STIndexExtractor`, `TemporalEntity`, `SpatialEntity`, `SpatioTemporalResult`
  - Added: `ExtractionPipeline`, `ExtractionResult`, `BatchExtractionResult`
  - Updated version to `0.2.0`

### 2. Updated CLI
- **File**: `stindex/cli.py`
- **Changes**:
  - Removed all temporal-only/spatial-only extraction flags (`--temporal-only`, `--spatial-only`, `--no-geocoding`)
  - Replaced `STIndexExtractor` with `ExtractionPipeline`
  - Updated to work with dict-based entity results instead of Pydantic models
  - Simplified configuration (now just passes dict to pipeline)

### 3. Consolidated Schemas
- **File**: `stindex/agents/response/schemas.py` (NEW)
- **Contents**:
  - Moved `TemporalType`, `TemporalEntity`, `SpatialEntity` from old `stindex/models/schemas.py`
  - Kept all agent workflow models (`ExtractionObservation`, `ExtractionReasoning`, `ExtractionActionResponse`)
  - Consolidated all Pydantic models in one location

### 4. Consolidated LLM Implementation
- **File**: `stindex/agents/llm/local.py`
- **Changes**:
  - Moved entire `LocalQwenLLM` implementation from `stindex/llm/local_llm.py`
  - Includes `LocalLLMWrapper`, `StructuredOutputLLM`
  - Added simpler `LocalLLM` wrapper for agents
  - Now fully self-contained

## Remaining Work 🔧

### 1. Update All Imports
Need to update imports in these files:

**Core Files**:
- `stindex/agents/extractor.py`:
  Change `from stindex.models.schemas import ...` →
  `from stindex.agents.response.schemas import ...`

**Utils**:
- `stindex/utils/enhanced_time_normalizer.py`
  Change `from stindex.models.schemas import TemporalType` →
  `from stindex.agents.response.schemas import TemporalType`

- `stindex/utils/time_normalizer.py`
  Same change

**Examples**:
- `examples/test_local_model.py`
- `examples/quick_test_qwen.py`
  Update to use `ExtractionPipeline` instead of `STIndexExtractor`
- `examples/advanced_example.py`

**Tests**:
- `tests/test_extractor.py`
- `tests/test_improvements.py`
- `tests/test_pdf_example.py`
- `tests/test_comprehensive.py`
- `tests/test_english_suite.py`
- `tests/test_time_normalizer.py`

All need to:
1. Replace `STIndexExtractor` → `ExtractionPipeline`
2. Replace `ExtractionConfig` → plain dict config
3. Update result handling (entities are now dicts, not Pydantic models)

### 2. Remove Legacy Modules
After imports are updated, delete:
- `stindex/models/` (entire directory)
- `stindex/llm/` (entire directory)
- `stindex/prompts/` (entire directory - duplicated in `stindex/agents/prompts/`)
- `stindex/extractors/extractor.py` (the STIndexExtractor facade)
- `stindex/core/` (if not already deleted)

### 3. Update Documentation
- `CLAUDE.md` - Update architecture section, remove temporal-only/spatial-only references
- `README.md` - Update API examples
- `docs/USER_GUIDE.md` - Update usage examples
- `docs/LOCAL_MODEL_GUIDE.md` - Update to new architecture

### 4. Update __init__.py Files
- `stindex/agents/response/__init__.py` - Export schemas
- `stindex/agents/llm/__init__.py` - Export LLM classes
- `stindex/models/__init__.py` - DELETE after migration
- `stindex/llm/__init__.py` - DELETE after migration
- `stindex/prompts/__init__.py` - DELETE after migration

## Migration Strategy

**Phase 1** (Current): Core infrastructure updated ✅
**Phase 2** (Next): Update all imports systematically
**Phase 3**: Run tests, fix any issues
**Phase 4**: Delete old modules
**Phase 5**: Update documentation

## Key Architectural Changes

### Old Architecture
```
STIndexExtractor (facade)
  ├─► TemporalExtractor (separate LLM call)
  └─► SpatialExtractor (separate spaCy + geocoding)
```

### New Architecture
```
ExtractionPipeline
  └─► SpatioTemporalExtractorAgent (observe-reason-act)
        ├─► OBSERVE: Preprocess & extract context
        ├─► REASON: Single LLM call for both temporal + spatial
        └─► ACT: Normalize temporal + geocode spatial (tools)
```

## Breaking Changes for Users

1. **Import change**:
   ```python
   # Old
   from stindex import STIndexExtractor, TemporalEntity, SpatialEntity

   # New
   from stindex import ExtractionPipeline
   ```

2. **Configuration change**:
   ```python
   # Old
   from stindex.models.schemas import ExtractionConfig
   config = ExtractionConfig(model_name="Qwen/Qwen3-8B")
   extractor = STIndexExtractor(config=config)

   # New
   config = {"model_name": "Qwen/Qwen3-8B"}
   pipeline = ExtractionPipeline(config=config)
   ```

3. **Result format change**:
   ```python
   # Old
   result = extractor.extract(text)  # Returns SpatioTemporalResult
   for entity in result.temporal_entities:  # Pydantic models
       print(entity.text, entity.normalized)

   # New
   result = pipeline.extract(text)  # Returns ExtractionResult
   for entity in result.temporal_entities:  # Dicts
       print(entity['text'], entity['normalized'])
   ```

4. **CLI changes**:
   ```bash
   # Old
   stindex extract "text" --temporal-only
   stindex extract "text" --spatial-only

   # New (unified extraction only)
   stindex extract "text"
   ```

## Notes

- The new architecture does **unified** spatiotemporal extraction only
- No more separate temporal-only or spatial-only modes
- Single LLM call extracts both types of entities
- Tools (normalize_temporal, geocode_location) handle postprocessing
- More efficient and follows modern agentic patterns
