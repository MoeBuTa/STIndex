# STIndex Tests

This directory contains all test scripts for the STIndex project.

## Test Structure

### Unit Tests (pytest)
- `test_extractor.py` - Integration tests for main extraction pipeline

Run with: `pytest` or `pytest tests/`

### Integration Test Scripts
These scripts perform comprehensive testing and save results to `data/output/`:

- `test_comprehensive.py` - Comprehensive capability test with 40+ test cases
- `test_english_suite.py` - Pure English test suite evaluation
- `test_improvements.py` - Tests for research-based improvements (year inference, geocoding)
- `test_pdf_example.py` - Validation test for the PDF example

Run individually:
```bash
python tests/test_comprehensive.py
python tests/test_english_suite.py
python tests/test_improvements.py
python tests/test_pdf_example.py
```

## Output Location

All integration test results are saved to `data/output/` with timestamped filenames:
- JSON format: Full test results with all entities extracted
- TXT format: Summary of test results

The `data/` folder is gitignored to keep test outputs out of version control.

## Notes

- Integration tests may require API keys or local LLM models depending on configuration
- Check `CLAUDE.md` for detailed testing commands and approaches
