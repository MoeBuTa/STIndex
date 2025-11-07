"""
STIndex v0.4.0 Usage Examples

This file demonstrates the new generic preprocessing and pipeline features.
"""

# ===== Example 1: Basic Preprocessing =====

from stindex import InputDocument, Preprocessor

# Create input documents from different sources
docs = [
    # Web URL
    InputDocument.from_url(
        url="https://example.com/news/article",
        metadata={"source": "news", "category": "health"}
    ),

    # Local file
    InputDocument.from_file(
        file_path="/path/to/document.pdf",
        metadata={"source": "reports", "year": 2025}
    ),

    # Raw text
    InputDocument.from_text(
        text="On March 15, 2025, a measles outbreak occurred in Perth, Australia.",
        title="Measles Alert",
        metadata={"publication_date": "2025-03-15", "region": "Australia"}
    ),
]

# Initialize preprocessor
preprocessor = Preprocessor(
    max_chunk_size=2000,
    chunk_overlap=200,
    chunking_strategy="paragraph",  # or "sliding_window", "semantic"
    parsing_method="unstructured"   # or "simple"
)

# Process documents
all_chunks = preprocessor.process_batch(docs)

print(f"Processed {len(all_chunks)} documents")
for i, chunks in enumerate(all_chunks):
    print(f"  Document {i}: {len(chunks)} chunks")


# ===== Example 2: Full Pipeline (Recommended) =====

from stindex import InputDocument, STIndexPipeline

# Create input documents
docs = [
    InputDocument.from_url("https://health.gov/alerts/measles-2025"),
    InputDocument.from_file("/path/to/health_report.pdf"),
]

# Initialize pipeline
pipeline = STIndexPipeline(
    # Extraction config
    extractor_config="extract",           # LLM provider config
    dimension_config="dimensions",        # Dimension config (temporal + spatial)

    # Preprocessing config
    max_chunk_size=2000,
    chunking_strategy="paragraph",

    # Output config
    output_dir="data/output",
    save_intermediate=True
)

# Run full pipeline: preprocessing → extraction → visualization
results = pipeline.run_pipeline(
    input_docs=docs,
    save_results=True,
    visualize=True
)

print(f"Pipeline complete! Processed {len(results)} chunks")


# ===== Example 3: Modular Execution =====

from stindex import STIndexPipeline

pipeline = STIndexPipeline(
    dimension_config="health_dimensions",
    output_dir="data/output"
)

# Step 1: Preprocessing only
print("Step 1: Preprocessing...")
all_chunks = pipeline.run_preprocessing(docs, save_chunks=True)
# Saves to: data/output/chunks/preprocessed_chunks.json

# Step 2: Extraction only (can run later or on different machine)
print("Step 2: Extraction...")
chunks = pipeline.load_chunks_from_file("data/output/chunks/preprocessed_chunks.json")
results = pipeline.run_extraction(chunks, save_results=True)
# Saves to: data/output/results/extraction_results.json

# Step 3: Visualization only
print("Step 3: Visualization...")
pipeline.run_visualization(
    results="data/output/results/extraction_results.json",
    output_dir="data/output/visualizations"
)


# ===== Example 4: Custom Dimensions =====

from stindex import InputDocument, STIndexPipeline

# Create dimension config: cfg/health_dimensions.yml
"""
dimensions:
  temporal:
    enabled: true
    extraction_type: "normalized"
    description: "Extract temporal expressions"

  spatial:
    enabled: true
    extraction_type: "geocoded"
    description: "Extract and geocode locations"

  disease:
    enabled: true
    extraction_type: "categorical"
    description: "Extract disease mentions"
    categories:
      - infectious_disease
      - chronic_disease
      - mental_health

  event_type:
    enabled: true
    extraction_type: "categorical"
    description: "Extract event types"
    categories:
      - outbreak
      - alert
      - report
"""

# Use custom dimensions
pipeline = STIndexPipeline(
    dimension_config="health_dimensions",
    output_dir="data/output"
)

docs = [InputDocument.from_text("Health alert text...")]
results = pipeline.run_pipeline(docs)

# Access extracted dimensions
for result in results:
    if result['extraction']['success']:
        entities = result['extraction']['entities']
        print(f"Temporal: {len(entities.get('temporal', []))}")
        print(f"Spatial: {len(entities.get('spatial', []))}")
        print(f"Disease: {len(entities.get('disease', []))}")
        print(f"Event Type: {len(entities.get('event_type', []))}")


# ===== Example 5: Case Study Pattern =====

"""
Recommended pattern for new case studies:

1. Create input documents
2. (Optional) Create custom dimension config
3. Run pipeline
4. Analyze results
"""

def run_case_study():
    from stindex import InputDocument, STIndexPipeline

    # 1. Define input sources
    urls = [
        "https://health.gov/alerts/alert1",
        "https://health.gov/alerts/alert2",
    ]

    docs = [
        InputDocument.from_url(url, metadata={"source": "health_gov"})
        for url in urls
    ]

    # 2. Configure pipeline
    pipeline = STIndexPipeline(
        dimension_config="case_studies/my_study/config/dimensions",
        output_dir="case_studies/my_study/data"
    )

    # 3. Run
    results = pipeline.run_pipeline(docs, visualize=True)

    # 4. Analyze
    print(f"Processed {len(results)} chunks")
    success_count = sum(1 for r in results if r['extraction']['success'])
    print(f"Success rate: {success_count}/{len(results)}")

    return results


# ===== Example 6: Error Handling =====

from stindex import InputDocument, STIndexPipeline

# Preprocessing handles errors gracefully
docs = [
    InputDocument.from_url("https://valid-url.com"),
    InputDocument.from_url("https://invalid-404.com"),  # Will fail
    InputDocument.from_file("/path/to/missing.pdf"),    # Will fail
]

pipeline = STIndexPipeline()

# Process continues even if some documents fail
all_chunks = pipeline.run_preprocessing(docs)
print(f"Successfully processed {len(all_chunks)}/{len(docs)} documents")


# ===== Example 7: CLI Usage =====

"""
# Full pipeline
python case_studies/public_health/scripts/run_case_study.py \\
    --mode pipeline \\
    --input-mode url \\
    --visualize

# Preprocessing only
python case_studies/public_health/scripts/run_case_study.py \\
    --mode preprocessing \\
    --input-mode file

# Extraction only
python case_studies/public_health/scripts/run_case_study.py \\
    --mode extraction \\
    --chunks-file data/chunks/preprocessed_chunks.json

# Visualization only
python case_studies/public_health/scripts/run_case_study.py \\
    --mode visualization \\
    --results-file data/results/extraction_results.json
"""


# ===== Example 8: Resume from Checkpoint =====

from stindex import STIndexPipeline

pipeline = STIndexPipeline(
    output_dir="data/output",
    save_intermediate=True
)

# First run (interrupted)
try:
    docs = [InputDocument.from_url(f"https://example.com/{i}") for i in range(100)]
    results = pipeline.run_pipeline(docs)
except KeyboardInterrupt:
    print("Interrupted! Chunks saved to: data/output/chunks/")

# Resume from checkpoint
chunks = pipeline.load_chunks_from_file("data/output/chunks/preprocessed_chunks.json")
results = pipeline.run_extraction(chunks, save_results=True)
