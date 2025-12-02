"""
STIndex RAG Module.

Provides a complete RAG (Retrieval-Augmented Generation) pipeline
for question answering using:
- STIndex dimensional extraction and filtering
- FAISS vector similarity search
- LLM-based answer generation (OpenAI, Anthropic)

Usage:
    from rag import RAGPipeline, RAGConfig

    # Initialize pipeline
    pipeline = RAGPipeline(config=RAGConfig(
        vector_index_path="data/vector/rag",
        warehouse_path="data/warehouse/rag",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    ))

    # Simple QA
    result = pipeline.answer("What is the capital of France?")
    print(result.answer)

    # With temporal filtering
    result = pipeline.answer(
        "What events occurred?",
        temporal_filter={"year": 2022},
    )

    # Multi-hop retrieval
    result = pipeline.answer(
        "Who directed the movie that won Best Picture in 2020?",
        use_multihop=True,
    )

Components:
    - RAGPipeline: End-to-end pipeline orchestrating retrieval and generation
    - RAGRetriever: High-level retriever with multi-hop support
    - RAGGenerator: LLM-based answer generator
    - RAGConfig: Pipeline configuration

Data Pipeline:
    1. Preprocess: rag/preprocess/corpus - Extract and chunk documents
    2. STIndex Ingestion: rag/preprocess/ingestion/stindex_ingest.py
    3. Vector Ingestion: rag/preprocess/ingestion/vector_ingest.py
    4. Retrieve & Generate: RAGPipeline
"""

# Pipeline API
from .pipeline import RAGPipeline, RAGConfig, RAGResult

# Retriever API
from .retriever import RAGRetriever, RAGRetrieverConfig, RetrievalContext

# Generator API
from .generator import RAGGenerator, GeneratorConfig, GenerationResult

__all__ = [
    # Main Pipeline
    "RAGPipeline",
    "RAGConfig",
    "RAGResult",
    # Retriever
    "RAGRetriever",
    "RAGRetrieverConfig",
    "RetrievalContext",
    # Generator
    "RAGGenerator",
    "GeneratorConfig",
    "GenerationResult",
]

__version__ = "0.1.0"
