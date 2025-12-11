"""
STIndex RAG Pipeline.

End-to-end RAG pipeline combining:
- STIndex dimensional extraction and filtering
- FAISS vector retrieval
- LLM-based answer generation

Usage:
    from rag import RAGPipeline, RAGConfig

    pipeline = RAGPipeline(config=RAGConfig(
        vector_index_path="data/vector/rag",
        warehouse_path="data/warehouse/rag",
    ))

    # Simple QA
    answer = pipeline.answer("What happened in Paris in 2022?")

    # With temporal filtering
    answer = pipeline.answer(
        "What events occurred?",
        temporal_filter={"year": 2022},
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .retriever import RAGRetriever, RAGRetrieverConfig, RetrievalContext
from .generator import RAGGenerator, GeneratorConfig, GenerationResult


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    # Retriever settings
    vector_index_path: str = "data/vector/rag"
    warehouse_path: Optional[str] = "data/warehouse/rag"
    retriever_k: int = 5
    max_context_length: int = 4000
    enable_multihop: bool = False
    max_hops: int = 2

    # Generator settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512

    # Pipeline settings
    include_citations: bool = True
    verbose: bool = False


@dataclass
class RAGResult:
    """Result of RAG pipeline."""
    answer: str
    question: str
    context: RetrievalContext
    generation: GenerationResult
    pipeline_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "question": self.question,
            "num_chunks_retrieved": len(self.context.chunks),
            "model": self.generation.model,
            "usage": self.generation.usage,
            "citations": self.generation.citations,
        }


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Orchestrates retrieval and generation for question answering.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        vector_index_path: Optional[str] = None,
        warehouse_path: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            config: Full configuration object
            vector_index_path: Override vector index path
            warehouse_path: Override warehouse path
            llm_provider: Override LLM provider
            llm_model: Override LLM model
        """
        self.config = config or RAGConfig()

        # Override settings if provided
        if vector_index_path:
            self.config.vector_index_path = vector_index_path
        if warehouse_path:
            self.config.warehouse_path = warehouse_path
        if llm_provider:
            self.config.llm_provider = llm_provider
        if llm_model:
            self.config.llm_model = llm_model

        # Initialize components
        self._init_retriever()
        self._init_generator()

        logger.info(f"RAGPipeline initialized")

    def _init_retriever(self) -> None:
        """Initialize retriever component."""
        retriever_config = RAGRetrieverConfig(
            vector_index_path=self.config.vector_index_path,
            warehouse_path=self.config.warehouse_path,
            default_k=self.config.retriever_k,
            max_context_length=self.config.max_context_length,
            max_hops=self.config.max_hops,
        )
        self.retriever = RAGRetriever(config=retriever_config)

    def _init_generator(self) -> None:
        """Initialize generator component."""
        generator_config = GeneratorConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            include_citations=self.config.include_citations,
        )
        self.generator = RAGGenerator(config=generator_config)

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        temporal_filter: Optional[Dict[str, Any]] = None,
        spatial_filter: Optional[Dict[str, Any]] = None,
        use_multihop: Optional[bool] = None,
    ) -> RAGResult:
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            k: Number of chunks to retrieve
            temporal_filter: Optional temporal filter
            spatial_filter: Optional spatial filter
            use_multihop: Whether to use multi-hop retrieval

        Returns:
            RAGResult with answer and metadata
        """
        k = k or self.config.retriever_k
        use_multihop = use_multihop if use_multihop is not None else self.config.enable_multihop

        # Retrieve context
        if use_multihop:
            context = self.retriever.retrieve_multihop(
                question=question,
                max_hops=self.config.max_hops,
                temporal_filter=temporal_filter,
                spatial_filter=spatial_filter,
            )
        else:
            context = self.retriever.retrieve(
                query=question,
                k=k,
                temporal_filter=temporal_filter,
                spatial_filter=spatial_filter,
            )

        # Format context for generator
        formatted_context = context.get_formatted_context(
            include_titles=True,
            max_length=self.config.max_context_length,
        )

        # Generate answer
        generation = self.generator.generate(
            question=question,
            context=formatted_context,
        )

        return RAGResult(
            answer=generation.answer,
            question=question,
            context=context,
            generation=generation,
            pipeline_stats={
                "chunks_retrieved": len(context.chunks),
                "hops": context.hops,
                "context_length": len(formatted_context),
            },
        )

    def answer_batch(
        self,
        questions: List[str],
        k: Optional[int] = None,
        temporal_filter: Optional[Dict[str, Any]] = None,
        spatial_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RAGResult]:
        """
        Answer multiple questions.

        Args:
            questions: List of questions
            k: Number of chunks per question
            temporal_filter: Shared temporal filter
            spatial_filter: Shared spatial filter

        Returns:
            List of RAGResult
        """
        results = []
        for question in questions:
            try:
                result = self.answer(
                    question=question,
                    k=k,
                    temporal_filter=temporal_filter,
                    spatial_filter=spatial_filter,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to answer: {question[:50]}... - {e}")
                # Create error result
                results.append(RAGResult(
                    answer=f"Error: {str(e)}",
                    question=question,
                    context=RetrievalContext(query=question, chunks=[], total_candidates=0),
                    generation=GenerationResult(
                        answer=f"Error: {str(e)}",
                        question=question,
                        context_used="",
                        model=self.config.llm_model,
                        provider=self.config.llm_provider,
                    ),
                ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "config": {
                "vector_index_path": self.config.vector_index_path,
                "warehouse_path": self.config.warehouse_path,
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "retriever_k": self.config.retriever_k,
            },
            "retriever_stats": self.retriever.get_stats(),
        }
