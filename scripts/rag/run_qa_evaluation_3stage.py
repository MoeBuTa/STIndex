#!/usr/bin/env python3
"""
QA Evaluation using Three-Stage Retriever.

Evaluates question answering on filtered MIRAGE questions using:
1. Three-Stage Retriever (BGE-M3 dense + sparse + reranker)
2. LLM-based answer generation

Usage:
    python scripts/rag/run_qa_evaluation_3stage.py \
        --questions data/evaluation/filtered_mirage_questions.jsonl \
        --index-path data/indices/textbook_bgem3 \
        --sample-limit 100

With dimensional filtering:
    python scripts/rag/run_qa_evaluation_3stage.py \
        --dimension-index data/indices/textbook_bgem3/indexes \
        --dimension-filter "drug:insulin"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.retriever.three_stage_retriever import ThreeStageRetriever


class ThreeStageQAEvaluator:
    """
    QA evaluator using Three-Stage Retriever.
    """

    def __init__(
        self,
        questions_file: str,
        index_path: str,
        dimension_index_path: Optional[str] = None,
        output_dir: str = "data/output/evaluations/qa_3stage",
        k: int = 5,
        stage1_k: int = 100,
        stage2_k: int = 30,
        llm_model: str = "Qwen3-4B-Instruct-2507",
        llm_base_url: str = "http://localhost:8001/v1",
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 256,
        device: str = "cuda",
        load_reranker: bool = True,
    ):
        """
        Initialize QA evaluator.

        Args:
            questions_file: Path to filtered MIRAGE questions JSONL
            index_path: Path to BGE-M3 index directory
            dimension_index_path: Optional path to dimension index
            output_dir: Output directory for evaluation results
            k: Final number of documents to retrieve
            stage1_k: Documents from dense stage
            stage2_k: Documents after sparse reranking
            llm_model: LLM model name for answer generation
            llm_base_url: MS-SWIFT server base URL
            llm_temperature: LLM sampling temperature
            llm_max_tokens: Maximum tokens for LLM response
            device: Device for models
            load_reranker: Whether to load the cross-encoder reranker
        """
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.k = k
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load questions
        logger.info(f"Loading questions from {self.questions_file}")
        self.questions = self._load_questions()
        logger.info(f"Loaded {len(self.questions)} questions")

        # Initialize retriever
        logger.info("Initializing Three-Stage Retriever...")
        self.retriever = ThreeStageRetriever(
            index_path=index_path,
            dimension_index_path=dimension_index_path,
            device=device,
            load_sparse=True,
            load_reranker=load_reranker,
            load_dimensions=dimension_index_path is not None,
        )
        logger.info("✓ Retriever ready")

        # Initialize LLM client for answer generation
        logger.info(f"Initializing LLM client: {llm_model}")
        self.llm_client = OpenAI(base_url=llm_base_url, api_key="EMPTY")
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        logger.info("✓ LLM client ready")

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from JSONL file."""
        questions = []
        with open(self.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        return questions

    def _extract_answer_from_options(
        self,
        question: Dict[str, Any],
        retrieved_chunks: List[str],
    ) -> str:
        """
        Extract answer using LLM-based generation with retrieved context.

        Args:
            question: Question dictionary with 'metadata.options'
            retrieved_chunks: List of retrieved text chunks

        Returns:
            Selected option letter (A, B, C, D)
        """
        options = question.get("metadata", {}).get("options", {})
        if not options:
            return ""

        question_text = question.get("question", "")

        # Combine retrieved context (use top 3 chunks)
        context = "\n\n".join(retrieved_chunks[:3])

        # Format options for prompt
        options_text = "\n".join([f"{letter}) {text}" for letter, text in sorted(options.items())])

        # Create prompt
        prompt = f"""Context:
{context}

Question: {question_text}

Options:
{options_text}

Based on the context provided, answer with the correct option letter (A, B, C, or D) only. Do not provide any explanation."""

        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a medical question answering assistant. Answer multiple choice questions by selecting the correct option letter based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )

            # Extract answer from response
            answer = response.choices[0].message.content.strip()

            # Extract option letter (A, B, C, or D)
            for char in answer.upper():
                if char in ['A', 'B', 'C', 'D']:
                    return char

            # If no valid option found, return empty
            logger.warning(f"Could not extract valid option from LLM response: {answer}")
            return ""

        except Exception as e:
            logger.error(f"Error calling LLM for answer generation: {e}")
            return ""

    def evaluate_question(
        self,
        question: Dict[str, Any],
        dimension_filters: Optional[List[Dict]] = None,
        dimension_mode: str = "boost",
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.

        Args:
            question: Question dictionary
            dimension_filters: Optional dimensional filters
            dimension_mode: "prefilter", "boost", or "postfilter"

        Returns:
            Evaluation result dictionary
        """
        question_id = question["id"]
        question_text = question["question"]
        correct_option = question.get("metadata", {}).get("correct_option", "")
        gold_answer = question.get("answer_labels", [""])[0]

        # Retrieve documents
        retrieval_results = self.retriever.retrieve(
            query=question_text,
            k=self.k,
            stage1_k=self.stage1_k,
            stage2_k=self.stage2_k,
            dimension_filters=dimension_filters,
            dimension_mode=dimension_mode,
        )

        # Extract retrieved chunks
        retrieved_chunks = [r.text for r in retrieval_results]

        # Get predicted answer using LLM
        predicted_option = self._extract_answer_from_options(question, retrieved_chunks)

        # Check correctness
        correct = (predicted_option == correct_option)

        return {
            "question_id": question_id,
            "question_text": question_text,
            "gold_answer": gold_answer,
            "correct_option": correct_option,
            "predicted_option": predicted_option,
            "predicted_answer": question.get("metadata", {}).get("options", {}).get(predicted_option, ""),
            "correct": correct,
            "retrieved_chunks": len(retrieved_chunks),
            "top_chunk": retrieved_chunks[0] if retrieved_chunks else "",
            "retrieval_scores": [r.final_score for r in retrieval_results],
        }

    def evaluate(
        self,
        sample_limit: Optional[int] = None,
        dimension_filters: Optional[List[Dict]] = None,
        dimension_mode: str = "boost",
    ) -> Dict[str, Any]:
        """
        Run full QA evaluation.

        Args:
            sample_limit: Limit number of questions to evaluate
            dimension_filters: Optional dimensional filters
            dimension_mode: "prefilter", "boost", or "postfilter"

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting QA evaluation with Three-Stage Retriever")

        # Limit sample size if specified
        questions_to_eval = self.questions[:sample_limit] if sample_limit else self.questions
        logger.info(f"Evaluating {len(questions_to_eval)} questions")

        # Evaluate each question
        results = []
        correct_count = 0

        for question in tqdm(questions_to_eval, desc="Evaluating questions"):
            result = self.evaluate_question(
                question,
                dimension_filters=dimension_filters,
                dimension_mode=dimension_mode,
            )
            results.append(result)
            if result["correct"]:
                correct_count += 1

        # Compute accuracy
        accuracy = correct_count / len(results) if results else 0.0

        # Generate reports
        detailed_df = pd.DataFrame(results)
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "retrieval_k": self.k,
            "stage1_k": self.stage1_k,
            "stage2_k": self.stage2_k,
            "correct_count": correct_count,
            "accuracy": round(accuracy, 4),
            "dimension_mode": dimension_mode,
            "dimension_filters": dimension_filters,
        }

        # Save reports
        output_files = self._save_reports(detailed_df, summary)

        logger.success("QA evaluation complete!")

        # Print summary
        self._print_summary(summary, output_files)

        return {
            "accuracy": accuracy,
            "total_questions": len(results),
            "correct_count": correct_count,
            "output_files": {k: str(v) for k, v in output_files.items()},
        }

    def _save_reports(
        self,
        detailed_df: pd.DataFrame,
        summary: Dict[str, Any],
    ) -> Dict[str, Path]:
        """Save evaluation reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed CSV
        csv_path = self.output_dir / f"qa_3stage_{timestamp}.csv"
        detailed_df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed report: {csv_path}")

        # Save summary JSON
        json_path = self.output_dir / f"qa_3stage_{timestamp}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary report: {json_path}")

        return {"csv": csv_path, "json": json_path}

    def _print_summary(
        self,
        summary: Dict[str, Any],
        output_files: Dict[str, Path],
    ):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("QA EVALUATION SUMMARY (Three-Stage Retriever)")
        print("=" * 60)
        print(f"\nTotal Questions: {summary['total_questions']}")
        print(f"Correct: {summary['correct_count']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"\nRetrieval Settings:")
        print(f"  Final K: {summary['retrieval_k']}")
        print(f"  Stage 1 (Dense): {summary['stage1_k']}")
        print(f"  Stage 2 (Sparse): {summary['stage2_k']}")
        if summary.get("dimension_filters"):
            print(f"  Dimension Mode: {summary['dimension_mode']}")
            print(f"  Dimension Filters: {summary['dimension_filters']}")
        print(f"\nReports saved to:")
        print(f"  CSV: {output_files['csv']}")
        print(f"  JSON: {output_files['json']}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="QA evaluation using Three-Stage Retriever"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/evaluation/filtered_mirage_questions.jsonl",
        help="Path to filtered MIRAGE questions"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/indices/textbook_bgem3",
        help="Path to BGE-M3 index directory"
    )
    parser.add_argument(
        "--dimension-index",
        type=str,
        default=None,
        help="Path to dimension index directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/evaluations/qa_3stage",
        help="Output directory"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Final number of documents to retrieve"
    )
    parser.add_argument(
        "--stage1-k",
        type=int,
        default=100,
        help="Documents from dense stage"
    )
    parser.add_argument(
        "--stage2-k",
        type=int,
        default=30,
        help="Documents after sparse reranking"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen3-4B-Instruct-2507",
        help="LLM model name for answer generation"
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="http://localhost:8001/v1",
        help="MS-SWIFT server base URL"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for retrieval models"
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranker (faster, less accurate)"
    )
    parser.add_argument(
        "--dimension-filter",
        type=str,
        action="append",
        help="Dimension filter in format 'dimension:value1,value2'"
    )
    parser.add_argument(
        "--dimension-mode",
        type=str,
        default="boost",
        choices=["prefilter", "boost", "postfilter"],
        help="Dimension filtering mode"
    )

    args = parser.parse_args()

    # Parse dimension filters
    dimension_filters = None
    if args.dimension_filter:
        dimension_filters = []
        for f in args.dimension_filter:
            parts = f.split(":", 1)
            if len(parts) == 2:
                dim_name = parts[0].strip()
                values = [v.strip() for v in parts[1].split(",")]
                dimension_filters.append({"dimension": dim_name, "values": values})

    # Run evaluation
    evaluator = ThreeStageQAEvaluator(
        questions_file=args.questions,
        index_path=args.index_path,
        dimension_index_path=args.dimension_index,
        output_dir=args.output,
        k=args.k,
        stage1_k=args.stage1_k,
        stage2_k=args.stage2_k,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        device=args.device,
        load_reranker=not args.no_reranker,
    )

    evaluator.evaluate(
        sample_limit=args.sample_limit,
        dimension_filters=dimension_filters,
        dimension_mode=args.dimension_mode,
    )


if __name__ == "__main__":
    main()
