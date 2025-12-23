"""
Unified evaluation pipeline for STIndex.

Runs all three evaluation dimensions:
1. Efficiency: Timing overhead, throughput, latency
2. Retrieval: Recall@k, Precision@k, MRR, NDCG (requires relevance judgments)
3. QA: Exact match, Token F1, Answer coverage

Usage:
    # Run all evaluations
    python -m stindex.eval.unified_evaluation_pipeline

    # Run specific evaluations
    python -m stindex.eval.unified_evaluation_pipeline --efficiency --qa

    # With custom configs
    python -m stindex.eval.unified_evaluation_pipeline \\
        --qa-questions data/evaluation/filtered_mirage_questions.jsonl \\
        --qa-sample-limit 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

# Import evaluation modules
from stindex.eval.efficiency_evaluation import run_efficiency_evaluation
from rag.eval.qa_evaluation import run_qa_evaluation


class UnifiedEvaluator:
    """
    Unified evaluator running all evaluation dimensions.
    """

    def __init__(
        self,
        output_dir: str = "data/output/evaluations",
        run_efficiency: bool = True,
        run_qa: bool = True,
    ):
        """
        Initialize unified evaluator.

        Args:
            output_dir: Base output directory for all evaluations
            run_efficiency: Whether to run efficiency evaluation
            run_qa: Whether to run QA evaluation
        """
        self.output_dir = Path(output_dir)
        self.run_efficiency = run_efficiency
        self.run_qa = run_qa

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_all(
        self,
        efficiency_timing_dir: str = "data/extraction_results_textbook_test",
        qa_questions_file: str = "data/evaluation/filtered_mirage_questions.jsonl",
        qa_corpus_path: str = "data/original/medcorp/train.jsonl",
        qa_indices_dir: str = "data/indices/medcorp",
        qa_question_type: str = "multiple_choice",
        qa_sample_limit: Optional[int] = None,
        qa_k: int = 5,
        qa_llm_model: str = "Qwen3-4B-Instruct-2507",
        qa_llm_base_url: str = "http://localhost:8001/v1",
        qa_llm_temperature: float = 0.1,
        qa_llm_max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Run all enabled evaluations.

        Args:
            efficiency_timing_dir: Directory with timing logs
            qa_questions_file: Path to filtered MIRAGE questions
            qa_corpus_path: Path to corpus
            qa_indices_dir: Directory with FAISS indices
            qa_sample_limit: Limit QA questions (for testing)
            qa_k: Number of documents to retrieve for QA
            qa_llm_model: LLM model name for answer generation
            qa_llm_base_url: MS-SWIFT server base URL
            qa_llm_temperature: LLM sampling temperature
            qa_llm_max_tokens: Maximum tokens for LLM response

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=" * 60)
        logger.info("UNIFIED EVALUATION PIPELINE")
        logger.info("=" * 60)

        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluations_run": [],
        }

        # 1. Efficiency Evaluation
        if self.run_efficiency:
            logger.info("\n[1/2] Running Efficiency Evaluation...")
            try:
                efficiency_results = run_efficiency_evaluation(
                    timing_log_dir=efficiency_timing_dir,
                    output_dir=str(self.output_dir / "efficiency"),
                )
                results["efficiency"] = efficiency_results
                results["evaluations_run"].append("efficiency")
                logger.success("✓ Efficiency evaluation complete")
            except Exception as e:
                logger.error(f"✗ Efficiency evaluation failed: {e}")
                results["efficiency"] = {"error": str(e)}

        # 2. QA Evaluation
        if self.run_qa:
            logger.info("\n[2/2] Running QA Evaluation...")
            try:
                qa_results = run_qa_evaluation(
                    questions_file=qa_questions_file,
                    corpus_path=qa_corpus_path,
                    indices_dir=qa_indices_dir,
                    output_dir=str(self.output_dir / "qa"),
                    question_type=qa_question_type,
                    sample_limit=qa_sample_limit,
                    k=qa_k,
                    llm_model=qa_llm_model,
                    llm_base_url=qa_llm_base_url,
                    llm_temperature=qa_llm_temperature,
                    llm_max_tokens=qa_llm_max_tokens,
                )
                results["qa"] = qa_results
                results["evaluations_run"].append("qa")
                logger.success("✓ QA evaluation complete")
            except Exception as e:
                logger.error(f"✗ QA evaluation failed: {e}")
                results["qa"] = {"error": str(e)}

        # Save unified summary
        summary_path = self.output_dir / f"unified_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Unified summary saved: {summary_path}")

        # Print final summary
        self._print_final_summary(results)

        return results

    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final evaluation summary."""
        print("\n" + "=" * 60)
        print("UNIFIED EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nTimestamp: {results['evaluation_timestamp']}")
        print(f"Evaluations Run: {', '.join(results['evaluations_run'])}")

        # Efficiency summary
        if "efficiency" in results and "error" not in results["efficiency"]:
            efficiency = results["efficiency"]["metrics"]
            print(f"\n[EFFICIENCY]")
            print(f"  Timing Overhead: {efficiency['timing_overhead_percent']:.4f}%")
            print(f"  Throughput: {efficiency['throughput']['chunks_per_second']:.2f} chunks/sec")
            print(f"  Latency P95: {efficiency['latency']['p95']:.2f} ms")

        # QA summary
        if "qa" in results and "error" not in results["qa"]:
            qa = results["qa"]["metrics"]
            accuracy = results["qa"]["accuracy"]
            print(f"\n[QA]")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Exact Match: {qa['exact_match']:.4f}")
            print(f"  Token F1: {qa['token_f1']:.4f}")
            print(f"  Answer Coverage: {qa['answer_coverage']:.4f}")

        print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation pipeline for STIndex"
    )

    # Which evaluations to run
    parser.add_argument(
        "--efficiency",
        action="store_true",
        help="Run efficiency evaluation only"
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run QA evaluation only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Run all evaluations (default)"
    )

    # Efficiency args
    parser.add_argument(
        "--efficiency-timing-dir",
        type=str,
        default="data/extraction_results_textbook_test",
        help="Directory with timing logs"
    )

    # QA args
    parser.add_argument(
        "--qa-questions",
        type=str,
        default="data/evaluation/filtered_mirage_questions.jsonl",
        help="Path to filtered MIRAGE questions"
    )
    parser.add_argument(
        "--qa-corpus",
        type=str,
        default="data/original/medcorp/train.jsonl",
        help="Path to corpus"
    )
    parser.add_argument(
        "--qa-indices",
        type=str,
        default="data/indices/medcorp",
        help="Directory with FAISS indices"
    )
    parser.add_argument(
        "--qa-sample-limit",
        type=int,
        default=None,
        help="Limit QA questions (for testing)"
    )
    parser.add_argument(
        "--qa-k",
        type=int,
        default=5,
        help="Number of documents to retrieve for QA"
    )
    parser.add_argument(
        "--qa-question-type",
        type=str,
        default="multiple_choice",
        choices=["multiple_choice", "open_ended", "boolean"],
        help="Type of questions (multiple_choice, open_ended, boolean)"
    )
    parser.add_argument(
        "--qa-llm-model",
        type=str,
        default="Qwen3-4B-Instruct-2507",
        help="LLM model name for answer generation"
    )
    parser.add_argument(
        "--qa-llm-base-url",
        type=str,
        default="http://localhost:8001/v1",
        help="MS-SWIFT server base URL"
    )
    parser.add_argument(
        "--qa-llm-temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature"
    )
    parser.add_argument(
        "--qa-llm-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for LLM response"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/evaluations",
        help="Output directory"
    )

    args = parser.parse_args()

    # Determine which evaluations to run
    run_efficiency = args.efficiency or (args.all and not args.qa)
    run_qa = args.qa or (args.all and not args.efficiency)

    # If specific flags are set, override --all
    if args.efficiency or args.qa:
        run_efficiency = args.efficiency
        run_qa = args.qa

    # Run unified evaluation
    evaluator = UnifiedEvaluator(
        output_dir=args.output,
        run_efficiency=run_efficiency,
        run_qa=run_qa,
    )

    evaluator.evaluate_all(
        efficiency_timing_dir=args.efficiency_timing_dir,
        qa_questions_file=args.qa_questions,
        qa_corpus_path=args.qa_corpus,
        qa_indices_dir=args.qa_indices,
        qa_question_type=args.qa_question_type,
        qa_sample_limit=args.qa_sample_limit,
        qa_k=args.qa_k,
        qa_llm_model=args.qa_llm_model,
        qa_llm_base_url=args.qa_llm_base_url,
        qa_llm_temperature=args.qa_llm_temperature,
        qa_llm_max_tokens=args.qa_llm_max_tokens,
    )


if __name__ == "__main__":
    main()
