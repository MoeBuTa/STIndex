"""
QA evaluation pipeline using RRF-4 retrieval.

Supports multiple question types:
- multiple_choice: MCQ with options (A, B, C, D)
- open_ended: Free-text questions
- boolean: Yes/No questions

Evaluates:
1. QA performance (Exact Match, Token F1, Accuracy)
2. Answer coverage in retrieved context

Usage:
    python -m rag.eval.qa_evaluation \
        --questions data/evaluation/filtered_mirage_questions.jsonl \
        --corpus data/original/medcorp/train.jsonl \
        --indices data/indices/medcorp \
        --output data/output/evaluations/qa \
        --question-type multiple_choice \
        --sample-limit 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
import pandas as pd
from loguru import logger
from tqdm import tqdm
from openai import OpenAI

from rag.retriever.rrf_retriever import RRFRetriever
from rag.eval.qa_metrics import QAMetrics

# Question types
QuestionType = Literal["multiple_choice", "open_ended", "boolean"]


class QAEvaluator:
    """
    QA evaluator using RRF-4 retrieval and MIRAGE questions.
    """

    def __init__(
        self,
        questions_file: str,
        corpus_path: str,
        indices_dir: str,
        output_dir: str = "data/output/evaluations/qa",
        question_type: QuestionType = "multiple_choice",
        k: int = 5,
        use_bm25: bool = True,
        use_contriever: bool = True,
        use_specter: bool = True,
        use_medcpt: bool = True,
        llm_model: str = "Qwen3-4B-Instruct-2507",
        llm_base_url: str = "http://localhost:8001/v1",
        llm_temperature: float = 0.1,
        llm_max_tokens: int = 256,
    ):
        """
        Initialize QA evaluator.

        Args:
            questions_file: Path to questions JSONL
            corpus_path: Path to corpus JSONL
            indices_dir: Directory with FAISS indices
            output_dir: Output directory for evaluation results
            question_type: Type of questions ("multiple_choice", "open_ended", "boolean")
            k: Number of documents to retrieve
            use_bm25: Whether to use BM25 in RRF
            use_contriever: Whether to use Contriever
            use_specter: Whether to use SPECTER
            use_medcpt: Whether to use MedCPT
            llm_model: LLM model name for answer generation
            llm_base_url: MS-SWIFT server base URL
            llm_temperature: LLM sampling temperature
            llm_max_tokens: Maximum tokens for LLM response
        """
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.question_type = question_type
        self.k = k

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load prompt template for question type
        self.prompt_template = self._load_prompt_template(question_type)

        # Load questions
        logger.info(f"Loading questions from {self.questions_file}")
        self.questions = self._load_questions()
        logger.info(f"Loaded {len(self.questions)} questions")

        # Initialize retriever
        logger.info("Initializing RRF retriever...")
        self.retriever = RRFRetriever(
            corpus_path=corpus_path,
            indices_dir=indices_dir,
            use_bm25=use_bm25,
            use_contriever=use_contriever,
            use_specter=use_specter,
            use_medcpt=use_medcpt,
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

    def _load_prompt_template(self, question_type: QuestionType) -> str:
        """
        Load prompt template for the given question type.

        Args:
            question_type: Type of question

        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "generator" / "prompts" / f"{question_type}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found for question type '{question_type}' at {prompt_path}"
            )

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _generate_answer(
        self,
        question: Dict[str, Any],
        retrieved_chunks: List[str],
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        Handles different question types:
        - multiple_choice: Returns option letter (A, B, C, D)
        - open_ended: Returns free-text answer
        - boolean: Returns Yes/No

        Args:
            question: Question dictionary
            retrieved_chunks: List of retrieved text chunks

        Returns:
            Generated answer (format depends on question_type)
        """
        question_text = question.get("question", "")

        # Combine retrieved context
        context = "\n\n".join(retrieved_chunks[:3])  # Use top 3 chunks

        # Format prompt based on question type
        if self.question_type == "multiple_choice":
            options = question.get("metadata", {}).get("options", {})
            if not options:
                logger.warning(f"Multiple choice question {question.get('id')} has no options")
                return ""

            options_text = "\n".join([f"{letter}) {text}" for letter, text in sorted(options.items())])
            prompt = self.prompt_template.format(
                context=context,
                question=question_text,
                options=options_text
            )
        else:
            # open_ended or boolean
            prompt = self.prompt_template.format(
                context=context,
                question=question_text
            )

        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )

            # Extract answer from response
            answer = response.choices[0].message.content.strip()

            # Post-process based on question type
            if self.question_type == "multiple_choice":
                # Extract option letter (A, B, C, or D)
                for char in answer.upper():
                    if char in ['A', 'B', 'C', 'D']:
                        return char
                logger.warning(f"Could not extract valid option from LLM response: {answer}")
                return ""

            elif self.question_type == "boolean":
                # Extract Yes/No
                answer_lower = answer.lower()
                if "yes" in answer_lower:
                    return "Yes"
                elif "no" in answer_lower:
                    return "No"
                else:
                    logger.warning(f"Could not extract Yes/No from LLM response: {answer}")
                    return answer  # Return raw answer

            else:
                # open_ended - return as is
                return answer

        except Exception as e:
            logger.error(f"Error calling LLM for answer generation: {e}")
            return ""

    def evaluate_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single question.

        Args:
            question: Question dictionary

        Returns:
            Evaluation result dictionary
        """
        question_id = question["id"]
        question_text = question["question"]
        gold_answer = question.get("answer_labels", [""])[0]

        # Retrieve documents
        retrieval_results = self.retriever.retrieve(question_text, k=self.k)

        # Extract retrieved chunks
        retrieved_chunks = [r.contents for r in retrieval_results]

        # Generate answer using LLM
        predicted_answer_raw = self._generate_answer(question, retrieved_chunks)

        # Build result based on question type
        result = {
            "question_id": question_id,
            "question_text": question_text,
            "gold_answer": gold_answer,
            "predicted_answer_raw": predicted_answer_raw,
            "retrieved_chunks": len(retrieved_chunks),
            "top_chunk": retrieved_chunks[0] if retrieved_chunks else "",
            "retrieval_scores": [r.score for r in retrieval_results],
        }

        if self.question_type == "multiple_choice":
            # For MCQ, compute accuracy
            correct_option = question.get("metadata", {}).get("correct_option", "")
            predicted_option = predicted_answer_raw  # Already option letter (A, B, C, D)

            # Get full text of predicted answer
            options = question.get("metadata", {}).get("options", {})
            predicted_answer_text = options.get(predicted_option, "")

            result.update({
                "correct_option": correct_option,
                "predicted_option": predicted_option,
                "predicted_answer": predicted_answer_text,
                "correct": (predicted_option == correct_option),
            })
        else:
            # For open-ended and boolean, use raw answer
            result.update({
                "predicted_answer": predicted_answer_raw,
            })

        return result

    def evaluate(
        self,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run full QA evaluation.

        Args:
            sample_limit: Limit number of questions to evaluate (for testing)

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting QA evaluation")

        # Limit sample size if specified
        questions_to_eval = self.questions[:sample_limit] if sample_limit else self.questions
        logger.info(f"Evaluating {len(questions_to_eval)} questions")

        # Evaluate each question
        results = []
        metrics = QAMetrics()

        for question in tqdm(questions_to_eval, desc="Evaluating questions"):
            result = self.evaluate_question(question)
            results.append(result)

            # Add to metrics
            metrics.add_question_result(
                question_id=result["question_id"],
                predicted_answer=result["predicted_answer"],
                gold_answer=result["gold_answer"],
                retrieved_context=result["top_chunk"],
            )

        # Compute aggregate metrics
        aggregate_metrics = metrics.compute_aggregate_metrics()

        # Compute accuracy (for multiple choice)
        accuracy = sum(1 for r in results if r["correct"]) / len(results)

        # Generate reports
        detailed_df = pd.DataFrame(results)
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "retrieval_k": self.k,
            "accuracy": round(accuracy, 4),
            "qa_metrics": aggregate_metrics,
            "targets": {
                "exact_match": 0.40,
                "token_f1": 0.55,
                "answer_coverage": 0.80,
            },
            "meets_targets": metrics.meets_targets(),
        }

        # Save reports
        output_files = self._save_reports(detailed_df, summary)

        logger.success("QA evaluation complete!")

        # Print summary
        self._print_summary(summary, output_files)

        return {
            "metrics": aggregate_metrics,
            "accuracy": accuracy,
            "meets_targets": metrics.meets_targets(),
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
        csv_path = self.output_dir / f"qa_eval_{timestamp}.csv"
        detailed_df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed report: {csv_path}")

        # Save summary JSON
        json_path = self.output_dir / f"qa_eval_{timestamp}_summary.json"
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
        metrics = summary["qa_metrics"]
        meets_targets = summary["meets_targets"]

        print("\n" + "=" * 60)
        print("QA EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nTotal Questions: {summary['total_questions']}")
        print(f"Retrieval K: {summary['retrieval_k']}")
        print(f"\nAccuracy (Multiple Choice): {summary['accuracy']:.2%}")

        print(f"\nQA Metrics:")
        print(f"  Exact Match: {metrics['exact_match']:.4f} (target: ≥0.40)")
        print(f"  Token F1: {metrics['token_f1']:.4f} (target: ≥0.55)")
        print(f"  Answer Coverage: {metrics['answer_coverage']:.4f} (target: ≥0.80)")

        print(f"\nTargets Met:")
        print(f"  Exact Match: {'✓' if meets_targets['exact_match'] else '✗'}")
        print(f"  Token F1: {'✓' if meets_targets['token_f1'] else '✗'}")
        print(f"  Answer Coverage: {'✓' if meets_targets['answer_coverage'] else '✗'}")

        print(f"\nReports saved to:")
        print(f"  CSV: {output_files['csv']}")
        print(f"  JSON: {output_files['json']}")
        print("=" * 60 + "\n")


def run_qa_evaluation(
    questions_file: str = "data/evaluation/filtered_mirage_questions.jsonl",
    corpus_path: str = "data/original/medcorp/train.jsonl",
    indices_dir: str = "data/indices/medcorp",
    output_dir: str = "data/output/evaluations/qa",
    question_type: QuestionType = "multiple_choice",
    sample_limit: Optional[int] = None,
    k: int = 5,
    llm_model: str = "Qwen3-4B-Instruct-2507",
    llm_base_url: str = "http://localhost:8001/v1",
    llm_temperature: float = 0.1,
    llm_max_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Run QA evaluation pipeline.

    Args:
        questions_file: Path to questions
        corpus_path: Path to corpus JSONL
        indices_dir: Directory with FAISS indices
        output_dir: Output directory
        question_type: Type of questions ("multiple_choice", "open_ended", "boolean")
        sample_limit: Limit number of questions (for testing)
        k: Number of documents to retrieve
        llm_model: LLM model name for answer generation
        llm_base_url: MS-SWIFT server base URL
        llm_temperature: LLM sampling temperature
        llm_max_tokens: Maximum tokens for LLM response

    Returns:
        Dictionary with evaluation results
    """
    evaluator = QAEvaluator(
        questions_file=questions_file,
        corpus_path=corpus_path,
        indices_dir=indices_dir,
        output_dir=output_dir,
        question_type=question_type,
        k=k,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
    )

    return evaluator.evaluate(sample_limit=sample_limit)


def main():
    parser = argparse.ArgumentParser(
        description="QA evaluation using RRF-4 retrieval and MIRAGE"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/evaluation/filtered_mirage_questions.jsonl",
        help="Path to filtered MIRAGE questions"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/original/medcorp/train.jsonl",
        help="Path to corpus JSONL"
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="data/indices/medcorp",
        help="Directory with FAISS indices"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/evaluations/qa",
        help="Output directory"
    )
    parser.add_argument(
        "--question-type",
        type=str,
        default="multiple_choice",
        choices=["multiple_choice", "open_ended", "boolean"],
        help="Type of questions (multiple_choice, open_ended, boolean)"
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
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable BM25 retriever"
    )
    parser.add_argument(
        "--no-contriever",
        action="store_true",
        help="Disable Contriever"
    )
    parser.add_argument(
        "--no-specter",
        action="store_true",
        help="Disable SPECTER"
    )
    parser.add_argument(
        "--no-medcpt",
        action="store_true",
        help="Disable MedCPT"
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
        "--llm-temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature"
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for LLM response"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = QAEvaluator(
        questions_file=args.questions,
        corpus_path=args.corpus,
        indices_dir=args.indices,
        output_dir=args.output,
        question_type=args.question_type,
        k=args.k,
        use_bm25=not args.no_bm25,
        use_contriever=not args.no_contriever,
        use_specter=not args.no_specter,
        use_medcpt=not args.no_medcpt,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
    )

    evaluator.evaluate(sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
