"""
Integration tests for QA evaluation pipeline.

Tests LLM-based QA evaluation with real MS-SWIFT server.
"""

import pytest
import json
from pathlib import Path
from stindex.eval.qa_evaluation import QAEvaluator


@pytest.fixture
def test_questions_sample(tmp_path):
    """Create a small sample of test questions"""
    questions = [
        {
            "id": "test_q1",
            "question": "What is the most common cause of diabetes mellitus type 2?",
            "answer_labels": ["insulin resistance"],
            "question_type": "multiple_choice",
            "metadata": {
                "source_dataset": "medqa",
                "options": {
                    "A": "Autoimmune destruction of beta cells",
                    "B": "Insulin resistance and relative insulin deficiency",
                    "C": "Genetic mutations in insulin gene",
                    "D": "Viral infection of pancreas"
                },
                "correct_option": "B"
            }
        },
        {
            "id": "test_q2",
            "question": "Which of the following is the first-line treatment for hypertension?",
            "answer_labels": ["thiazide diuretics"],
            "question_type": "multiple_choice",
            "metadata": {
                "source_dataset": "medqa",
                "options": {
                    "A": "Beta blockers",
                    "B": "Thiazide diuretics",
                    "C": "Alpha blockers",
                    "D": "Calcium channel blockers"
                },
                "correct_option": "B"
            }
        }
    ]

    # Write to temporary JSONL file
    questions_file = tmp_path / "test_questions.jsonl"
    with open(questions_file, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    return str(questions_file)


class TestQAEvaluatorInitialization:
    """Test QA evaluator initialization"""

    def test_evaluator_init(self, test_questions_sample, tmp_path):
        """Test evaluator initialization"""
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
            k=5,
            llm_model="Qwen3-4B-Instruct-2507",
            llm_base_url="http://localhost:8001/v1",
            llm_temperature=0.1,
            llm_max_tokens=256,
        )

        assert evaluator.k == 5
        assert evaluator.llm_model == "Qwen3-4B-Instruct-2507"
        assert evaluator.llm_temperature == 0.1
        assert evaluator.llm_max_tokens == 256
        assert len(evaluator.questions) == 2

    def test_load_questions(self, test_questions_sample, tmp_path):
        """Test loading questions from JSONL"""
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
        )

        assert len(evaluator.questions) == 2
        assert evaluator.questions[0]["id"] == "test_q1"
        assert evaluator.questions[1]["id"] == "test_q2"


@pytest.mark.integration
class TestQAEvaluationWithRealServer:
    """Integration tests with real MS-SWIFT server"""

    @pytest.mark.skipif(
        not Path("data/indices/medcorp").exists(),
        reason="FAISS indices not available"
    )
    def test_evaluate_single_question(self, test_questions_sample, tmp_path):
        """Test evaluating a single question with real LLM server"""
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
            k=5,
            llm_model="Qwen3-4B-Instruct-2507",
            llm_base_url="http://localhost:8001/v1",
            llm_temperature=0.1,
            llm_max_tokens=256,
        )

        # Evaluate first question
        question = evaluator.questions[0]
        result = evaluator.evaluate_question(question)

        # Check result structure
        assert "question_id" in result
        assert "predicted_option" in result
        assert "correct_option" in result
        assert "correct" in result
        assert "retrieved_chunks" in result

        # Check that we got a valid answer
        assert result["question_id"] == "test_q1"
        assert result["correct_option"] == "B"
        assert result["predicted_option"] in ["A", "B", "C", "D", ""]

        # Check retrieval worked
        assert result["retrieved_chunks"] > 0

    @pytest.mark.skipif(
        not Path("data/indices/medcorp").exists(),
        reason="FAISS indices not available"
    )
    def test_full_evaluation_pipeline(self, test_questions_sample, tmp_path):
        """Test full evaluation pipeline with real LLM server"""
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
            k=5,
            llm_model="Qwen3-4B-Instruct-2507",
            llm_base_url="http://localhost:8001/v1",
            llm_temperature=0.1,
            llm_max_tokens=256,
        )

        # Run evaluation on 2 questions
        results = evaluator.evaluate(sample_limit=2)

        # Check results structure
        assert "metrics" in results
        assert "accuracy" in results
        assert "meets_targets" in results
        assert "output_files" in results

        # Check metrics
        metrics = results["metrics"]
        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert "answer_coverage" in metrics

        # Check accuracy
        assert 0.0 <= results["accuracy"] <= 1.0

        # Check output files exist
        csv_path = Path(results["output_files"]["csv"])
        json_path = Path(results["output_files"]["json"])
        assert csv_path.exists()
        assert json_path.exists()

    @pytest.mark.skipif(
        not Path("data/indices/medcorp").exists(),
        reason="FAISS indices not available"
    )
    def test_llm_answer_extraction(self, test_questions_sample, tmp_path):
        """Test LLM-based answer extraction"""
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
            k=5,
            llm_model="Qwen3-4B-Instruct-2507",
            llm_base_url="http://localhost:8001/v1",
            llm_temperature=0.1,
            llm_max_tokens=256,
        )

        # Get a question
        question = evaluator.questions[0]

        # Retrieve context
        retrieval_results = evaluator.retriever.retrieve(question["question"], k=5)
        retrieved_chunks = [r.contents for r in retrieval_results]

        # Extract answer using LLM
        predicted_option = evaluator._extract_answer_from_options(
            question, retrieved_chunks
        )

        # Check we got a valid option
        assert predicted_option in ["A", "B", "C", "D", ""]


class TestQAEvaluationEdgeCases:
    """Test edge cases"""

    def test_empty_options(self, tmp_path):
        """Test handling questions with no options"""
        questions = [
            {
                "id": "test_q1",
                "question": "What is diabetes?",
                "answer_labels": ["metabolic disease"],
                "question_type": "open_ended",
                "metadata": {}
            }
        ]

        questions_file = tmp_path / "test_questions.jsonl"
        with open(questions_file, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")

        evaluator = QAEvaluator(
            questions_file=str(questions_file),
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
        )

        # Should not crash with empty options
        answer = evaluator._extract_answer_from_options(
            questions[0], ["Some context"]
        )
        assert answer == ""

    def test_llm_server_error_handling(self, test_questions_sample, tmp_path):
        """Test handling LLM server errors"""
        # Use invalid base URL
        evaluator = QAEvaluator(
            questions_file=test_questions_sample,
            corpus_path="data/original/medcorp/train.jsonl",
            indices_dir="data/indices/medcorp",
            output_dir=str(tmp_path / "output"),
            llm_base_url="http://invalid-server:9999/v1",
        )

        question = evaluator.questions[0]
        retrieved_chunks = ["Some context"]

        # Should return empty string on error, not crash
        answer = evaluator._extract_answer_from_options(question, retrieved_chunks)
        assert answer == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])
