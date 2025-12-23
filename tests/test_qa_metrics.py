"""
Unit tests for QA metrics.

Tests exact match, token F1, and answer coverage computations.
"""

import pytest
from stindex.eval.qa_metrics import (
    QAMetrics,
    normalize_answer,
    tokenize,
)


class TestNormalization:
    """Test answer normalization functions"""

    def test_normalize_answer_lowercase(self):
        """Test lowercase normalization"""
        assert normalize_answer("HELLO WORLD") == "hello world"

    def test_normalize_answer_punctuation(self):
        """Test punctuation removal"""
        assert normalize_answer("Hello, World!") == "hello world"

    def test_normalize_answer_articles(self):
        """Test article removal"""
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat and an elephant") == "cat and elephant"

    def test_normalize_answer_whitespace(self):
        """Test extra whitespace removal"""
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_tokenize(self):
        """Test tokenization"""
        tokens = tokenize("The quick brown fox!")
        assert tokens == ["quick", "brown", "fox"]


class TestExactMatch:
    """Test exact match metric"""

    def test_exact_match_identical(self):
        """Test exact match with identical strings"""
        metrics = QAMetrics()
        assert metrics.exact_match("hello world", "hello world") is True

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive"""
        metrics = QAMetrics()
        assert metrics.exact_match("Hello World", "hello world") is True

    def test_exact_match_punctuation(self):
        """Test exact match ignores punctuation"""
        metrics = QAMetrics()
        assert metrics.exact_match("Hello, World!", "Hello World") is True

    def test_exact_match_articles(self):
        """Test exact match ignores articles"""
        metrics = QAMetrics()
        assert metrics.exact_match("the cat", "cat") is True

    def test_exact_match_different(self):
        """Test exact match with different strings"""
        metrics = QAMetrics()
        assert metrics.exact_match("hello world", "goodbye world") is False


class TestTokenF1:
    """Test token F1 metric"""

    def test_token_f1_identical(self):
        """Test F1 with identical strings"""
        metrics = QAMetrics()
        result = metrics.token_f1("hello world", "hello world")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_token_f1_partial_overlap(self):
        """Test F1 with partial overlap"""
        metrics = QAMetrics()
        result = metrics.token_f1("hello world", "hello universe")
        # hello: 1 common token
        # pred: 2 tokens (hello, world)
        # gold: 2 tokens (hello, universe)
        # precision = 1/2 = 0.5
        # recall = 1/2 = 0.5
        # f1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_token_f1_no_overlap(self):
        """Test F1 with no overlap"""
        metrics = QAMetrics()
        result = metrics.token_f1("hello world", "goodbye universe")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_token_f1_empty_prediction(self):
        """Test F1 with empty prediction"""
        metrics = QAMetrics()
        result = metrics.token_f1("", "hello world")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_token_f1_empty_gold(self):
        """Test F1 with empty gold answer"""
        metrics = QAMetrics()
        result = metrics.token_f1("hello world", "")
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_token_f1_both_empty(self):
        """Test F1 with both empty"""
        metrics = QAMetrics()
        result = metrics.token_f1("", "")
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0


class TestAnswerCoverage:
    """Test answer coverage metric"""

    def test_coverage_full(self):
        """Test coverage when all answer tokens in context"""
        metrics = QAMetrics()
        answer = "diabetes treatment"
        context = "The diabetes treatment involves insulin therapy"
        coverage = metrics.answer_coverage(answer, context)
        assert coverage == 1.0

    def test_coverage_partial(self):
        """Test coverage with partial overlap"""
        metrics = QAMetrics()
        answer = "diabetes insulin"
        context = "The patient has diabetes"
        coverage = metrics.answer_coverage(answer, context)
        assert coverage == 0.5  # Only "diabetes" is in context

    def test_coverage_none(self):
        """Test coverage with no overlap"""
        metrics = QAMetrics()
        answer = "cancer treatment"
        context = "The patient has diabetes"
        coverage = metrics.answer_coverage(answer, context)
        assert coverage == 0.0

    def test_coverage_empty_answer(self):
        """Test coverage with empty answer"""
        metrics = QAMetrics()
        coverage = metrics.answer_coverage("", "Some context")
        assert coverage == 1.0  # Empty answer is trivially covered


class TestQAMetricsAggregation:
    """Test aggregate metrics computation"""

    def test_add_question_result(self):
        """Test adding question results"""
        metrics = QAMetrics()
        metrics.add_question_result(
            question_id="q1",
            predicted_answer="diabetes",
            gold_answer="diabetes",
            retrieved_context="The patient has diabetes",
        )

        assert metrics.total_questions == 1
        assert metrics.exact_match_count == 1
        assert "q1" in metrics.question_results

    def test_compute_aggregate_metrics(self):
        """Test computing aggregate metrics"""
        metrics = QAMetrics()

        # Add exact match
        metrics.add_question_result(
            question_id="q1",
            predicted_answer="diabetes",
            gold_answer="diabetes",
            retrieved_context="diabetes treatment",
        )

        # Add partial match
        metrics.add_question_result(
            question_id="q2",
            predicted_answer="diabetes treatment",
            gold_answer="diabetes insulin",
            retrieved_context="diabetes treatment with insulin",
        )

        # Add no match
        metrics.add_question_result(
            question_id="q3",
            predicted_answer="cancer",
            gold_answer="diabetes",
            retrieved_context="cancer treatment",
        )

        aggregate = metrics.compute_aggregate_metrics()

        # Exact match: 1/3 = 0.333...
        assert 0.33 <= aggregate["exact_match"] <= 0.34

        # Token F1: (1.0 + partial_f1 + 0.0) / 3
        assert aggregate["token_f1"] > 0.0

        # Answer coverage should be computed for all questions with context
        assert aggregate["answer_coverage"] >= 0.0
        assert aggregate["answer_coverage"] <= 1.0

    def test_meets_targets(self):
        """Test target threshold checking"""
        metrics = QAMetrics()

        # Add questions with high scores
        for i in range(10):
            metrics.add_question_result(
                question_id=f"q{i}",
                predicted_answer="diabetes",
                gold_answer="diabetes",
                retrieved_context="The patient has diabetes",
            )

        targets = metrics.meets_targets(
            target_exact_match=0.50,
            target_token_f1=0.50,
            target_answer_coverage=0.50,
        )

        # All targets should be met with 100% correct answers
        assert targets["exact_match"] == True
        assert targets["token_f1"] == True
        assert targets["answer_coverage"] == True


class TestQAMetricsEdgeCases:
    """Test edge cases"""

    def test_empty_metrics(self):
        """Test metrics with no questions"""
        metrics = QAMetrics()
        aggregate = metrics.compute_aggregate_metrics()

        assert aggregate["exact_match"] == 0.0
        assert aggregate["token_f1"] == 0.0
        assert aggregate["answer_coverage"] == 0.0

    def test_correlation_insufficient_data(self):
        """Test correlation with insufficient data"""
        metrics = QAMetrics()
        correlation = metrics.retrieval_qa_correlation()
        assert correlation is None  # Not enough data

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = QAMetrics()
        metrics.add_question_result(
            question_id="q1",
            predicted_answer="diabetes",
            gold_answer="diabetes",
        )

        result_dict = metrics.to_dict()
        assert "metrics" in result_dict
        assert "total_questions" in result_dict
        assert result_dict["total_questions"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
