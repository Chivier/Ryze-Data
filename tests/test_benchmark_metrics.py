"""Unit tests for benchmark metrics.

All metrics are pure functions — no mocking needed.
"""

import pytest

from src.benchmark.metrics import (
    accuracy,
    bleu_4,
    bleu_4_score,
    compute_all_metrics,
    exact_match,
    exact_match_score,
    normalize_text,
    rouge_l,
    rouge_l_score,
    token_f1,
    token_f1_score,
)


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Hello  World  ") == "hello world"

    def test_punctuation_removal(self):
        assert normalize_text("Hello, World!") == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_case_insensitive(self):
        assert normalize_text("ABC") == "abc"


class TestAccuracy:
    def test_all_correct(self):
        preds = ["A", "B", "C"]
        refs = ["A", "B", "C"]
        assert accuracy(preds, refs) == 1.0

    def test_none_correct(self):
        preds = ["A", "B", "C"]
        refs = ["D", "E", "F"]
        assert accuracy(preds, refs) == 0.0

    def test_partial(self):
        preds = ["A", "B", "C"]
        refs = ["A", "X", "C"]
        assert accuracy(preds, refs) == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        preds = ["hello"]
        refs = ["HELLO"]
        assert accuracy(preds, refs) == 1.0

    def test_empty(self):
        assert accuracy([], []) == 0.0


class TestExactMatch:
    def test_match(self):
        assert exact_match("hello world", "Hello World") == 1.0

    def test_no_match(self):
        assert exact_match("hello", "world") == 0.0

    def test_whitespace(self):
        assert exact_match("hello  world", "hello world") == 1.0

    def test_score_list(self):
        preds = ["hello", "world", "foo"]
        refs = ["hello", "bar", "foo"]
        assert exact_match_score(preds, refs) == pytest.approx(2 / 3)


class TestBLEU4:
    def test_identical(self):
        text = "the cat sat on the mat"
        score = bleu_4(text, text)
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        score = bleu_4("a b c d e", "x y z w v")
        assert score == 0.0

    def test_partial_overlap(self):
        pred = "the quick brown fox jumps over the lazy dog"
        ref = "the quick brown cat jumps over the lazy dog"
        score = bleu_4(pred, ref)
        assert 0.0 < score < 1.0

    def test_empty_prediction(self):
        assert bleu_4("", "hello world") == 0.0

    def test_empty_reference(self):
        assert bleu_4("hello world", "") == 0.0

    def test_short_prediction(self):
        """Very short predictions should have low BLEU due to brevity penalty."""
        score = bleu_4("cat", "the cat sat on the mat")
        assert score < 0.5

    def test_score_list(self):
        preds = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        ]
        refs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        ]
        assert bleu_4_score(preds, refs) == pytest.approx(1.0)


class TestROUGEL:
    def test_identical(self):
        text = "the cat sat on the mat"
        assert rouge_l(text, text) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert rouge_l("a b c", "x y z") == 0.0

    def test_partial_overlap(self):
        pred = "the cat sat on the mat"
        ref = "the cat is on the mat"
        score = rouge_l(pred, ref)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert rouge_l("", "hello") == 0.0
        assert rouge_l("hello", "") == 0.0

    def test_subsequence(self):
        """LCS should capture longest common subsequence."""
        pred = "a b c d"
        ref = "a c d"
        score = rouge_l(pred, ref)
        # LCS = "a c d" (length 3), precision = 3/4, recall = 3/3 = 1
        # F1 = 2 * (3/4) * 1 / (3/4 + 1) = 6/7
        assert score == pytest.approx(6 / 7, abs=0.01)

    def test_score_list(self):
        preds = ["a b c", "x y z"]
        refs = ["a b c", "x y z"]
        assert rouge_l_score(preds, refs) == pytest.approx(1.0)


class TestTokenF1:
    def test_identical(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("hello", "world") == 0.0

    def test_partial_overlap(self):
        # pred: {the, cat, sat}, ref: {the, cat, is}
        # overlap = 2, precision = 2/3, recall = 2/3, F1 = 2/3
        score = token_f1("the cat sat", "the cat is")
        assert score == pytest.approx(2 / 3)

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_one_empty(self):
        assert token_f1("", "hello") == 0.0
        assert token_f1("hello", "") == 0.0

    def test_score_list(self):
        preds = ["a b", "c d"]
        refs = ["a b", "c d"]
        assert token_f1_score(preds, refs) == pytest.approx(1.0)


class TestComputeAllMetrics:
    def test_multiple_choice(self):
        preds = ["A", "B", "C"]
        refs = ["A", "B", "D"]
        result = compute_all_metrics(preds, refs, "multiple_choice")
        assert "accuracy" in result
        assert result["accuracy"] == pytest.approx(2 / 3)
        assert "exact_match" not in result

    def test_free_text(self):
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo baz"]
        result = compute_all_metrics(preds, refs, "free_text")
        assert "exact_match" in result
        assert "bleu_4" in result
        assert "rouge_l" in result
        assert "token_f1" in result
        assert "accuracy" not in result

    def test_with_ocr_times(self):
        preds = ["A"]
        refs = ["A"]
        result = compute_all_metrics(
            preds, refs, "multiple_choice", ocr_times=[1.5, 2.5]
        )
        assert result["avg_ocr_time"] == pytest.approx(2.0)

    def test_empty(self):
        result = compute_all_metrics([], [], "multiple_choice")
        assert result["accuracy"] == 0.0
