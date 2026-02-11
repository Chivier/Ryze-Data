"""Benchmark evaluation metrics.

All functions are pure with no external dependencies.
Designed for comparing OCR pipeline quality via downstream VQA tasks.
"""

import re
from collections import Counter
from typing import Dict, List, Optional


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace.

    Args:
        text: Raw text string.

    Returns:
        Normalized text.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute accuracy for multiple-choice questions.

    Args:
        predictions: List of predicted answers.
        references: List of correct answers.

    Returns:
        Accuracy score between 0 and 1.
    """
    if not predictions:
        return 0.0
    correct = sum(
        1
        for pred, ref in zip(predictions, references)
        if normalize_text(pred) == normalize_text(ref)
    )
    return correct / len(predictions)


def exact_match(prediction: str, reference: str) -> float:
    """Compute exact match between prediction and reference.

    Args:
        prediction: Predicted answer text.
        reference: Reference answer text.

    Returns:
        1.0 if exact match after normalization, 0.0 otherwise.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """Compute average exact match score over a list.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Average exact match score.
    """
    if not predictions:
        return 0.0
    return sum(exact_match(p, r) for p, r in zip(predictions, references)) / len(
        predictions
    )


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list.

    Args:
        tokens: List of tokens.
        n: N-gram size.

    Returns:
        Counter of n-gram tuples.
    """
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_4(prediction: str, reference: str) -> float:
    """Compute BLEU-4 score between prediction and reference.

    Uses brevity penalty and clipped n-gram precision for n=1..4.

    Args:
        prediction: Predicted text.
        reference: Reference text.

    Returns:
        BLEU-4 score between 0 and 1.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0.0

    # Clipped n-gram precision for n=1..4
    log_precisions = 0.0
    for n in range(1, 5):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)

        if not pred_ngrams:
            return 0.0

        clipped_count = 0
        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        precision = clipped_count / sum(pred_ngrams.values())
        if precision == 0:
            return 0.0

        import math

        log_precisions += math.log(precision)

    import math

    return bp * math.exp(log_precisions / 4)


def bleu_4_score(predictions: List[str], references: List[str]) -> float:
    """Compute average BLEU-4 score over a list.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Average BLEU-4 score.
    """
    if not predictions:
        return 0.0
    return sum(bleu_4(p, r) for p, r in zip(predictions, references)) / len(predictions)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of longest common subsequence.

    Args:
        a: First token list.
        b: Second token list.

    Returns:
        Length of LCS.
    """
    m, n = len(a), len(b)
    # Space-optimized: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n] if m > 0 else 0


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference.

    Uses the longest common subsequence (LCS) approach.

    Args:
        prediction: Predicted text.
        reference: Reference text.

    Returns:
        ROUGE-L F1 score between 0 and 1.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)

    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def rouge_l_score(predictions: List[str], references: List[str]) -> float:
    """Compute average ROUGE-L score over a list.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Average ROUGE-L F1 score.
    """
    if not predictions:
        return 0.0
    return sum(rouge_l(p, r) for p, r in zip(predictions, references)) / len(
        predictions
    )


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score.

    Based on token overlap between prediction and reference.

    Args:
        prediction: Predicted text.
        reference: Reference text.

    Returns:
        Token F1 score between 0 and 1.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Compute overlap
    overlap = sum((pred_counter & ref_counter).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def token_f1_score(predictions: List[str], references: List[str]) -> float:
    """Compute average token F1 score over a list.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.

    Returns:
        Average token F1 score.
    """
    if not predictions:
        return 0.0
    return sum(token_f1(p, r) for p, r in zip(predictions, references)) / len(
        predictions
    )


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    question_type: str = "free_text",
    ocr_times: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute all relevant metrics for a given question type.

    Args:
        predictions: List of predicted answers.
        references: List of reference answers.
        question_type: "multiple_choice" or "free_text".
        ocr_times: Optional list of OCR processing times per sample.

    Returns:
        Dictionary of metric name to score.
    """
    results: Dict[str, float] = {}

    if question_type == "multiple_choice":
        results["accuracy"] = accuracy(predictions, references)
    else:
        results["exact_match"] = exact_match_score(predictions, references)
        results["bleu_4"] = bleu_4_score(predictions, references)
        results["rouge_l"] = rouge_l_score(predictions, references)
        results["token_f1"] = token_f1_score(predictions, references)

    if ocr_times:
        results["avg_ocr_time"] = sum(ocr_times) / len(ocr_times)

    return results
