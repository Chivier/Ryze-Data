"""Extended benchmark metrics for OCR-precompute evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from src.benchmark.metrics import (
    accuracy,
    bleu_4_score,
    exact_match_score,
    normalize_text,
    rouge_l_score,
    token_f1_score,
)


def _token_precision_recall_f1_single(
    prediction: str, reference: str
) -> tuple[float, float, float]:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_precision_score(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(
        _token_precision_recall_f1_single(pred, ref)[0]
        for pred, ref in zip(predictions, references)
    ) / len(predictions)


def token_recall_score(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(
        _token_precision_recall_f1_single(pred, ref)[1]
        for pred, ref in zip(predictions, references)
    ) / len(predictions)


def macro_precision_recall_f1(
    predictions: list[str], references: list[str]
) -> tuple[float, float, float]:
    """Compute macro precision/recall/F1 for multiple-choice answer strings."""
    if not predictions:
        return 0.0, 0.0, 0.0

    normalized_pairs = [
        (normalize_text(pred), normalize_text(ref))
        for pred, ref in zip(predictions, references)
    ]
    labels = sorted(
        {
            label
            for pair in normalized_pairs
            for label in pair
            if label != ""
        }
    )
    if not labels:
        return 0.0, 0.0, 0.0

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for pred, ref in normalized_pairs:
            if pred == label and ref == label:
                tp += 1
            elif pred == label and ref != label:
                fp += 1
            elif pred != label and ref == label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = len(labels)
    return (
        macro_precision / label_count,
        macro_recall / label_count,
        macro_f1 / label_count,
    )


def compute_multiple_choice_metrics(
    predictions: list[str],
    references: list[str],
    *,
    extra: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Compute metrics for MC benchmark experiments."""
    macro_p, macro_r, macro_f1 = macro_precision_recall_f1(predictions, references)
    result: dict[str, float] = {
        "accuracy": accuracy(predictions, references),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "num_samples": float(len(predictions)),
    }
    if extra:
        result.update(extra)
    return result


def compute_free_text_metrics(
    predictions: list[str],
    references: list[str],
    *,
    extra: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Compute metrics for free-text benchmark experiments."""
    result: dict[str, float] = {
        "exact_match": exact_match_score(predictions, references),
        "bleu_4": bleu_4_score(predictions, references),
        "rouge_l": rouge_l_score(predictions, references),
        "token_precision": token_precision_score(predictions, references),
        "token_recall": token_recall_score(predictions, references),
        "token_f1": token_f1_score(predictions, references),
        "num_samples": float(len(predictions)),
    }
    if extra:
        result.update(extra)
    return result
