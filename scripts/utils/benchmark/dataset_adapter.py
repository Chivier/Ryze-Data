"""Adapters for loading benchmark samples with prompts + images + references.

This module pairs shared image-caching loaders with raw HuggingFace rows so the
benchmark runner can always access:
1) image paths
2) question prompt
3) choices (for MC)
4) ground-truth answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.utils._shared.dataset_loader import OCRSample, load_dataset_samples

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkRawSample:
    """Single benchmark sample for QA evaluation."""

    sample_id: str
    image_paths: list[str]
    question: str
    choices: Optional[list[str]]
    reference: str
    question_type: str
    dataset: str


def _parse_sample_index(sample_id: str) -> int:
    """Parse integer suffix from sample id, e.g. ``arxivqa_42`` -> ``42``."""
    try:
        return int(sample_id.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid sample id format: {sample_id}") from exc


def _resolve_arxiv_reference(label: str, choices: list[str]) -> str:
    """Resolve ArxivQA letter label to option text when possible."""
    normalized_label = str(label).strip()
    if not normalized_label:
        return ""
    if not choices:
        return normalized_label

    label_char = normalized_label[0].upper()
    label_idx = ord(label_char) - ord("A")
    if 0 <= label_idx < len(choices):
        return choices[label_idx]
    return normalized_label


def _to_choices(value: object) -> Optional[list[str]]:
    """Normalize choices into a list of strings or ``None``."""
    if value is None:
        return None
    if isinstance(value, list):
        parsed = [str(item) for item in value]
        return parsed if parsed else None
    return [str(value)]


def _to_text(value: object) -> str:
    """Normalize values from HF rows into plain string."""
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0])
    return str(value)


def _load_hf_dataset(dataset_name: str):
    """Load raw HuggingFace dataset split by benchmark dataset name."""
    from datasets import load_dataset

    if dataset_name == "arxivqa":
        return load_dataset("MMInstruction/ArxivQA", split="train")
    if dataset_name == "slidevqa":
        return load_dataset("NTT-hil-insight/SlideVQA", split="test")
    raise ValueError(
        f"Unknown dataset: {dataset_name}. Expected one of: arxivqa, slidevqa."
    )


def _build_sample(
    dataset_name: str,
    ocr_sample: OCRSample,
    raw_row: dict,
) -> BenchmarkRawSample:
    """Build benchmark sample from cached OCR image sample + raw HF row."""
    question = _to_text(raw_row.get("question"))

    if dataset_name == "arxivqa":
        choices = _to_choices(raw_row.get("options"))
        label = _to_text(raw_row.get("label"))
        reference = _resolve_arxiv_reference(label=label, choices=choices or [])
        question_type = "multiple_choice"
    else:
        choices = None
        reference = _to_text(raw_row.get("answer"))
        question_type = "free_text"

    return BenchmarkRawSample(
        sample_id=ocr_sample.sample_id,
        image_paths=list(ocr_sample.image_paths),
        question=question,
        choices=choices,
        reference=reference,
        question_type=question_type,
        dataset=dataset_name,
    )


def load_benchmark_samples(
    dataset_name: str,
    cache_dir: str,
    max_samples: int = 0,
) -> list[BenchmarkRawSample]:
    """Load benchmark samples with raw prompt + reference + cached image paths."""
    cache_path = str(Path(cache_dir).resolve())

    image_samples = list(
        load_dataset_samples(
            dataset_name=dataset_name,
            cache_dir=cache_path,
            max_samples=max_samples,
        )
    )
    if not image_samples:
        raise ValueError(f"No samples loaded from dataset '{dataset_name}'.")

    raw_dataset = _load_hf_dataset(dataset_name)
    samples: list[BenchmarkRawSample] = []

    for ocr_sample in image_samples:
        raw_idx = _parse_sample_index(ocr_sample.sample_id)
        if raw_idx < 0 or raw_idx >= len(raw_dataset):
            logger.warning(
                "Skipping %s: raw index %d out of range for dataset size %d",
                ocr_sample.sample_id,
                raw_idx,
                len(raw_dataset),
            )
            continue

        raw_row = raw_dataset[raw_idx]
        if not isinstance(raw_row, dict):
            logger.warning("Skipping %s: raw row is not dict", ocr_sample.sample_id)
            continue

        sample = _build_sample(
            dataset_name=dataset_name,
            ocr_sample=ocr_sample,
            raw_row=raw_row,
        )
        samples.append(sample)

    if not samples:
        raise ValueError(f"No valid samples built for dataset '{dataset_name}'.")

    return samples
