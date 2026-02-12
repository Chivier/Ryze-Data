"""Utilities for OCR-precompute benchmark evaluation."""

from scripts.utils.benchmark.dataset_adapter import (
    BenchmarkRawSample,
    load_benchmark_samples,
)
from scripts.utils.benchmark.metrics_ext import (
    compute_free_text_metrics,
    compute_multiple_choice_metrics,
)
from scripts.utils.benchmark.ocr_resolver import (
    EXPERIMENT_ORDER,
    OCR_EXPERIMENTS,
    parse_experiments,
    read_ocr_markdown,
    resolve_ocr_dir,
)

__all__ = [
    "BenchmarkRawSample",
    "EXPERIMENT_ORDER",
    "OCR_EXPERIMENTS",
    "compute_free_text_metrics",
    "compute_multiple_choice_metrics",
    "load_benchmark_samples",
    "parse_experiments",
    "read_ocr_markdown",
    "resolve_ocr_dir",
]
