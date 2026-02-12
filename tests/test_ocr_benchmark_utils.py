"""Unit tests for OCR-precompute benchmark utility modules."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.utils._shared.dataset_loader import OCRSample
from scripts.utils.benchmark.client_pool import VLLMClientPool
from scripts.utils.benchmark.dataset_adapter import load_benchmark_samples
from scripts.utils.benchmark.metrics_ext import (
    compute_free_text_metrics,
    compute_multiple_choice_metrics,
)
from scripts.utils.benchmark.ocr_resolver import (
    find_ocr_markdown_path,
    parse_experiments,
    resolve_ocr_dir,
)


def test_parse_experiments_uses_canonical_order():
    parsed = parse_experiments("us,baseline,baseline2")
    assert parsed == ["baseline", "baseline2", "us"]


def test_parse_experiments_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown experiments"):
        parse_experiments("baseline,unknown")


def test_resolve_ocr_dir():
    assert resolve_ocr_dir("data/ocr_precompute", "arxivqa", "baseline") is None
    path = resolve_ocr_dir("data/ocr_precompute", "arxivqa", "baseline1")
    assert str(path).endswith("data/ocr_precompute/deepseek_ocr_v1/arxivqa")


def test_find_ocr_markdown_path_priority(tmp_path: Path):
    sample_dir = tmp_path / "arxivqa_0"
    sample_dir.mkdir()
    md_path = sample_dir / "arxivqa_0.md"
    mmd_path = sample_dir / "result.mmd"
    md_path.write_text("primary", encoding="utf-8")
    mmd_path.write_text("secondary", encoding="utf-8")

    found = find_ocr_markdown_path(tmp_path, "arxivqa_0")
    assert found == md_path


def test_compute_multiple_choice_metrics():
    metrics = compute_multiple_choice_metrics(
        predictions=["A", "A", "B", "C"],
        references=["A", "B", "B", "C"],
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx(0.8333, abs=1e-3)
    assert metrics["macro_recall"] == pytest.approx(0.8333, abs=1e-3)
    assert metrics["macro_f1"] == pytest.approx(0.7778, abs=1e-3)


def test_compute_free_text_metrics():
    metrics = compute_free_text_metrics(
        predictions=["alpha beta"],
        references=["alpha gamma"],
    )
    assert metrics["token_precision"] == pytest.approx(0.5)
    assert metrics["token_recall"] == pytest.approx(0.5)
    assert metrics["token_f1"] == pytest.approx(0.5)
    assert "exact_match" in metrics
    assert "bleu_4" in metrics
    assert "rouge_l" in metrics


def test_vllm_client_pool_round_robin(monkeypatch):
    called_urls: list[str] = []

    def fake_post(url, json, headers, timeout):  # noqa: ANN001
        called_urls.append(url)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
        }
        return response

    monkeypatch.setattr("scripts.utils.benchmark.client_pool.requests.post", fake_post)

    pool = VLLMClientPool(
        endpoints=["http://host-a:8000/v1", "http://host-b:8001"],
        model="Qwen3-VL-8B",
    )
    assert pool.chat([{"role": "user", "content": "q1"}]) == "answer"
    assert pool.chat([{"role": "user", "content": "q2"}]) == "answer"

    assert called_urls[0].endswith("host-a:8000/v1/chat/completions")
    assert called_urls[1].endswith("host-b:8001/v1/chat/completions")


def test_dataset_adapter_arxivqa_mapping(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"fake")

    fake_ocr_samples = [
        OCRSample(sample_id="arxivqa_0", image_paths=[str(image_path)], dataset="arxivqa")
    ]
    fake_dataset = [
        {
            "question": "Which option is correct?",
            "options": ["Option A", "Option B"],
            "label": "B",
        }
    ]

    monkeypatch.setattr(
        "scripts.utils.benchmark.dataset_adapter.load_dataset_samples",
        lambda dataset_name, cache_dir, max_samples: iter(fake_ocr_samples),
    )
    monkeypatch.setattr(
        "scripts.utils.benchmark.dataset_adapter._load_hf_dataset",
        lambda dataset_name: fake_dataset,
    )

    samples = load_benchmark_samples("arxivqa", str(tmp_path), max_samples=0)
    assert len(samples) == 1
    assert samples[0].question == "Which option is correct?"
    assert samples[0].choices == ["Option A", "Option B"]
    assert samples[0].reference == "Option B"
    assert samples[0].question_type == "multiple_choice"
