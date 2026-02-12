#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Custom inference runner: load samples from id_mapping JSON files,
render PDF pages to images, and submit to an OpenAI-compatible API
endpoint through user-defined processing.

Edit ``process_sample()`` below to customise prompt construction.

Data sources:
    - data/id_mappings/{dataset}_id_mapping.json   (QA metadata)
    - data/benchmark_data/ocr_pdfs/{dataset}/{sample_id}.pdf
    - data/ocr_precompute/{model}/{dataset}/{sample_id}/{sample_id}.md

Image resolution:
    - ArxivQA:  PDF → single PNG  (full page)
    - SlideVQA: PDF → extract ``evidence_pages`` only

Usage:
    python scripts/run_custom_inference.py \\
        --dataset arxivqa \\
        --experiment baseline1 \\
        --endpoints http://localhost:8000/v1 \\
        --max-samples 5
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.benchmark.client_pool import VLLMClientPool
from scripts.utils.benchmark.ocr_resolver import read_ocr_markdown, resolve_ocr_dir
from scripts.utils.benchmark.prompt_builder import (
    build_baseline_text_prompt,
    build_multimodal_messages,
    build_ocr_text_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """A single inference sample loaded from the id_mapping JSON."""

    sample_id: str
    image_paths: list[str]
    question: str
    choices: Optional[list[str]]
    reference: str
    question_type: str  # "multiple_choice" or "free_text"
    dataset: str


# ---------------------------------------------------------------------------
# *** USER: Edit this function to build your own prompt. ***
# ---------------------------------------------------------------------------

def process_sample(
    sample_id: str,
    image_paths: list[str],
    ocr_markdown: str | None,
    question: str,
    choices: list[str] | None,
    reference: str,
    question_type: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages for one sample.

    This is the main customisation point.  Replace the body of this function
    with your own prompt construction logic.

    Args:
        sample_id:     Unique identifier, e.g. ``"arxivqa_0"``.
        image_paths:   Absolute paths to cached PNG/JPG images.
                       ArxivQA: single image.  SlideVQA: multiple slides.
        ocr_markdown:  Precomputed OCR markdown, or ``None`` for baseline.
        question:      The question text from the id_mapping JSON.
        choices:       Options list (ArxivQA), or ``None`` (SlideVQA).
        reference:     Ground-truth answer (``label`` or ``answer``).
        question_type: ``"multiple_choice"`` or ``"free_text"``.

    Returns:
        OpenAI chat messages list, e.g.::

            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "text", "text": "your prompt here"},
            ]}]

    Tip:
        You can use the helper ``build_multimodal_messages(image_paths, text)``
        to encode images as base64 and wrap them into the messages format.
    """
    # --- Default implementation (works out of the box) ---
    if ocr_markdown is not None:
        text_prompt = build_ocr_text_prompt(
            ocr_markdown=ocr_markdown,
            question=question,
            choices=choices,
        )
    else:
        text_prompt = build_baseline_text_prompt(
            question=question,
            choices=choices,
        )
    return build_multimodal_messages(
        image_paths=image_paths,
        text_prompt=text_prompt,
    )


# ---------------------------------------------------------------------------
# PDF → PNG rendering
# ---------------------------------------------------------------------------

def _render_pdf_pages(
    pdf_path: Path,
    pages: list[int] | None,
    cache_dir: Path,
    sample_id: str,
) -> list[str]:
    """Render specific pages of a PDF to cached PNG files.

    Args:
        pdf_path:   Path to the source PDF.
        pages:      1-based page numbers to extract, or ``None`` for all pages.
        cache_dir:  Directory to store rendered PNGs.
        sample_id:  Used for cache file naming.

    Returns:
        Sorted list of absolute PNG paths.
    """
    from pdf2image import convert_from_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    if pages is not None:
        # Check if all requested page images are already cached.
        cached_paths = []
        all_cached = True
        for page_num in sorted(pages):
            png_path = cache_dir / f"{sample_id}_p{page_num}.png"
            if png_path.exists():
                cached_paths.append(str(png_path))
            else:
                all_cached = False
                break
        if all_cached:
            return cached_paths

        # Render only the requested pages (pdf2image uses 1-based indexing).
        result_paths: list[str] = []
        for page_num in sorted(pages):
            png_path = cache_dir / f"{sample_id}_p{page_num}.png"
            if not png_path.exists():
                images = convert_from_path(
                    str(pdf_path),
                    first_page=page_num,
                    last_page=page_num,
                )
                if images:
                    images[0].save(str(png_path), "PNG")
            result_paths.append(str(png_path))
        return result_paths

    # All pages — check single-page cache first (ArxivQA common case).
    single_png = cache_dir / f"{sample_id}.png"
    if single_png.exists():
        return [str(single_png)]

    images = convert_from_path(str(pdf_path))
    if len(images) == 1:
        images[0].save(str(single_png), "PNG")
        return [str(single_png)]

    result_paths = []
    for i, img in enumerate(images, start=1):
        png_path = cache_dir / f"{sample_id}_p{i}.png"
        if not png_path.exists():
            img.save(str(png_path), "PNG")
        result_paths.append(str(png_path))
    return result_paths


# ---------------------------------------------------------------------------
# Dataset loading from id_mapping JSON + PDF
# ---------------------------------------------------------------------------

def load_samples_from_mapping(
    dataset_name: str,
    mapping_path: Path,
    pdf_dir: Path,
    image_cache_dir: Path,
    max_samples: int = 0,
) -> list[Sample]:
    """Load samples from an id_mapping JSON and render images from PDFs.

    ArxivQA: ``{pdf_dir}/arxivqa_0.pdf`` → full page PNG.

    SlideVQA: ``{pdf_dir}/slidevqa_0.pdf`` → extract ``evidence_pages``
    (1-based page numbers) to individual PNGs.
    """
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    samples: list[Sample] = []

    for sample_id, item in raw.items():
        pdf_path = pdf_dir / f"{sample_id}.pdf"
        if not pdf_path.exists():
            logger.warning("Skipping %s: PDF not found at %s", sample_id, pdf_path)
            continue

        if dataset_name == "arxivqa":
            image_paths = _render_pdf_pages(
                pdf_path=pdf_path,
                pages=None,
                cache_dir=image_cache_dir,
                sample_id=sample_id,
            )
            question = item.get("question", "")
            choices = item.get("options")
            reference = str(item.get("label", ""))
            question_type = "multiple_choice"
        elif dataset_name == "slidevqa":
            evidence = item.get("evidence_pages")
            if not evidence:
                logger.warning("Skipping %s: no evidence_pages in mapping", sample_id)
                continue
            image_paths = _render_pdf_pages(
                pdf_path=pdf_path,
                pages=evidence,
                cache_dir=image_cache_dir,
                sample_id=sample_id,
            )
            question = item.get("question", "")
            choices = None
            reference = str(item.get("answer", ""))
            question_type = "free_text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if not image_paths:
            logger.warning("Skipping %s: no images rendered from PDF", sample_id)
            continue

        samples.append(
            Sample(
                sample_id=sample_id,
                image_paths=image_paths,
                question=question,
                choices=choices,
                reference=reference,
                question_type=question_type,
                dataset=dataset_name,
            )
        )

        if max_samples > 0 and len(samples) >= max_samples:
            break

    return samples


# ---------------------------------------------------------------------------
# Infrastructure (normally no need to modify below)
# ---------------------------------------------------------------------------

def _read_endpoint_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    tokens = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(part.strip() for part in line.split(",") if part.strip())
    return tokens


def _resolve_endpoints(endpoints_arg: Optional[str], endpoints_file: Path) -> list[str]:
    if endpoints_arg:
        parsed = [t.strip() for t in endpoints_arg.split(",") if t.strip()]
        if parsed:
            return parsed
    from_file = _read_endpoint_file(endpoints_file)
    if from_file:
        return from_file
    return ["http://localhost:8000/v1"]


def _load_result_cache(path: Path) -> dict[str, str]:
    cached: dict[str, str] = {}
    if not path.exists():
        return cached
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            sid = str(item.get("sample_id", "")).strip()
            if sid:
                cached[sid] = str(item.get("answer", ""))
    return cached


def _append_result(
    path: Path,
    sample: Sample,
    answer: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "sample_id": sample.sample_id,
        "answer": answer,
        "reference": sample.reference,
        "question_type": sample.question_type,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _infer_one(
    pool: VLLMClientPool,
    messages: list[dict[str, Any]],
) -> str:
    return pool.chat(messages=messages).strip()


def _run(
    *,
    experiment: str,
    samples: list[Sample],
    ocr_dir: Optional[Path],
    pool: VLLMClientPool,
    output_dir: Path,
    fail_on_missing_ocr: bool,
    concurrency: int,
) -> None:
    exp_dir = output_dir / experiment
    results_path = exp_dir / "results.jsonl"
    cached = _load_result_cache(results_path)

    num_cached = len(cached)
    num_missing_ocr = 0
    pending: list[tuple[Sample, list[dict[str, Any]]]] = []

    for sample in samples:
        if sample.sample_id in cached:
            continue

        ocr_markdown: Optional[str] = None
        if experiment != "baseline":
            if ocr_dir is None:
                raise ValueError(f"OCR directory is required for experiment '{experiment}'.")
            ocr_markdown = read_ocr_markdown(ocr_dir=ocr_dir, sample_id=sample.sample_id)
            if ocr_markdown is None:
                if fail_on_missing_ocr:
                    raise FileNotFoundError(
                        f"Missing OCR markdown for {sample.sample_id} in {ocr_dir}"
                    )
                num_missing_ocr += 1
                _append_result(path=results_path, sample=sample, answer="")
                continue

        messages = process_sample(
            sample_id=sample.sample_id,
            image_paths=sample.image_paths,
            ocr_markdown=ocr_markdown,
            question=sample.question,
            choices=sample.choices,
            reference=sample.reference,
            question_type=sample.question_type,
        )
        pending.append((sample, messages))

    logger.info(
        "Experiment %s: %d cached, %d pending, %d missing OCR",
        experiment,
        num_cached,
        len(pending),
        num_missing_ocr,
    )

    if not pending:
        return

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map: dict[Future[str], Sample] = {}
        for sample, messages in pending:
            future = executor.submit(_infer_one, pool, messages)
            future_map[future] = sample

        completed = 0
        for future in as_completed(future_map):
            sample = future_map[future]
            answer = ""
            try:
                answer = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Inference failed: sample=%s error=%s",
                    sample.sample_id,
                    exc,
                )
            _append_result(path=results_path, sample=sample, answer=answer)
            completed += 1
            if completed % 50 == 0:
                logger.info("Progress: %d/%d", completed, len(pending))

    logger.info(
        "Done — %d new, %d cached, %d missing OCR. Results: %s",
        len(pending),
        num_cached,
        num_missing_ocr,
        results_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom inference runner with user-defined prompt processing"
    )
    parser.add_argument("--dataset", required=True, choices=["arxivqa", "slidevqa"])
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["baseline", "baseline1", "baseline2", "baseline3", "us"],
        help="Experiment (determines which OCR precompute to load)",
    )
    parser.add_argument(
        "--mapping-dir",
        default="data/id_mappings",
        help="Directory containing {dataset}_id_mapping.json files",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/benchmark_data/ocr_pdfs",
        help="Root directory containing {dataset}/ subdirs with PDFs",
    )
    parser.add_argument(
        "--image-cache-dir",
        default="data/benchmark_data/inference_images",
        help="Directory to cache rendered PNG images from PDFs",
    )
    parser.add_argument("--ocr-root", default="data/ocr_precompute")
    parser.add_argument("--output-dir", default="data/custom_inference")
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--endpoints", default=None, help="Comma-separated API base URLs")
    parser.add_argument(
        "--endpoints-file",
        default="logs/benchmark/latest/vllm_pool_endpoints.txt",
    )
    parser.add_argument("--model", default="Qwen3-VL-8B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)

    missing_group = parser.add_mutually_exclusive_group()
    missing_group.add_argument(
        "--fail-on-missing-ocr",
        dest="fail_on_missing_ocr",
        action="store_true",
    )
    missing_group.add_argument(
        "--allow-missing-ocr",
        dest="fail_on_missing_ocr",
        action="store_false",
    )
    parser.set_defaults(fail_on_missing_ocr=True)

    return parser.parse_args()


MAPPING_FILE_MAP = {
    "arxivqa": "arxivqa_id_mapping.json",
    "slidevqa": "slidevqa_id_mapping.json",
}


def main() -> int:
    args = _parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    endpoints = _resolve_endpoints(
        endpoints_arg=args.endpoints,
        endpoints_file=Path(args.endpoints_file),
    )
    logger.info("Dataset: %s | Experiment: %s", args.dataset, args.experiment)
    logger.info("Endpoints: %s", endpoints)

    # Resolve paths
    mapping_path = Path(args.mapping_dir).resolve() / MAPPING_FILE_MAP[args.dataset]
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    pdf_dir = Path(args.pdf_dir).resolve() / args.dataset
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    image_cache_dir = Path(args.image_cache_dir).resolve() / args.dataset

    samples = load_samples_from_mapping(
        dataset_name=args.dataset,
        mapping_path=mapping_path,
        pdf_dir=pdf_dir,
        image_cache_dir=image_cache_dir,
        max_samples=args.max_samples,
    )
    logger.info("Loaded %d samples from %s", len(samples), mapping_path)

    ocr_dir = resolve_ocr_dir(
        ocr_root=args.ocr_root,
        dataset_name=args.dataset,
        experiment=args.experiment,
    )
    if ocr_dir is not None and not ocr_dir.exists():
        raise FileNotFoundError(
            f"OCR directory does not exist for experiment '{args.experiment}': {ocr_dir}"
        )

    pool = VLLMClientPool(
        endpoints=endpoints,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    dataset_output_dir = Path(args.output_dir).resolve() / args.dataset
    _run(
        experiment=args.experiment,
        samples=samples,
        ocr_dir=ocr_dir,
        pool=pool,
        output_dir=dataset_output_dir,
        fail_on_missing_ocr=args.fail_on_missing_ocr,
        concurrency=args.concurrency,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
