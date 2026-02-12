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
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.benchmark.client_pool import VLLMClientPool
from scripts.utils.benchmark.ocr_resolver import read_ocr_markdown, resolve_ocr_dir
from scripts.utils.benchmark.prompt_builder import build_multimodal_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample dataclass  (metadata only — no PDF rendering at load time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    """A single inference sample loaded from the id_mapping JSON.

    PDF rendering is deferred; ``pdf_path`` and ``evidence_pages`` are kept
    so images can be rendered lazily right before inference.
    """

    sample_id: str
    pdf_path: str
    evidence_pages: list[int] | None  # 1-based pages for SlideVQA, None for ArxivQA
    question: str
    choices: Optional[list[str]]
    reference: str
    question_type: str  # "multiple_choice" or "free_text"
    dataset: str


# ---------------------------------------------------------------------------
# Inference request container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InferenceRequest:
    """Wraps chat messages with optional per-request generation overrides."""

    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    response_format: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_LETTER_PREFIX_RE = re.compile(r"^[A-Da-d][.):\s]+\s*")


def _strip_choice_prefix(text: str) -> str:
    """Remove existing letter prefix like ``A. ``, ``A) `` from an option."""
    return _LETTER_PREFIX_RE.sub("", text).strip()


def _format_choices_clean(choices: list[str]) -> str:
    """Format choices as ``A. text\\nB. text\\n…``, avoiding double prefixes."""
    return "\n".join(
        f"{chr(65 + i)}. {_strip_choice_prefix(c)}" for i, c in enumerate(choices)
    )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _extract_option_letter(text: str) -> str:
    """Extract a single option letter (A–D) from model output.

    Handles JSON like ``{"answer": "B"}``, bare letters, and letters with
    trailing punctuation or text.
    """
    text = text.strip()
    # 1. Try JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            val = str(data.get("answer", "")).strip().upper()
            if len(val) == 1 and val in "ABCD":
                return val
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Leading letter (possibly quoted)
    m = re.match(r'^["\']?\s*([A-Da-d])\s*["\']?\s*[.)\]},:\s]', text)
    if m:
        return m.group(1).upper()
    # 3. Exactly one character
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    # 4. Fallback — return raw (evaluation will still normalize)
    return text


def _normalize_mc_reference(reference: str) -> str:
    """Normalize an ArxivQA label to a single letter (A/B/C/D).

    Handles ``"B"``, ``"C) Four"``, ``"A. some text"`` etc.
    """
    ref = reference.strip()
    m = re.match(r"^([A-Da-d])\b", ref)
    return m.group(1).upper() if m else ref


def postprocess_answer(answer: str, question_type: str) -> str:
    """Post-process raw model output based on question type."""
    if question_type == "multiple_choice":
        return _extract_option_letter(answer)
    return answer.strip()


# ---------------------------------------------------------------------------
# *** USER: Edit this function to build your own prompt. ***
# ---------------------------------------------------------------------------

_MC_SYSTEM = (
    "You are a visual question answering assistant. "
    "For the multiple-choice question, respond with ONLY a JSON object in "
    'the exact format: {"answer": "X"} where X is the option letter '
    "(A, B, C, or D). Do not include any other text."
)

_FT_SYSTEM = (
    "You are a visual question answering assistant. "
    "Answer as briefly as possible. Output ONLY the final answer — "
    "no explanation, reasoning, or extra text."
)


def process_sample(
    sample_id: str,
    image_paths: list[str],
    ocr_markdown: str | None,
    question: str,
    choices: list[str] | None,
    reference: str,
    question_type: str,
) -> InferenceRequest:
    """Build an ``InferenceRequest`` for one sample.

    * **ArxivQA** (``multiple_choice``): returns a JSON-mode request with
      ``max_tokens=32`` so the model outputs ``{"answer": "B"}``.
    * **SlideVQA** (``free_text``): returns a concise prompt requesting the
      shortest possible answer.
    """
    if question_type == "multiple_choice" and choices:
        # ---- ArxivQA: structured JSON output ----
        options_block = _format_choices_clean(choices)
        if ocr_markdown is not None:
            user_text = (
                f"OCR text from the document:\n{ocr_markdown}\n\n"
                f"Question: {question}\n\n{options_block}"
            )
        else:
            user_text = f"Question: {question}\n\n{options_block}"

        messages = [
            {"role": "system", "content": _MC_SYSTEM},
            *build_multimodal_messages(image_paths, user_text),
        ]
        return InferenceRequest(
            messages=messages,
            max_tokens=32,
            response_format={"type": "json_object"},
        )

    # ---- SlideVQA: concise free-text ----
    if ocr_markdown is not None:
        user_text = (
            f"OCR text:\n{ocr_markdown}\n\n"
            f"Question: {question}"
        )
    else:
        user_text = f"Question: {question}"

    messages = [
        {"role": "system", "content": _FT_SYSTEM},
        *build_multimodal_messages(image_paths, user_text),
    ]
    return InferenceRequest(messages=messages)


# ---------------------------------------------------------------------------
# PDF → PNG rendering  (lazy, with on-disk cache)
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
# Dataset loading from id_mapping JSON  (metadata only, no rendering)
# ---------------------------------------------------------------------------

def load_samples_from_mapping(
    dataset_name: str,
    mapping_path: Path,
    pdf_dir: Path,
    max_samples: int = 0,
) -> list[Sample]:
    """Load sample metadata from an id_mapping JSON.

    No PDF rendering happens here — images are rendered lazily at inference
    time so the script starts quickly even with large datasets.
    """
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    num_pdf_missing = 0
    num_no_evidence = 0

    for sample_id, item in raw.items():
        pdf_path = pdf_dir / f"{sample_id}.pdf"
        if not pdf_path.exists():
            num_pdf_missing += 1
            continue

        if dataset_name == "arxivqa":
            question = item.get("question", "")
            choices = item.get("options")
            reference = str(item.get("label", ""))
            question_type = "multiple_choice"
            evidence_pages = None
        elif dataset_name == "slidevqa":
            evidence_pages = item.get("evidence_pages")
            if not evidence_pages:
                num_no_evidence += 1
                continue
            question = item.get("question", "")
            choices = None
            reference = str(item.get("answer", ""))
            question_type = "free_text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        samples.append(
            Sample(
                sample_id=sample_id,
                pdf_path=str(pdf_path),
                evidence_pages=evidence_pages,
                question=question,
                choices=choices,
                reference=reference,
                question_type=question_type,
                dataset=dataset_name,
            )
        )

        if max_samples > 0 and len(samples) >= max_samples:
            break

    if num_pdf_missing:
        logger.warning(
            "Skipped %d / %d entries: PDF not found in %s",
            num_pdf_missing, len(raw), pdf_dir,
        )
    if num_no_evidence:
        logger.warning(
            "Skipped %d entries: no evidence_pages in mapping", num_no_evidence,
        )

    return samples


# ---------------------------------------------------------------------------
# Infrastructure
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
    reference: str | None = None,
    raw_answer: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "sample_id": sample.sample_id,
        "answer": answer,
        "raw_answer": raw_answer if raw_answer is not None else answer,
        "reference": reference if reference is not None else sample.reference,
        "question_type": sample.question_type,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _normalize_reference(reference: str, question_type: str) -> str:
    """Normalize reference for storage: extract letter for MC questions."""
    if question_type == "multiple_choice":
        return _normalize_mc_reference(reference)
    return reference


def _process_one(
    sample: Sample,
    pool: VLLMClientPool,
    ocr_dir: Optional[Path],
    image_cache_dir: Path,
    experiment: str,
    fail_on_missing_ocr: bool,
) -> tuple[str, str, str | None]:
    """Render images, build prompt, run inference for a single sample.

    Returns:
        ``(raw_answer, status, error_message)`` where *status* is one of
        ``"ok"``, ``"render_failed"``, ``"missing_ocr"``.
    """
    # 1. Render PDF → PNG (cached on disk, cheap if already rendered)
    image_paths = _render_pdf_pages(
        pdf_path=Path(sample.pdf_path),
        pages=sample.evidence_pages,
        cache_dir=image_cache_dir,
        sample_id=sample.sample_id,
    )
    if not image_paths:
        return "", "render_failed", "no images rendered from PDF"

    # 2. Read OCR markdown (if applicable)
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
            return "", "missing_ocr", None

    # 3. Build prompt
    request = process_sample(
        sample_id=sample.sample_id,
        image_paths=image_paths,
        ocr_markdown=ocr_markdown,
        question=sample.question,
        choices=sample.choices,
        reference=sample.reference,
        question_type=sample.question_type,
    )

    # 4. Call API
    kwargs: dict[str, Any] = {}
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.response_format is not None:
        kwargs["response_format"] = request.response_format
    raw_answer = pool.chat(messages=request.messages, **kwargs).strip()

    return raw_answer, "ok", None


def _run(
    *,
    experiment: str,
    samples: list[Sample],
    ocr_dir: Optional[Path],
    image_cache_dir: Path,
    pool: VLLMClientPool,
    output_dir: Path,
    fail_on_missing_ocr: bool,
    concurrency: int,
) -> None:
    exp_dir = output_dir / experiment
    results_path = exp_dir / "results.jsonl"
    cached = _load_result_cache(results_path)

    # Filter to pending samples (not already in cache).
    pending = [s for s in samples if s.sample_id not in cached]
    num_cached = len(cached)

    logger.info(
        "Experiment %s: %d cached, %d pending",
        experiment,
        num_cached,
        len(pending),
    )

    if not pending:
        return

    num_missing_ocr = 0
    num_render_failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map: dict[Future[tuple[str, str, str | None]], Sample] = {}
        for sample in pending:
            future = executor.submit(
                _process_one,
                sample,
                pool,
                ocr_dir,
                image_cache_dir,
                experiment,
                fail_on_missing_ocr,
            )
            future_map[future] = sample

        for future in as_completed(future_map):
            sample = future_map[future]
            raw_answer = ""
            answer = ""
            try:
                raw_answer, status, err_msg = future.result()

                if status == "render_failed":
                    num_render_failed += 1
                    logger.warning(
                        "Skipping %s: %s", sample.sample_id, err_msg,
                    )
                elif status == "missing_ocr":
                    num_missing_ocr += 1
                else:
                    answer = postprocess_answer(raw_answer, sample.question_type)

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed: sample=%s error=%s", sample.sample_id, exc,
                )

            _append_result(
                path=results_path,
                sample=sample,
                answer=answer,
                reference=_normalize_reference(sample.reference, sample.question_type),
                raw_answer=raw_answer,
            )
            completed += 1
            if completed % 50 == 0:
                logger.info("Progress: %d/%d", completed, len(pending))

    logger.info(
        "Done — %d new, %d cached, %d render-failed, %d missing OCR. Results: %s",
        completed,
        num_cached,
        num_render_failed,
        num_missing_ocr,
        results_path,
    )


def _load_dotenv() -> None:
    """Load ``.env`` from project root (if present) into ``os.environ``."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def _parse_args() -> argparse.Namespace:
    # Load .env BEFORE argparse so env-var defaults pick up .env values.
    _load_dotenv()

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

    parser.add_argument(
        "--endpoints",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Comma-separated API base URLs (default: $OPENAI_BASE_URL, then endpoints-file)",
    )
    parser.add_argument(
        "--endpoints-file",
        default="logs/benchmark/latest/vllm_pool_endpoints.txt",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL") or "Qwen3-VL-8B",
        help="Model name (default: $OPENAI_MODEL, then Qwen3-VL-8B)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "EMPTY",
        help="API key (default: $OPENAI_API_KEY / $API_KEY, then 'EMPTY')",
    )
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
        max_samples=args.max_samples,
    )
    logger.info("Loaded %d sample metadata from %s", len(samples), mapping_path)

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
        image_cache_dir=image_cache_dir,
        pool=pool,
        output_dir=dataset_output_dir,
        fail_on_missing_ocr=args.fail_on_missing_ocr,
        concurrency=args.concurrency,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
