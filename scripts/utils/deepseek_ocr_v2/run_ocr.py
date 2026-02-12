#!/usr/bin/env python3
"""Standalone OCR script using DeepSeek-OCR v2.

Processes all images from ArxivQA or SlideVQA datasets using local
inference with deepseek-ai/DeepSeek-OCR-2 (transformers by default,
optional vLLM).

Usage:
    python run_ocr.py --dataset arxivqa --gpu 0 [--max-samples 5]
    python run_ocr.py --dataset slidevqa --gpu 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek_ocr_v2"
HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
MAX_TOKENS = 8192
IMAGE_SIZE = 768
BASE_SIZE = 1024


def _pick_writable_cache_root(primary: Path) -> Path:
    """Pick a writable cache root, with safe fallbacks."""
    candidates = [
        primary,
        Path(__file__).resolve().parent / ".cache",
        Path("/tmp/deepseek_ocr_v2_cache"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"No writable cache directory available. Tried: {', '.join(str(p) for p in candidates)}"
    )


def configure_cache_env(cache_dir: Path) -> None:
    """Route third-party caches into the project cache directory."""
    third_party_cache = _pick_writable_cache_root(cache_dir / "third_party_cache")
    env_defaults = {
        "XDG_CACHE_HOME": str(third_party_cache),
        "HF_HOME": str(third_party_cache / "huggingface"),
        "HF_HUB_CACHE": str(third_party_cache / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(third_party_cache / "huggingface" / "datasets"),
        "TORCH_HOME": str(third_party_cache / "torch"),
    }

    for env_name, env_value in env_defaults.items():
        if env_name not in os.environ:
            os.environ[env_name] = env_value
            Path(env_value).mkdir(parents=True, exist_ok=True)


def load_model_vllm(gpu_id: int):
    """Load DeepSeek-OCR v2 model and sampling params via vLLM."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    except Exception as e:
        raise RuntimeError(
            "vLLM DeepSeek-OCR integration is required for --backend vllm. "
            "Install it first, e.g. `uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`."
        ) from e

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info("Loading %s with vLLM on GPU %d...", HF_MODEL_ID, gpu_id)

    llm = LLM(
        model=HF_MODEL_ID,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        trust_remote_code=True,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
    }
    ngram_args = {
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": {128821, 128822},
    }
    try:
        sampling_params = SamplingParams(extra_args=ngram_args, **sampling_kwargs)
    except TypeError:
        logger.warning(
            "This vLLM version does not support SamplingParams.extra_args; using default sampling args."
        )
        sampling_params = SamplingParams(**sampling_kwargs)

    logger.info("vLLM model loaded")
    return llm, sampling_params


def load_model_transformers(gpu_id: int):
    """Load DeepSeek-OCR v2 model via transformers (README-aligned)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info("Loading %s with transformers on GPU %d...", HF_MODEL_ID, gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    # The README recommends flash_attention_2.
    try:
        model = AutoModel.from_pretrained(
            HF_MODEL_ID,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        attn_impl = "flash_attention_2"
    except Exception as e:
        logger.warning(
            "Failed to load with _attn_implementation=flash_attention_2 (%s). Falling back to eager.",
            e,
        )
        model = AutoModel.from_pretrained(
            HF_MODEL_ID,
            _attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        )
        attn_impl = "eager"

    model = model.eval().cuda().to(torch.bfloat16)
    logger.info("Transformers model loaded (attn=%s)", attn_impl)
    return model, tokenizer


def infer_single_image_vllm(llm, sampling_params, image_path: str) -> str:
    """Run OCR inference on a single image via vLLM."""
    from PIL import Image

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    model_input = [
        {
            "prompt": DEFAULT_PROMPT,
            "multi_modal_data": {"image": image},
        }
    ]
    outputs = llm.generate(model_input, sampling_params)
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("vLLM returned empty output")
    return outputs[0].outputs[0].text


def _extract_markdown(result: Any) -> str | None:
    """Normalize `model.infer(...)` return payload into markdown text."""
    if isinstance(result, str):
        value = result.strip()
        return value or None

    if isinstance(result, dict):
        for key in ("markdown", "text", "result", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(result, (list, tuple)):
        for item in result:
            value = _extract_markdown(item)
            if value:
                return value

    return None


def infer_single_image_transformers(model, tokenizer, image_path: str, output_path: str) -> str:
    """Run OCR inference on a single image via transformers."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = model.infer(
        tokenizer,
        prompt=DEFAULT_PROMPT,
        image_file=image_path,
        output_path=str(output_dir),
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=True,
        save_results=True,
    )

    markdown = _extract_markdown(result)
    if markdown:
        return markdown

    result_mmd = output_dir / "result.mmd"
    if result_mmd.exists():
        return result_mmd.read_text(encoding="utf-8")

    raise RuntimeError("DeepSeek-OCR-2 inference produced no markdown output")


def process_sample(
    backend: str,
    model,
    tokenizer_or_sampling,
    sample_id: str,
    image_paths: list[str],
    output_dir: Path,
) -> bool:
    """Process a single sample with DeepSeek-OCR v2."""
    md_output_dir = output_dir / sample_id
    md_path = md_output_dir / f"{sample_id}.md"

    if md_path.exists():
        logger.debug("Skipping %s (already processed)", sample_id)
        return True

    try:
        md_output_dir.mkdir(parents=True, exist_ok=True)

        page_markdowns = []
        for image_path in image_paths:
            if backend == "vllm":
                md = infer_single_image_vllm(model, tokenizer_or_sampling, image_path)
            elif backend == "transformers":
                md = infer_single_image_transformers(
                    model,
                    tokenizer_or_sampling,
                    image_path,
                    str(md_output_dir),
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            page_markdowns.append(md)

        full_md = "\n\n---\n\n".join(page_markdowns)
        md_path.write_text(full_md, encoding="utf-8")
        return True

    except Exception as e:
        logger.error("Failed to process %s: %s", sample_id, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR v2 on HF datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxivqa", "slidevqa"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/ocr_precompute/deepseek_ocr_v2/{dataset})",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/benchmark_data",
        help="Shared image cache directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to process (0=all)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "transformers"],
        default="transformers",
        help="Inference backend (default: transformers)",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional HuggingFace endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        logger.info("Using HuggingFace endpoint: %s", args.hf_endpoint)

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = (project_root / args.cache_dir).resolve()
    configure_cache_env(cache_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "data" / "ocr_precompute" / MODEL_NAME / args.dataset).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "DeepSeek-OCR v2 — dataset=%s, gpu=%d, backend=%s, output=%s",
        args.dataset,
        args.gpu,
        args.backend,
        output_dir,
    )

    selected_backend = None
    model = None
    tokenizer_or_sampling = None

    if args.backend in ("auto", "vllm"):
        try:
            model, tokenizer_or_sampling = load_model_vllm(args.gpu)
            selected_backend = "vllm"
        except Exception as e:
            if args.backend == "vllm":
                logger.error("Model loading failed with vLLM backend: %s", e)
                logger.error(
                    "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
                )
                return 1
            logger.warning("vLLM backend unavailable for DeepSeek-OCR-2, fallback to transformers: %s", e)

    if selected_backend is None:
        try:
            model, tokenizer_or_sampling = load_model_transformers(args.gpu)
            selected_backend = "transformers"
        except Exception as e:
            logger.error("Model loading failed with transformers backend: %s", e)
            logger.error(
                "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
            )
            logger.error("You may need to reinstall dependencies: bash setup_env.sh")
            return 1

    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    try:
        for sample in load_dataset_samples(args.dataset, str(cache_dir), args.max_samples):
            md_path = output_dir / sample.sample_id / f"{sample.sample_id}.md"
            if md_path.exists():
                skipped += 1
                continue

            ok = process_sample(
                selected_backend,
                model,
                tokenizer_or_sampling,
                sample.sample_id,
                sample.image_paths,
                output_dir,
            )
            if ok:
                success += 1
            else:
                failed += 1

            if (success + failed) % 50 == 0:
                logger.info("Progress: %d success, %d failed, %d skipped", success, failed, skipped)
    except Exception as e:
        logger.error("Dataset loading failed: %s", e)
        logger.error(
            "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
        )
        return 1

    elapsed = time.time() - start_time
    logger.info(
        "Done in %.1fs — %d success, %d failed, %d skipped",
        elapsed,
        success,
        failed,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
