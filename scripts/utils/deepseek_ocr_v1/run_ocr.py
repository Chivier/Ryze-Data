#!/usr/bin/env python3
"""Standalone OCR script using DeepSeek-OCR v1 with vLLM.

Processes all images from ArxivQA or SlideVQA datasets via local
inference using deepseek-ai/DeepSeek-OCR and the official vLLM recipe.

Usage:
    python run_ocr.py --dataset arxivqa --gpu 0 [--max-samples 5]
    python run_ocr.py --dataset slidevqa --gpu 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek_ocr_v1"
HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR"

# Keep legacy prompt as default so outputs stay aligned with prior implementation.
LEGACY_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
# Official plain-prompt style from vLLM DeepSeek-OCR recipe.
RECIPE_PROMPT = "<image>\n Free OCR."
MAX_TOKENS = 8192


def _pick_writable_cache_root(primary: Path) -> Path:
    """Pick a writable cache root, with safe fallbacks."""
    candidates = [
        primary,
        Path(__file__).resolve().parent / ".cache",
        Path("/tmp/deepseek_ocr_v1_cache"),
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


def _build_sampling_params():
    """Build vLLM SamplingParams with DeepSeek OCR n-gram controls."""
    from vllm import SamplingParams

    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
    }
    # DeepSeek OCR recipe-specific anti-repeat parameters.
    ngram_args = {
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": {128821, 128822},
    }
    try:
        return SamplingParams(extra_args=ngram_args, **sampling_kwargs)
    except TypeError:
        logger.warning(
            "This vLLM build does not support SamplingParams.extra_args; using basic sampling params."
        )
        return SamplingParams(**sampling_kwargs)


def load_model_vllm(
    gpu_id: int,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
):
    """Load DeepSeek-OCR v1 model and sampling params via vLLM."""
    try:
        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    except Exception as e:
        raise RuntimeError(
            "vLLM DeepSeek-OCR integration is required. Install with "
            "`uv pip install -U vllm --torch-backend auto`."
        ) from e

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info("Loading %s with vLLM on GPU %d...", HF_MODEL_ID, gpu_id)

    llm_kwargs = {
        "model": HF_MODEL_ID,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "trust_remote_code": True,
        "logits_processors": [NGramPerReqLogitsProcessor],
    }
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    try:
        llm = LLM(**llm_kwargs)
    except TypeError as e:
        # Keep compatibility with older vLLM builds that may miss some kwargs.
        fallback_keys = ("mm_processor_cache_gb", "max_num_batched_tokens", "max_model_len")
        dropped_keys = [key for key in fallback_keys if key in llm_kwargs and key in str(e)]
        if not dropped_keys:
            raise
        for key in dropped_keys:
            llm_kwargs.pop(key, None)
        logger.warning("Retrying vLLM load without unsupported kwargs: %s", ", ".join(dropped_keys))
        llm = LLM(**llm_kwargs)

    sampling_params = _build_sampling_params()
    logger.info("vLLM model loaded")
    return llm, sampling_params


def infer_single_image_vllm(
    llm,
    sampling_params,
    prompt: str,
    image_path: str,
) -> str:
    """Run OCR inference on a single image."""
    from PIL import Image

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    model_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }
    ]
    outputs = llm.generate(model_inputs, sampling_params)
    if not outputs or not outputs[0].outputs:
        raise RuntimeError("vLLM returned empty output")
    return outputs[0].outputs[0].text


MAX_IMAGE_SIZE = (1024, 1024)


def _resize_images(image_paths: list[str], work_dir: str) -> list[str]:
    """Resize images to fit within MAX_IMAGE_SIZE, preserving aspect ratio.

    Images that already fit are returned as-is. Oversized images are saved to
    work_dir as resized copies.
    """
    from PIL import Image

    resized: list[str] = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        if img.width <= MAX_IMAGE_SIZE[0] and img.height <= MAX_IMAGE_SIZE[1]:
            resized.append(path)
            continue
        img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
        out = os.path.join(work_dir, f"resized_{i}_{Path(path).name}")
        img.save(out)
        resized.append(out)
    return resized


def process_sample(
    llm,
    sampling_params,
    prompt: str,
    sample_id: str,
    image_paths: list[str],
    output_dir: Path,
) -> bool:
    """Process a single sample with DeepSeek-OCR v1."""
    md_output_dir = output_dir / sample_id
    md_path = md_output_dir / f"{sample_id}.md"

    if md_path.exists():
        logger.debug("Skipping %s (already processed)", sample_id)
        return True

    try:
        md_output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dsocr1_resize_") as tmp_dir:
            resized_paths = _resize_images(image_paths, tmp_dir)
            page_markdowns: list[str] = []
            for image_path in resized_paths:
                page_markdowns.append(
                    infer_single_image_vllm(
                        llm=llm,
                        sampling_params=sampling_params,
                        prompt=prompt,
                        image_path=image_path,
                    )
                )
        full_md = "\n\n---\n\n".join(page_markdowns)
        md_path.write_text(full_md, encoding="utf-8")
        return True
    except Exception as e:
        logger.error("Failed to process %s: %s", sample_id, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR v1 on HF datasets via vLLM")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxivqa", "slidevqa"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/ocr_precompute/deepseek_ocr_v1/{dataset})",
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
        "--prompt-mode",
        choices=["legacy", "recipe"],
        default="legacy",
        help="Prompt preset. `legacy` preserves previous implementation output style.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom prompt override; include <image> token for multimodal parsing",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM scheduler limit override (for memory tuning)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max model length override",
    )
    # Compatibility flag: old scripts may still pass --backend auto|transformers.
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "transformers"],
        default="vllm",
        help="Deprecated. This script now runs with vLLM only.",
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

    if args.backend != "vllm":
        logger.warning(
            "Backend '%s' is deprecated for deepseek_ocr_v1. Using vLLM backend.",
            args.backend,
        )

    if args.prompt:
        prompt = args.prompt
    elif args.prompt_mode == "recipe":
        prompt = RECIPE_PROMPT
    else:
        prompt = LEGACY_PROMPT

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
        "DeepSeek-OCR v1 (vLLM) — dataset=%s, gpu=%d, output=%s",
        args.dataset,
        args.gpu,
        output_dir,
    )

    try:
        llm, sampling_params = load_model_vllm(
            gpu_id=args.gpu,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        logger.error("Model loading failed with vLLM backend: %s", e)
        logger.error("Install vLLM first: uv pip install -U vllm --torch-backend auto")
        logger.error(
            "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
        )
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
                llm=llm,
                sampling_params=sampling_params,
                prompt=prompt,
                sample_id=sample.sample_id,
                image_paths=sample.image_paths,
                output_dir=output_dir,
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
