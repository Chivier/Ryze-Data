#!/usr/bin/env python3
"""Standalone OCR script using DeepSeek-OCR v1.

Processes all images from ArxivQA or SlideVQA datasets using local
HuggingFace inference with deepseek-ai/DeepSeek-OCR.

Usage:
    python run_ocr.py --dataset arxivqa --gpu 0 [--max-samples 5]
    python run_ocr.py --dataset slidevqa --gpu 1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek_ocr_v1"
HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
IMAGE_SIZE = 640
INCLUDE_TEST_COMPRESS = True
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
BASE_SIZE = 448


def load_model(gpu_id: int):
    """Load DeepSeek-OCR v1 model and tokenizer.

    Args:
        gpu_id: GPU device ID.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info("Loading %s on GPU %d...", HF_MODEL_ID, gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)

    # Try flash_attention_2, fall back to eager
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).cuda()

    logger.info("Model loaded (attn=%s, dtype=bfloat16)", attn_impl)
    return model, tokenizer


def infer_single_image(model, tokenizer, image_path: str, output_path: str) -> str:
    """Run OCR inference on a single image.

    Args:
        model: Loaded DeepSeek model.
        tokenizer: Loaded tokenizer.
        image_path: Path to the image file.
        output_path: Directory for model output.

    Returns:
        Markdown string produced by the model.
    """
    result = model.infer(
        tokenizer,
        prompt=DEFAULT_PROMPT,
        image_file=image_path,
        output_path=output_path,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )
    return result


def process_sample(
    model,
    tokenizer,
    sample_id: str,
    image_paths: list[str],
    output_dir: Path,
) -> bool:
    """Process a single sample with DeepSeek-OCR v1.

    Args:
        model: Loaded DeepSeek model.
        tokenizer: Loaded tokenizer.
        sample_id: Unique sample identifier.
        image_paths: Paths to source images.
        output_dir: Directory for Markdown output.

    Returns:
        True if successful, False otherwise.
    """
    md_output_dir = output_dir / sample_id
    md_path = md_output_dir / f"{sample_id}.md"

    if md_path.exists():
        logger.debug("Skipping %s (already processed)", sample_id)
        return True

    try:
        md_output_dir.mkdir(parents=True, exist_ok=True)

        page_markdowns = []
        for img_path in image_paths:
            md = infer_single_image(model, tokenizer, img_path, str(md_output_dir))
            page_markdowns.append(md)

        full_md = "\n\n---\n\n".join(page_markdowns)
        md_path.write_text(full_md, encoding="utf-8")
        return True

    except Exception as e:
        logger.error("Failed to process %s: %s", sample_id, e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR v1 on HF datasets")
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = (project_root / args.cache_dir).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "data" / "ocr_precompute" / MODEL_NAME / args.dataset).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DeepSeek-OCR v1 — dataset=%s, gpu=%d, output=%s", args.dataset, args.gpu, output_dir)

    model, tokenizer = load_model(args.gpu)

    success = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    for sample in load_dataset_samples(args.dataset, str(cache_dir), args.max_samples):
        md_path = output_dir / sample.sample_id / f"{sample.sample_id}.md"
        if md_path.exists():
            skipped += 1
            continue

        ok = process_sample(model, tokenizer, sample.sample_id, sample.image_paths, output_dir)
        if ok:
            success += 1
        else:
            failed += 1

        if (success + failed) % 50 == 0:
            logger.info("Progress: %d success, %d failed, %d skipped", success, failed, skipped)

    elapsed = time.time() - start_time
    logger.info(
        "Done in %.1fs — %d success, %d failed, %d skipped",
        elapsed,
        success,
        failed,
        skipped,
    )


if __name__ == "__main__":
    main()
