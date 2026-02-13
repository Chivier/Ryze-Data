#!/usr/bin/env python3
"""Standalone OCR script using Microsoft MarkItDown.

Processes all images from ArxivQA or SlideVQA datasets, converts them
to PDFs, then runs MarkItDown to produce Markdown output.

Usage:
    python run_ocr.py --dataset arxivqa [--max-samples 5]
    python run_ocr.py --dataset slidevqa --output-dir /tmp/out
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Add parent dirs to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples
from _shared.image_utils import images_to_pdf

logger = logging.getLogger(__name__)

MODEL_NAME = "markitdown"


def _pick_writable_cache_root(primary: Path) -> Path:
    """Pick a writable cache root, with safe fallbacks."""
    candidates = [
        primary,
        Path(__file__).resolve().parent / ".cache",
        Path("/tmp/markitdown_cache"),
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
    sample_id: str,
    image_paths: list[str],
    output_dir: Path,
    pdf_cache_dir: Path,
) -> bool:
    """Process a single sample with MarkItDown.

    Args:
        sample_id: Unique sample identifier.
        image_paths: Paths to source images.
        output_dir: Directory for Markdown output.
        pdf_cache_dir: Directory for cached PDFs.

    Returns:
        True if successful, False otherwise.
    """
    from markitdown import MarkItDown

    md_output_dir = output_dir / sample_id
    md_path = md_output_dir / f"{sample_id}.md"

    # Resume: skip if output already exists
    if md_path.exists():
        logger.debug("Skipping %s (already processed)", sample_id)
        return True

    # Convert images to PDF (with resize)
    pdf_path = str(pdf_cache_dir / f"{sample_id}.pdf")
    if not Path(pdf_path).exists():
        try:
            with tempfile.TemporaryDirectory(prefix="markitdown_resize_") as tmp_dir:
                resized_paths = _resize_images(image_paths, tmp_dir)
                images_to_pdf(resized_paths, pdf_path)
        except Exception as e:
            logger.error("Failed to create PDF for %s: %s", sample_id, e)
            return False

    # Run MarkItDown
    try:
        converter = MarkItDown()
        result = converter.convert(pdf_path)

        md_output_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(result.text_content, encoding="utf-8")
        return True

    except Exception as e:
        logger.error("Failed to process %s: %s", sample_id, e)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MarkItDown OCR on HF datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxivqa", "slidevqa"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/ocr_precompute/markitdown/{dataset})",
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

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = (project_root / args.cache_dir).resolve()
    configure_cache_env(cache_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "data" / "ocr_precompute" / MODEL_NAME / args.dataset).resolve()

    pdf_cache_dir = project_root / "data" / "benchmark_data" / "ocr_pdfs" / args.dataset
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("MarkItDown OCR — dataset=%s, output=%s", args.dataset, output_dir)

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

            ok = process_sample(sample.sample_id, sample.image_paths, output_dir, pdf_cache_dir)
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
