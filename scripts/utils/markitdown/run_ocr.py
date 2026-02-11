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
import sys
import time
from pathlib import Path

# Add parent dirs to path for shared imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples
from _shared.image_utils import images_to_pdf

logger = logging.getLogger(__name__)

MODEL_NAME = "markitdown"


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

    # Convert images to PDF
    pdf_path = str(pdf_cache_dir / f"{sample_id}.pdf")
    if not Path(pdf_path).exists():
        try:
            images_to_pdf(image_paths, pdf_path)
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


def main():
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = (project_root / args.cache_dir).resolve()

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
