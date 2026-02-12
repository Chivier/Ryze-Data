#!/usr/bin/env python3
"""Generate reverse mapping tables from sequential OCR IDs to original dataset identifiers.

Produces JSON files that map pipeline sample IDs (e.g. ``arxivqa_42``) back to
the original image filenames or metadata in the HuggingFace datasets.

Usage:
    python generate_id_mapping.py --dataset arxivqa
    python generate_id_mapping.py --dataset slidevqa
    python generate_id_mapping.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared.dataset_loader import _load_dataset_with_auth, _load_hf_token

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "id_mappings"


def generate_arxivqa_mapping(max_samples: int = 0) -> dict:
    """Generate mapping: arxivqa_{idx} -> original image path and metadata."""
    logger.info("Loading ArxivQA dataset for mapping...")
    dataset = _load_dataset_with_auth("MMInstruction/ArxivQA", split="train")
    total = len(dataset)
    logger.info("ArxivQA: %d total samples", total)

    if max_samples > 0:
        total = min(max_samples, total)
        dataset = dataset.select(range(total))

    mapping = {}
    for idx, item in enumerate(dataset):
        sample_id = f"arxivqa_{idx}"
        entry = {
            "original_image": item.get("image", None),
        }
        # Include all available metadata for cross-referencing.
        for field in ("id", "question", "options", "label", "rationale"):
            val = item.get(field)
            if val is not None:
                entry[field] = val
        mapping[sample_id] = entry

        if (idx + 1) % 10000 == 0:
            logger.info("ArxivQA mapping progress: %d / %d", idx + 1, total)

    logger.info("ArxivQA mapping complete: %d entries", len(mapping))
    return mapping


def generate_slidevqa_mapping(max_samples: int = 0) -> dict:
    """Generate mapping: slidevqa_{idx} -> original slide metadata."""
    logger.info("Loading SlideVQA dataset for mapping...")
    dataset = _load_dataset_with_auth("NTT-hil-insight/SlideVQA", split="test")
    total = len(dataset)
    logger.info("SlideVQA: %d total samples", total)

    if max_samples > 0:
        total = min(max_samples, total)
        dataset = dataset.select(range(total))

    mapping = {}
    for idx, item in enumerate(dataset):
        sample_id = f"slidevqa_{idx}"

        # Count how many page_N columns have non-null images.
        page_count = 0
        for key in item:
            if key.startswith("page_") and item[key] is not None:
                page_count += 1

        entry: dict = {
            "num_slides": page_count,
        }
        # Include all available metadata for cross-referencing.
        for field in ("qa_id", "deck_name", "deck_url", "question", "answer", "arithmetic_expression", "evidence_pages"):
            val = item.get(field)
            if val is not None:
                entry[field] = val
        mapping[sample_id] = entry

        if (idx + 1) % 1000 == 0:
            logger.info("SlideVQA mapping progress: %d / %d", idx + 1, total)

    logger.info("SlideVQA mapping complete: %d entries", len(mapping))
    return mapping


def save_mapping(mapping: dict, output_path: Path) -> None:
    """Save mapping dict to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("Saved mapping to %s (%d entries)", output_path, len(mapping))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OCR ID -> original filename mapping")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxivqa", "slidevqa", "all"],
        help="Which dataset(s) to generate mappings for",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
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

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    datasets_to_process = (
        ["arxivqa", "slidevqa"] if args.dataset == "all" else [args.dataset]
    )

    generators = {
        "arxivqa": generate_arxivqa_mapping,
        "slidevqa": generate_slidevqa_mapping,
    }

    for ds_name in datasets_to_process:
        try:
            mapping = generators[ds_name](max_samples=args.max_samples)
            save_mapping(mapping, output_dir / f"{ds_name}_id_mapping.json")
        except Exception as e:
            logger.error("Failed to generate mapping for %s: %s", ds_name, e)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
