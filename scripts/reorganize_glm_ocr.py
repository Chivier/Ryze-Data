#!/usr/bin/env python3
"""Reorganize glm_ocr flat output into the standard precompute directory structure.

Reads id_mapping JSON files to map between glm_ocr filenames and benchmark
sample IDs, then creates:
    data/ocr_precompute/glm_ocr_organized/{dataset}/{sample_id}/{sample_id}.md

Original glm_ocr files are preserved (not moved or deleted).

Usage:
    python scripts/reorganize_glm_ocr.py [--dry-run] [--dataset arxivqa|slidevqa|all]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GLM_OCR_DIR = "data/ocr_precompute/glm_ocr"
DEFAULT_OUTPUT_DIR = "data/ocr_precompute/glm_ocr_organized"
DEFAULT_MAPPING_DIR = "data/id_mappings"


def reorganize_arxivqa(
    glm_ocr_dir: Path,
    output_dir: Path,
    mapping_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Reorganize ArxivQA glm_ocr .txt and .jpg files into standard structure.

    Builds a reverse map from ArxivID stems (extracted from the
    ``original_image`` field) to ``arxivqa_N`` sample IDs, then copies each
    ``.txt`` or ``.jpg`` file into ``{output_dir}/arxivqa_N/arxivqa_N.md``.
    """
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    # Build reverse map: arxiv_stem -> sample_id
    stem_to_sample: dict[str, str] = {}
    for sample_id, entry in mapping.items():
        original_image = entry.get("original_image", "")
        stem = Path(original_image).stem  # e.g. "0904.0709_0"
        if stem:
            stem_to_sample[stem] = sample_id

    stats = {
        "created": 0,
        "skipped_no_mapping": 0,
        "skipped_exists": 0,
        "errors": 0,
    }

    # Process both .txt and .jpg files (both contain text, renamed to .md)
    source_files = sorted(
        [*glm_ocr_dir.glob("*.txt"), *glm_ocr_dir.glob("*.jpg")]
    )
    logger.info("Found %d source files (.txt + .jpg) in %s", len(source_files), glm_ocr_dir)

    for src_path in source_files:
        stem = src_path.stem
        if stem not in stem_to_sample:
            stats["skipped_no_mapping"] += 1
            continue

        sample_id = stem_to_sample[stem]
        target_dir = output_dir / sample_id
        target_file = target_dir / f"{sample_id}.md"

        if target_file.exists():
            stats["skipped_exists"] += 1
            continue

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                content = src_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("Cannot read %s as text, skipping", src_path.name)
                stats["errors"] += 1
                continue
            target_file.write_text(content, encoding="utf-8")

        stats["created"] += 1

    return stats


def reorganize_slidevqa(
    glm_ocr_dir: Path,
    output_dir: Path,
    mapping_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Reorganize SlideVQA glm_ocr per-page .md files into combined files.

    Multiple ``slidevqa_N`` entries can share the same slide deck.  The
    numeric ID used in glm_ocr filenames (e.g. ``0`` in ``0_page_1.md``)
    corresponds to the smallest ``qa_id`` among samples that share a deck.
    """
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    # Group samples by deck_name and find the smallest qa_id per deck.
    deck_samples: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for sample_id, entry in mapping.items():
        deck_name = entry["deck_name"]
        qa_id = entry["qa_id"]
        deck_samples[deck_name].append((qa_id, sample_id, entry))

    deck_to_glm_id: dict[str, int] = {}
    for deck_name, samples in deck_samples.items():
        samples.sort(key=lambda x: x[0])
        deck_to_glm_id[deck_name] = samples[0][0]

    # Index available page files: {numeric_id: {page_num: path}}
    page_index: dict[int, dict[int, Path]] = defaultdict(dict)
    for f in glm_ocr_dir.iterdir():
        if not f.name.endswith(".md") or "_page_" not in f.name:
            continue
        parts = f.stem.split("_page_")
        if len(parts) == 2:
            try:
                num_id = int(parts[0])
                page_num = int(parts[1])
                page_index[num_id][page_num] = f
            except ValueError:
                continue

    logger.info(
        "Found %d unique decks with page files in %s",
        len(page_index),
        glm_ocr_dir,
    )

    stats = {"created": 0, "skipped_exists": 0, "missing_pages": 0, "no_pages": 0}

    for sample_id, entry in mapping.items():
        deck_name = entry["deck_name"]
        num_slides = entry.get("num_slides", 20)
        glm_id = deck_to_glm_id[deck_name]

        target_dir = output_dir / sample_id
        target_file = target_dir / f"{sample_id}.md"

        if target_file.exists():
            stats["skipped_exists"] += 1
            continue

        if glm_id not in page_index:
            stats["no_pages"] += 1
            continue

        pages = page_index[glm_id]
        page_contents: list[str] = []

        for page_num in range(1, num_slides + 1):
            if page_num in pages:
                content = pages[page_num].read_text(encoding="utf-8").strip()
                page_contents.append(content)
            else:
                page_contents.append(f"<!-- Page {page_num} missing -->")
                stats["missing_pages"] += 1

        combined = "\n\n---\n\n".join(page_contents)

        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file.write_text(combined, encoding="utf-8")

        stats["created"] += 1

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize glm_ocr files into standard precompute structure"
    )
    parser.add_argument(
        "--dataset",
        choices=["arxivqa", "slidevqa", "all"],
        default="all",
        help="Which dataset to reorganize (default: all)",
    )
    parser.add_argument(
        "--glm-ocr-dir",
        default=str(PROJECT_ROOT / DEFAULT_GLM_OCR_DIR),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--mapping-dir",
        default=str(PROJECT_ROOT / DEFAULT_MAPPING_DIR),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing any files",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets = ["arxivqa", "slidevqa"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        logger.info("Processing %s ...", ds)
        glm_dir = Path(args.glm_ocr_dir) / ds
        out_dir = Path(args.output_dir) / ds
        mapping = Path(args.mapping_dir) / f"{ds}_id_mapping.json"

        if not glm_dir.exists():
            logger.error("Source directory not found: %s", glm_dir)
            continue
        if not mapping.exists():
            logger.error("Mapping file not found: %s", mapping)
            continue

        if ds == "arxivqa":
            stats = reorganize_arxivqa(
                glm_dir, out_dir, mapping, dry_run=args.dry_run
            )
        else:
            stats = reorganize_slidevqa(
                glm_dir, out_dir, mapping, dry_run=args.dry_run
            )

        prefix = "[DRY RUN] " if args.dry_run else ""
        logger.info("%s%s results: %s", prefix, ds, stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
