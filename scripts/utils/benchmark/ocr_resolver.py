"""OCR path and content resolver for benchmark experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

EXPERIMENT_ORDER = ["baseline", "baseline1", "baseline2", "baseline3", "us", "baseline4"]
OCR_EXPERIMENTS = ["baseline1", "baseline2", "baseline3", "us", "baseline4"]

EXPERIMENT_TO_MODEL_DIR = {
    "baseline1": "deepseek_ocr_v1",
    "baseline2": "deepseek_ocr_v2",
    "baseline3": "markitdown",
    "us": "marker",
    "baseline4": "glm_ocr_organized",
}


def parse_experiments(raw: str) -> list[str]:
    """Parse comma-separated experiments and validate supported names."""
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    if not requested:
        raise ValueError("No experiments requested.")

    invalid = sorted(set(requested) - set(EXPERIMENT_ORDER))
    if invalid:
        raise ValueError(
            f"Unknown experiments: {invalid}. "
            f"Valid values: {', '.join(EXPERIMENT_ORDER)}"
        )

    # Keep canonical order to stabilize output and summary reports.
    return [name for name in EXPERIMENT_ORDER if name in requested]


def resolve_ocr_dir(
    ocr_root: str | Path,
    dataset_name: str,
    experiment: str,
) -> Optional[Path]:
    """Resolve OCR directory path for one experiment.

    Returns ``None`` for the baseline experiment.
    """
    if experiment == "baseline":
        return None

    if experiment not in EXPERIMENT_TO_MODEL_DIR:
        raise ValueError(f"Experiment '{experiment}' does not require OCR directory.")

    root = Path(ocr_root).resolve()
    return root / EXPERIMENT_TO_MODEL_DIR[experiment] / dataset_name


def find_ocr_markdown_path(
    ocr_dir: str | Path,
    sample_id: str,
) -> Optional[Path]:
    """Find OCR markdown file for a sample using known output conventions."""
    base = Path(ocr_dir)
    sample_dir = base / sample_id

    candidates = [
        sample_dir / f"{sample_id}.md",
        sample_dir / "result.mmd",
        base / f"{sample_id}.md",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path

    if sample_dir.exists() and sample_dir.is_dir():
        md_files = sorted(sample_dir.glob("*.md"))
        if md_files:
            return md_files[0]

    return None


def read_ocr_markdown(ocr_dir: str | Path, sample_id: str) -> Optional[str]:
    """Read OCR markdown content by sample id from OCR output directory."""
    md_path = find_ocr_markdown_path(ocr_dir=ocr_dir, sample_id=sample_id)
    if md_path is None:
        return None
    return md_path.read_text(encoding="utf-8")
