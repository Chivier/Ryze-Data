"""Standalone HuggingFace dataset loading and image extraction.

Loads ArxivQA and SlideVQA datasets and yields OCRSample objects with
cached image paths for downstream OCR processing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class OCRSample:
    """A single sample ready for OCR processing.

    Attributes:
        sample_id: Unique identifier, e.g. "arxivqa_0" or "slidevqa_42".
        image_paths: Absolute paths to cached PNG images.
        dataset: Dataset name ("arxivqa" or "slidevqa").
    """

    sample_id: str
    image_paths: list[str]
    dataset: str


def _decode_image(image):
    """Decode an image value from HuggingFace datasets into a PIL Image.

    HF datasets may return images in several forms depending on feature
    type and decoding settings:
      - ``PIL.Image.Image`` — already decoded.
      - ``dict`` with ``bytes`` and/or ``path`` keys — partially decoded.
      - ``str`` — relative path inside the dataset cache (not usable directly).

    Args:
        image: Raw image value from a dataset row.

    Returns:
        PIL Image, or None if decoding fails.
    """
    import io

    from PIL import Image

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, dict):
        # HF dict format: {"bytes": b"...", "path": "..."}
        if image.get("bytes"):
            return Image.open(io.BytesIO(image["bytes"]))
        path = image.get("path")
        if path and Path(path).exists():
            return Image.open(path)

    return None


def load_arxivqa(
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load ArxivQA dataset and yield OCRSample objects.

    Each sample has a single figure image.  The ``image`` column is
    explicitly cast to ``datasets.Image(decode=True)`` so that string
    paths stored inside the dataset are resolved and decoded to PIL
    Images automatically.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset

    logger.info("Loading ArxivQA dataset...")
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    logger.info("ArxivQA: %d total samples", len(dataset))

    # Force image column to be decoded as PIL Images.
    # Without this, some datasets return raw string paths that point
    # into the HF cache and are not usable directly.
    if "image" in dataset.column_names:
        dataset = dataset.cast_column("image", HFImage(decode=True))

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = Path(cache_dir) / "arxivqa_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset):
        sample_id = f"arxivqa_{idx}"
        image_path = str(image_dir / f"{sample_id}.png")

        if not Path(image_path).exists():
            image = item.get("image")
            pil_image = _decode_image(image)
            if pil_image is None:
                logger.warning("Skipping %s: could not decode image", sample_id)
                continue
            pil_image.save(image_path)

        yield OCRSample(
            sample_id=sample_id,
            image_paths=[image_path],
            dataset="arxivqa",
        )


def load_slidevqa(
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load SlideVQA dataset and yield OCRSample objects.

    Each sample has multiple slide images.  Image columns are cast to
    ``datasets.Image(decode=True)`` to guarantee PIL decoding.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    from datasets import Image as HFImage
    from datasets import load_dataset

    logger.info("Loading SlideVQA dataset...")
    dataset = load_dataset("NTT-hil-insight/SlideVQA", split="test")
    logger.info("SlideVQA: %d total samples", len(dataset))

    # Force all image columns to be decoded as PIL Images.
    for col in dataset.column_names:
        if col in ("image", "images") or col.startswith("page_"):
            try:
                dataset = dataset.cast_column(col, HFImage(decode=True))
            except Exception:
                pass  # Column may not be an image type

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = Path(cache_dir) / "slidevqa_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset):
        sample_id = f"slidevqa_{idx}"

        # Extract slide images — may be a list or page_N columns
        images = item.get("images", [])
        if not images:
            single_image = item.get("image")
            if single_image is not None:
                images = [single_image]

        image_paths = []
        for img_idx, image in enumerate(images):
            img_path = str(image_dir / f"{sample_id}_slide_{img_idx}.png")
            if not Path(img_path).exists():
                pil_image = _decode_image(image)
                if pil_image is None:
                    logger.warning("Skipping %s slide %d: could not decode image", sample_id, img_idx)
                    continue
                pil_image.save(img_path)
            image_paths.append(img_path)

        if not image_paths:
            logger.warning("Skipping %s: no images found", sample_id)
            continue

        yield OCRSample(
            sample_id=sample_id,
            image_paths=image_paths,
            dataset="slidevqa",
        )


def load_dataset_samples(
    dataset_name: str,
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load dataset by name and yield OCRSample objects.

    Args:
        dataset_name: "arxivqa" or "slidevqa".
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name == "arxivqa":
        yield from load_arxivqa(cache_dir, max_samples)
    elif dataset_name == "slidevqa":
        yield from load_slidevqa(cache_dir, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'arxivqa' or 'slidevqa'.")
