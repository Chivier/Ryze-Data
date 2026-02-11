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


def load_arxivqa(
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load ArxivQA dataset and yield OCRSample objects.

    Each sample has a single figure image.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    from datasets import load_dataset
    from PIL import Image

    logger.info("Loading ArxivQA dataset...")
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    logger.info("ArxivQA: %d total samples", len(dataset))

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = Path(cache_dir) / "arxivqa_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset):
        sample_id = f"arxivqa_{idx}"
        image_path = str(image_dir / f"{sample_id}.png")

        # Extract image (may be PIL Image or string path)
        image = item.get("image")
        if image is not None and not Path(image_path).exists():
            if isinstance(image, str):
                # String path — copy or symlink
                import shutil

                if Path(image).exists():
                    shutil.copy2(image, image_path)
                else:
                    logger.warning("Image path not found for %s: %s", sample_id, image)
                    continue
            elif isinstance(image, Image.Image):
                image.save(image_path)
            else:
                logger.warning("Unknown image type for %s: %s", sample_id, type(image))
                continue

        if not Path(image_path).exists():
            logger.warning("Skipping %s: image not found at %s", sample_id, image_path)
            continue

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

    Each sample has multiple slide images.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    from datasets import load_dataset

    logger.info("Loading SlideVQA dataset...")
    dataset = load_dataset("NTT-hil-insight/SlideVQA", split="test")
    logger.info("SlideVQA: %d total samples", len(dataset))

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
                image.save(img_path)
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
