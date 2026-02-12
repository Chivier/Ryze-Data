"""Standalone HuggingFace dataset loading and image extraction.

Loads ArxivQA and SlideVQA datasets and yields OCRSample objects with
cached image paths for downstream OCR processing.
"""

import logging
import os
import tarfile
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

    Handles the forms returned by HF datasets:
      - ``PIL.Image.Image`` — already decoded (e.g. SlideVQA page_N cols).
      - ``dict`` with ``bytes`` and/or ``path`` keys.
      - ``str`` — file path.

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
        raw = image.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw))
        path = image.get("path")
        if path:
            path_obj = Path(path)
            if path_obj.exists():
                return Image.open(path_obj)

    if isinstance(image, str):
        path_obj = Path(image)
        if path_obj.exists():
            return Image.open(path_obj)

    return None


def _find_images_tgz() -> Path | None:
    """Find images.tgz in the HuggingFace hub cache for ArxivQA."""
    candidate_hub_roots = []
    if os.environ.get("HF_HUB_CACHE"):
        candidate_hub_roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        candidate_hub_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        candidate_hub_roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    candidate_hub_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    # Deduplicate while preserving order.
    seen = set()
    for root in candidate_hub_roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)

        snapshots_dir = root / "datasets--MMInstruction--ArxivQA" / "snapshots"
        if not snapshots_dir.exists():
            continue

        for snap in snapshots_dir.iterdir():
            tgz = snap / "images.tgz"
            if tgz.exists():
                return tgz.resolve()
    return None


def _load_hf_token() -> str | None:
    """Load HF token from env vars or common token file locations."""
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token and token.strip():
            logger.info("Using HuggingFace token from env var %s", env_name)
            return token.strip()

    candidate_paths = []
    if os.environ.get("HF_TOKEN_PATH"):
        candidate_paths.append(Path(os.environ["HF_TOKEN_PATH"]))
    if os.environ.get("HF_HOME"):
        candidate_paths.append(Path(os.environ["HF_HOME"]) / "token")
    if os.environ.get("XDG_CACHE_HOME"):
        candidate_paths.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "token")
    candidate_paths.extend(
        [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".huggingface" / "token",
        ]
    )

    seen = set()
    for token_path in candidate_paths:
        path_str = str(token_path)
        if path_str in seen:
            continue
        seen.add(path_str)

        if not token_path.exists():
            continue

        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if token:
            logger.info("Using HuggingFace token from %s", token_path)
            return token

    return None


def _load_dataset_with_auth(repo_id: str, split: str):
    """Load HF dataset with explicit auth fallback for changed cache roots."""
    from datasets import load_dataset

    token = _load_hf_token()
    if token:
        try:
            return load_dataset(repo_id, split=split, token=token)
        except TypeError:
            # Backward compatibility with older `datasets` versions.
            return load_dataset(repo_id, split=split, use_auth_token=token)
    return load_dataset(repo_id, split=split)


def _ensure_arxivqa_images(cache_dir: str) -> Path:
    """Ensure ArxivQA raw images are extracted and return the base directory.

    The ArxivQA dataset stores image paths as relative strings
    (e.g. ``images/1501.00713_1.jpg``).  The actual images come from
    ``images.tgz`` which is downloaded by HuggingFace but not
    auto-extracted.

    This function checks for extracted images in ``{cache_dir}/arxivqa_raw_images/``
    and extracts ``images.tgz`` from the HF hub cache if needed.

    Returns:
        Path to the base directory such that ``base / item["image"]``
        resolves to the actual image file.
    """
    raw_dir = Path(cache_dir) / "arxivqa_raw_images"
    images_subdir = raw_dir / "images"

    if images_subdir.exists() and any(images_subdir.iterdir()):
        return raw_dir

    tgz_path = _find_images_tgz()
    if tgz_path is None:
        raise FileNotFoundError(
            "ArxivQA images.tgz not found in HuggingFace cache. "
            "Run `load_dataset('MMInstruction/ArxivQA')` first to download it, "
            "or manually place images.tgz in the HF hub cache."
        )

    logger.info(
        "Extracting ArxivQA images.tgz → %s (one-time, may take a few minutes)...",
        raw_dir,
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tgz_path), "r:gz") as tar:
        tar.extractall(str(raw_dir))
    logger.info("Extraction complete — %d files", sum(1 for _ in images_subdir.iterdir()))
    return raw_dir


def load_arxivqa(
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load ArxivQA dataset and yield OCRSample objects.

    The ``image`` column in ArxivQA is a plain string (e.g.
    ``images/1501.00713_1.jpg``), **not** an HF ``Image`` feature.
    The actual image files come from ``images.tgz`` which is downloaded
    by HuggingFace but not auto-extracted.  This function extracts it
    on first run and resolves the paths.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    logger.info("Loading ArxivQA dataset...")
    dataset = _load_dataset_with_auth("MMInstruction/ArxivQA", split="train")
    logger.info("ArxivQA: %d total samples", len(dataset))

    # Ensure the raw images from images.tgz are available.
    raw_images_base = _ensure_arxivqa_images(cache_dir)

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = Path(cache_dir) / "arxivqa_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    for idx, item in enumerate(dataset):
        sample_id = f"arxivqa_{idx}"
        image_path = str(image_dir / f"{sample_id}.png")

        if not Path(image_path).exists():
            image_val = item.get("image")
            if image_val is None:
                logger.warning("Skipping %s: no image field", sample_id)
                continue

            # image_val is a string like "images/1501.00713_1.jpg"
            if isinstance(image_val, str):
                src = raw_images_base / image_val
                if not src.exists():
                    logger.warning("Skipping %s: image not found at %s", sample_id, src)
                    continue
                pil_image = Image.open(str(src))
            else:
                # Fallback for other types (PIL Image, dict, etc.)
                pil_image = _decode_image(image_val)
                if pil_image is None:
                    logger.warning("Skipping %s: could not decode image", sample_id)
                    continue

            pil_image.save(image_path)

        yield OCRSample(
            sample_id=sample_id,
            image_paths=[image_path],
            dataset="arxivqa",
        )


def _extract_slidevqa_images(item: dict) -> list:
    """Extract SlideVQA images across known schema variants."""
    images = item.get("images")
    if isinstance(images, list) and images:
        return images

    single_image = item.get("image")
    if single_image is not None:
        return [single_image]

    # The official SlideVQA schema stores pages in page_1..page_20 columns.
    page_images = []
    for key, value in item.items():
        if not key.startswith("page_") or value is None:
            continue
        try:
            page_idx = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        page_images.append((page_idx, value))

    page_images.sort(key=lambda pair: pair[0])
    return [image for _, image in page_images]


def load_slidevqa(
    cache_dir: str,
    max_samples: int = 0,
) -> Iterator[OCRSample]:
    """Load SlideVQA dataset and yield OCRSample objects.

    Each sample has multiple slide images stored as HF ``Image``
    features (``page_1`` .. ``page_20``).  The default loading
    behaviour decodes them to PIL Images automatically.

    Args:
        cache_dir: Directory for caching extracted images.
        max_samples: Maximum samples to yield. 0 means all.

    Yields:
        OCRSample for each dataset item.
    """
    logger.info("Loading SlideVQA dataset...")
    dataset = _load_dataset_with_auth("NTT-hil-insight/SlideVQA", split="test")
    logger.info("SlideVQA: %d total samples", len(dataset))

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    image_dir = Path(cache_dir) / "slidevqa_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(dataset):
        sample_id = f"slidevqa_{idx}"

        # Extract slide images from known schema variants.
        images = _extract_slidevqa_images(item)

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
