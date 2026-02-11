"""SlideVQA dataset loader for benchmark evaluation.

SlideVQA contains questions about slide deck images.
Each sample has multiple slide images + a question + a free-text answer.
"""

import logging
from pathlib import Path
from typing import List

from src.benchmark.datasets.base import BaseBenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)


class SlideVQADataset(BaseBenchmarkDataset):
    """SlideVQA dataset: multi-slide questions.

    Uses the HuggingFace ``datasets`` library to load
    ``NTT-hil-insight/SlideVQA``.

    Note: This dataset may require accepting a license agreement
    on the HuggingFace Hub before download.
    """

    DATASET_NAME = "slidevqa"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._dataset = None

    def download(self) -> None:
        """Download the SlideVQA dataset via HuggingFace datasets."""
        try:
            from datasets import load_dataset

            logger.info("Downloading SlideVQA dataset...")
            self._dataset = load_dataset("NTT-hil-insight/SlideVQA", split="test")
            logger.info(f"Downloaded {len(self._dataset)} samples")
        except ImportError:
            raise RuntimeError("Install the 'datasets' library: pip install datasets")

    def is_downloaded(self) -> bool:
        """Check if dataset is loaded in memory."""
        return self._dataset is not None

    def load_samples(self, max_samples: int = 0) -> List[BenchmarkSample]:
        """Load SlideVQA samples as BenchmarkSample instances.

        Args:
            max_samples: Maximum samples to load. 0 means all.

        Returns:
            List of BenchmarkSample with question_type="free_text".
        """
        if not self.is_downloaded():
            self.download()

        samples = []
        dataset = self._dataset
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        image_dir = Path(self.data_dir) / "slidevqa_images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(dataset):
            try:
                sample = self._parse_sample(item, idx, image_dir)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {e}")

        logger.info(f"Loaded {len(samples)} SlideVQA samples")
        return samples

    def _parse_sample(self, item, idx: int, image_dir: Path) -> BenchmarkSample:
        """Parse a single HuggingFace dataset item into a BenchmarkSample.

        Args:
            item: A row from the HuggingFace dataset.
            idx: Sample index.
            image_dir: Directory to save extracted images.

        Returns:
            BenchmarkSample instance.
        """
        # Extract slide images — may be a list of PIL images or a single image
        image_paths = []
        images = item.get("images", [])
        if not images:
            # Some dataset versions use "image" (singular)
            single_image = item.get("image")
            if single_image is not None:
                images = [single_image]

        for img_idx, image in enumerate(images):
            img_path = str(image_dir / f"slidevqa_{idx}_slide_{img_idx}.png")
            if not Path(img_path).exists():
                image.save(img_path)
            image_paths.append(img_path)

        question = item.get("question", "")
        answer = item.get("answer", "")

        return BenchmarkSample(
            sample_id=f"slidevqa_{idx}",
            image_paths=image_paths,
            question=question,
            choices=None,
            correct_answer=str(answer),
            question_type="free_text",
            metadata={"dataset": "slidevqa"},
        )
