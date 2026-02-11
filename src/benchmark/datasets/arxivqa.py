"""ArxivQA dataset loader for benchmark evaluation.

ArxivQA contains figure-based multiple-choice questions from arXiv papers.
Each sample has a single figure image + MC question (2-10 options) + correct label.
"""

import logging
from pathlib import Path
from typing import List

from src.benchmark.datasets.base import BaseBenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)


class ArxivQADataset(BaseBenchmarkDataset):
    """ArxivQA dataset: figure-based MC questions from arXiv papers.

    Uses the HuggingFace ``datasets`` library to load
    ``MMInstruction/ArxivQA``.
    """

    DATASET_NAME = "arxivqa"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._dataset = None

    def download(self) -> None:
        """Download the ArxivQA dataset via HuggingFace datasets."""
        try:
            from datasets import load_dataset

            logger.info("Downloading ArxivQA dataset...")
            self._dataset = load_dataset("MMInstruction/ArxivQA", split="train")
            logger.info(f"Downloaded {len(self._dataset)} samples")
        except ImportError:
            raise RuntimeError("Install the 'datasets' library: pip install datasets")

    def is_downloaded(self) -> bool:
        """Check if dataset is loaded in memory."""
        return self._dataset is not None

    def load_samples(self, max_samples: int = 0) -> List[BenchmarkSample]:
        """Load ArxivQA samples as BenchmarkSample instances.

        Args:
            max_samples: Maximum samples to load. 0 means all.

        Returns:
            List of BenchmarkSample with question_type="multiple_choice".
        """
        if not self.is_downloaded():
            self.download()

        samples = []
        dataset = self._dataset
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        image_dir = Path(self.data_dir) / "arxivqa_images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(dataset):
            try:
                sample = self._parse_sample(item, idx, image_dir)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Skipping sample {idx}: {e}")

        logger.info(f"Loaded {len(samples)} ArxivQA samples")
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
        # Extract image
        image = item.get("image")
        image_path = str(image_dir / f"arxivqa_{idx}.png")
        if image is not None and not Path(image_path).exists():
            image.save(image_path)

        # Extract question and choices
        question = item.get("question", "")
        options = item.get("options", [])
        label = item.get("label", "")

        # Map label letter to full option text
        if options and label:
            label_idx = ord(label.upper()) - ord("A")
            if 0 <= label_idx < len(options):
                correct_answer = options[label_idx]
            else:
                correct_answer = label
        else:
            correct_answer = label

        return BenchmarkSample(
            sample_id=f"arxivqa_{idx}",
            image_paths=[image_path],
            question=question,
            choices=options if options else None,
            correct_answer=correct_answer,
            question_type="multiple_choice",
            metadata={"dataset": "arxivqa", "label": label},
        )
