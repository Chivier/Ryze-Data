"""Base data structures and abstract dataset class for benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkSample:
    """A single benchmark evaluation sample.

    Attributes:
        sample_id: Unique identifier for this sample.
        image_paths: Source image paths (1 for ArxivQA, N for SlideVQA).
        question: The question text.
        choices: Multiple-choice options, or None for free-text.
        correct_answer: The ground-truth answer.
        question_type: Either "multiple_choice" or "free_text".
        metadata: Additional sample-specific metadata.
    """

    sample_id: str
    image_paths: List[str]
    question: str
    choices: Optional[List[str]]
    correct_answer: str
    question_type: str  # "multiple_choice" | "free_text"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmarkDataset(ABC):
    """Abstract base class for benchmark dataset loaders.

    Subclasses must define DATASET_NAME and implement download(),
    load_samples(), and is_downloaded().
    """

    DATASET_NAME: str = ""

    def __init__(self, data_dir: str):
        """Initialize the dataset loader.

        Args:
            data_dir: Root directory for storing dataset files.
        """
        self.data_dir = data_dir

    @abstractmethod
    def download(self) -> None:
        """Download the dataset to data_dir."""

    @abstractmethod
    def load_samples(self, max_samples: int = 0) -> List[BenchmarkSample]:
        """Load benchmark samples from the dataset.

        Args:
            max_samples: Maximum number of samples to load.
                0 means load all.

        Returns:
            List of BenchmarkSample instances.
        """

    @abstractmethod
    def is_downloaded(self) -> bool:
        """Check whether the dataset has been downloaded.

        Returns:
            True if data files exist locally.
        """
