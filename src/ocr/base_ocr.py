"""Base class for OCR models with shared functionality."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

# ---------------------------------------------------------------------------
# Shared image resize utilities
# ---------------------------------------------------------------------------

MAX_IMAGE_SIZE = (1024, 1024)


def resize_image(
    image_path: str,
    max_size: tuple[int, int] = MAX_IMAGE_SIZE,
) -> str:
    """Resize an image to fit within *max_size* (preserving aspect ratio).

    If the image already fits, the original path is returned unchanged.
    Otherwise a ``*.resized.png`` copy is written next to the original.
    """
    from PIL import Image

    img = Image.open(image_path)
    if img.width <= max_size[0] and img.height <= max_size[1]:
        return image_path
    img.thumbnail(max_size, Image.LANCZOS)
    out = str(Path(image_path).with_suffix(".resized.png"))
    img.save(out)
    return out


def resize_images(
    image_paths: list[str],
    work_dir: str,
    max_size: tuple[int, int] = MAX_IMAGE_SIZE,
) -> list[str]:
    """Batch-resize images into *work_dir*, preserving aspect ratio.

    Images that already fit are returned as-is.  Oversized images are saved
    as resized copies inside *work_dir*.
    """
    from PIL import Image

    resized: list[str] = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        if img.width <= max_size[0] and img.height <= max_size[1]:
            resized.append(path)
            continue
        img.thumbnail(max_size, Image.LANCZOS)
        out = os.path.join(work_dir, f"resized_{i}_{Path(path).name}")
        img.save(out)
        resized.append(out)
    return resized


@dataclass
class OCRResult:
    """Result of processing a single PDF through OCR.

    Fields match the existing CSV columns from chunked-ocr.py exactly.
    """

    paper_name: str
    original_pdf_path: str
    ocr_status: str
    ocr_time: str
    ocr_result_path: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for CSV serialization."""
        return asdict(self)

    @staticmethod
    def make_result(
        pdf_path: str,
        status: str = "failed",
        result_path: str = "",
    ) -> "OCRResult":
        """Factory method to create an OCRResult with timestamp.

        Args:
            pdf_path: Path to the source PDF file.
            status: OCR processing status string.
            result_path: Path to OCR output directory.

        Returns:
            New OCRResult instance.
        """
        return OCRResult(
            paper_name=Path(pdf_path).stem,
            original_pdf_path=pdf_path,
            ocr_status=status,
            ocr_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ocr_result_path=result_path,
        )


class BaseOCRModel(ABC):
    """Abstract base class for OCR model implementations.

    Subclasses must define MODEL_NAME as a class attribute and implement:
        - name (property): Human-readable model name.
        - process_single(): Process a single PDF file.
        - is_available(): Check if the model's dependencies are installed.

    Optional overrides:
        - supports_batch(): Whether batch processing is natively supported.
        - process_batch(): Batch processing implementation.
    """

    MODEL_NAME: str = ""

    def __init__(self, output_dir: str):
        """Initialize the OCR model.

        Args:
            output_dir: Directory to write OCR output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging for this model."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def create_progress(self) -> Progress:
        """Create a rich Progress instance for tracking."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this OCR model."""

    @abstractmethod
    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF file.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            OCRResult with processing outcome.
        """

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check whether this model's dependencies are installed.

        Returns:
            True if the model can be used, False otherwise.
        """

    def supports_batch(self) -> bool:
        """Whether this model supports native batch processing.

        Returns:
            False by default. Override to return True if the model
            can process multiple files more efficiently in a batch.
        """
        return False

    def process_batch(
        self,
        pdf_paths: List[str],
        gpu_count: int = 1,
        workers_per_gpu: int = 1,
    ) -> List[OCRResult]:
        """Process multiple PDF files.

        Default implementation calls process_single() sequentially.
        Override for models that support native batch processing.

        Args:
            pdf_paths: List of absolute paths to PDF files.
            gpu_count: Number of GPUs available.
            workers_per_gpu: Workers to use per GPU.

        Returns:
            List of OCRResult for each PDF.
        """
        results = []
        for pdf_path in pdf_paths:
            result = self.process_single(pdf_path)
            results.append(result)
        return results

    def _make_result(
        self,
        pdf_path: str,
        status: str = "failed",
        result_path: str = "",
    ) -> OCRResult:
        """Convenience wrapper for OCRResult.make_result.

        Args:
            pdf_path: Path to the source PDF file.
            status: OCR processing status string.
            result_path: Path to OCR output directory.

        Returns:
            New OCRResult instance.
        """
        return OCRResult.make_result(pdf_path, status, result_path)

    def get_paper_output_dir(self, pdf_path: str) -> Path:
        """Get the output directory for a paper.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Path to the paper-specific output directory.
        """
        paper_name = Path(pdf_path).stem
        return self.output_dir / paper_name
