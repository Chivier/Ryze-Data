"""DeepSeek OCR stub implementation."""

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class DeepSeekOCR(BaseOCRModel):
    """OCR model using DeepSeek-VL2 for vision-based PDF extraction.

    This is a stub. Install ``deepseek-vl2`` and implement
    ``process_single`` to enable this model.
    """

    MODEL_NAME = "deepseek"

    @property
    def name(self) -> str:
        return "DeepSeek-VL2"

    @classmethod
    def is_available(cls) -> bool:
        """Check if deepseek-vl2 is installed."""
        try:
            import deepseek_vl2  # noqa: F401

            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF with DeepSeek-VL2.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            OCRResult with processing outcome.

        Raises:
            NotImplementedError: This model is not yet implemented.
        """
        raise NotImplementedError(
            "DeepSeekOCR is a stub. Implement process_single() to use it."
        )
