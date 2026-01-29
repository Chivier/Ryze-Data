"""MarkItDown OCR stub implementation."""

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class MarkItDownOCR(BaseOCRModel):
    """OCR model using Microsoft MarkItDown for document conversion.

    This is a stub. Install ``markitdown`` and implement
    ``process_single`` to enable this model.
    """

    MODEL_NAME = "markitdown"

    @property
    def name(self) -> str:
        return "MarkItDown"

    @classmethod
    def is_available(cls) -> bool:
        """Check if markitdown is installed."""
        try:
            import markitdown  # noqa: F401

            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF with MarkItDown.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            OCRResult with processing outcome.

        Raises:
            NotImplementedError: This model is not yet implemented.
        """
        raise NotImplementedError(
            "MarkItDownOCR is a stub. Implement process_single() to use it."
        )
