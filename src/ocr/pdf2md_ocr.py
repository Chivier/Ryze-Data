"""pdf2md OCR stub implementation."""

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class Pdf2MdOCR(BaseOCRModel):
    """OCR model using pdf2md for PDF-to-Markdown conversion.

    This is a stub. Install ``pdf2md`` and implement
    ``process_single`` to enable this model.
    """

    MODEL_NAME = "pdf2md"

    @property
    def name(self) -> str:
        return "pdf2md"

    @classmethod
    def is_available(cls) -> bool:
        """Check if pdf2md is installed."""
        try:
            import pdf2md  # noqa: F401

            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF with pdf2md.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            OCRResult with processing outcome.

        Raises:
            NotImplementedError: This model is not yet implemented.
        """
        raise NotImplementedError(
            "Pdf2MdOCR is a stub. Implement process_single() to use it."
        )
