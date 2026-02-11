"""MarkItDown OCR implementation using Microsoft's markitdown library."""

from pathlib import Path

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class MarkItDownOCR(BaseOCRModel):
    """OCR model using Microsoft MarkItDown for PDF-to-Markdown conversion.

    Uses the ``markitdown`` library to convert PDF files to Markdown.
    Install with: ``pip install markitdown``
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
        """
        paper_name = Path(pdf_path).stem
        paper_output_dir = self.output_dir / paper_name

        try:
            from markitdown import MarkItDown

            paper_output_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Processing {paper_name} with MarkItDown")
            converter = MarkItDown()
            result = converter.convert(pdf_path)

            md_path = paper_output_dir / f"{paper_name}.md"
            md_path.write_text(result.text_content, encoding="utf-8")

            return self._make_result(
                pdf_path,
                status="success",
                result_path=str(paper_output_dir),
            )

        except ImportError:
            self.logger.error("markitdown is not installed")
            return self._make_result(
                pdf_path, status="failed: markitdown not installed"
            )
        except Exception as e:
            self.logger.error(f"Failed to process {paper_name}: {e}")
            return self._make_result(pdf_path, status=f"failed: {e}")
