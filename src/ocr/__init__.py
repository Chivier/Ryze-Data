"""OCR module for PDF-to-Markdown conversion with extensible model support."""

import src.ocr.deepseek_ocr  # noqa: F401
import src.ocr.deepseek_ocr_v2  # noqa: F401

# Import concrete models to trigger @OCRRegistry.register decorators.
import src.ocr.marker_ocr  # noqa: F401
import src.ocr.markitdown_ocr  # noqa: F401
import src.ocr.pdf2md_ocr  # noqa: F401
from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.device_utils import detect_devices
from src.ocr.registry import OCRRegistry
from src.ocr.status_tracker import OCRStatusTracker

__all__ = [
    "BaseOCRModel",
    "OCRResult",
    "OCRRegistry",
    "OCRStatusTracker",
    "detect_devices",
]
