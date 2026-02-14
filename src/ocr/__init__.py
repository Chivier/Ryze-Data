"""OCR module for PDF-to-Markdown conversion with extensible model support."""

# Import concrete models to trigger @OCRRegistry.register decorators.
import src.ocr.deepseek_ocr  # noqa: F401  (v1, vLLM backend)
import src.ocr.glm_ocr  # noqa: F401  (GLM-OCR, vLLM + Z.AI API)
import src.ocr.marker_ocr  # noqa: F401
import src.ocr.markitdown_ocr  # noqa: F401
import src.ocr.paddle_ocr  # noqa: F401  (PaddleOCR)
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
