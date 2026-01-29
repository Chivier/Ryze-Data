"""DeepSeek-OCR v2 implementation using local HuggingFace inference."""

from src.ocr.deepseek_base import BaseDeepSeekOCR
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class DeepSeekOCRv2(BaseDeepSeekOCR):
    """OCR model using DeepSeek-OCR-2 (v2) for vision-based PDF extraction.

    Uses ``deepseek-ai/DeepSeek-OCR-2`` with 768px input images.
    """

    MODEL_NAME = "deepseek-ocr-v2"
    HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
    IMAGE_SIZE = 768
    INCLUDE_TEST_COMPRESS = False

    @property
    def name(self) -> str:
        return "DeepSeek-OCR v2"
