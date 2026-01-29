"""DeepSeek-OCR v1 implementation using local HuggingFace inference."""

from src.ocr.deepseek_base import BaseDeepSeekOCR
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class DeepSeekOCRv1(BaseDeepSeekOCR):
    """OCR model using DeepSeek-OCR (v1) for vision-based PDF extraction.

    Uses ``deepseek-ai/DeepSeek-OCR`` with 640px input images and
    ``test_compress=True`` for inference.
    """

    MODEL_NAME = "deepseek-ocr"
    HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
    IMAGE_SIZE = 640
    INCLUDE_TEST_COMPRESS = True

    @property
    def name(self) -> str:
        return "DeepSeek-OCR v1"
