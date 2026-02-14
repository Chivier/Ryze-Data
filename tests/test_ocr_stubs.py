"""Unit tests for OCR stub models (MarkItDown)."""

import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

from src.ocr.markitdown_ocr import MarkItDownOCR


class TestMarkItDownOCR:
    """Test MarkItDownOCR (now a full implementation)."""

    def test_model_name(self):
        assert MarkItDownOCR.MODEL_NAME == "markitdown"

    def test_is_available_returns_bool(self):
        result = MarkItDownOCR.is_available()
        assert isinstance(result, bool)

    def test_process_single_without_markitdown(self):
        """Without markitdown installed, process_single returns a failed result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MarkItDownOCR(output_dir=tmpdir)
            with patch.dict(sys.modules, {"markitdown": None}):
                result = model.process_single("/data/test.pdf")
                assert "failed" in result.ocr_status

    def test_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MarkItDownOCR(output_dir=tmpdir)
            assert model.name == "MarkItDown"
