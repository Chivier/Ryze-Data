"""Unit tests for OCR stub models (MarkItDown, pdf2md)."""

import sys
import tempfile
from types import ModuleType

import pytest
from unittest.mock import MagicMock, patch

from src.ocr.markitdown_ocr import MarkItDownOCR
from src.ocr.pdf2md_ocr import Pdf2MdOCR


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


class TestPdf2MdOCR:
    """Test Pdf2MdOCR stub."""

    def test_model_name(self):
        assert Pdf2MdOCR.MODEL_NAME == "pdf2md"

    def test_is_available_returns_bool(self):
        result = Pdf2MdOCR.is_available()
        assert isinstance(result, bool)

    def test_process_single_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Pdf2MdOCR(output_dir=tmpdir)
            with pytest.raises(NotImplementedError, match="stub"):
                model.process_single("/data/test.pdf")

    def test_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Pdf2MdOCR(output_dir=tmpdir)
            assert model.name == "pdf2md"
