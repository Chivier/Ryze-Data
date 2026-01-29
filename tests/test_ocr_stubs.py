"""Unit tests for OCR stub models (DeepSeek, MarkItDown, pdf2md)."""

import tempfile

import pytest

from src.ocr.deepseek_ocr import DeepSeekOCR
from src.ocr.markitdown_ocr import MarkItDownOCR
from src.ocr.pdf2md_ocr import Pdf2MdOCR


class TestDeepSeekOCR:
    """Test DeepSeekOCR stub."""

    def test_model_name(self):
        assert DeepSeekOCR.MODEL_NAME == "deepseek"

    def test_is_available_returns_bool(self):
        result = DeepSeekOCR.is_available()
        assert isinstance(result, bool)

    def test_process_single_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DeepSeekOCR(output_dir=tmpdir)
            with pytest.raises(NotImplementedError, match="stub"):
                model.process_single("/data/test.pdf")

    def test_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DeepSeekOCR(output_dir=tmpdir)
            assert model.name == "DeepSeek-VL2"


class TestMarkItDownOCR:
    """Test MarkItDownOCR stub."""

    def test_model_name(self):
        assert MarkItDownOCR.MODEL_NAME == "markitdown"

    def test_is_available_returns_bool(self):
        result = MarkItDownOCR.is_available()
        assert isinstance(result, bool)

    def test_process_single_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MarkItDownOCR(output_dir=tmpdir)
            with pytest.raises(NotImplementedError, match="stub"):
                model.process_single("/data/test.pdf")

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
