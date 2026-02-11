"""Unit tests for MarkItDownOCR implementation with mocked markitdown."""

import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.ocr.markitdown_ocr import MarkItDownOCR


class TestMarkItDownOCR:
    """Test MarkItDownOCR with mocked markitdown library."""

    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model(self, output_dir):
        return MarkItDownOCR(output_dir=output_dir)

    def test_model_name(self, model):
        assert model.MODEL_NAME == "markitdown"
        assert model.name == "MarkItDown"

    def test_is_available_with_module(self):
        """When markitdown is importable, is_available returns True."""
        fake_module = ModuleType("markitdown")
        with patch.dict(sys.modules, {"markitdown": fake_module}):
            assert MarkItDownOCR.is_available() is True

    def test_process_single_success(self, model, output_dir):
        """Test successful PDF processing."""
        paper_name = "test_paper"
        pdf_path = f"/data/{paper_name}.pdf"

        # Create a fake markitdown module with MarkItDown class
        mock_result = MagicMock()
        mock_result.text_content = "# Test Paper\nContent here."

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        mock_class = MagicMock(return_value=mock_converter)

        fake_module = ModuleType("markitdown")
        fake_module.MarkItDown = mock_class

        with patch.dict(sys.modules, {"markitdown": fake_module}):
            result = model.process_single(pdf_path)

        assert result.ocr_status == "success"
        assert result.paper_name == paper_name

        md_path = Path(output_dir) / paper_name / f"{paper_name}.md"
        assert md_path.exists()
        assert md_path.read_text() == "# Test Paper\nContent here."

    def test_process_single_conversion_error(self, model, output_dir):
        """Test handling of conversion error."""
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Conversion failed")

        mock_class = MagicMock(return_value=mock_converter)

        fake_module = ModuleType("markitdown")
        fake_module.MarkItDown = mock_class

        with patch.dict(sys.modules, {"markitdown": fake_module}):
            result = model.process_single("/data/error.pdf")

        assert "failed" in result.ocr_status
        assert "Conversion failed" in result.ocr_status

    def test_process_single_output_structure(self, model, output_dir):
        """Test that output follows expected directory structure."""
        mock_result = MagicMock()
        mock_result.text_content = "# Paper"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        mock_class = MagicMock(return_value=mock_converter)

        fake_module = ModuleType("markitdown")
        fake_module.MarkItDown = mock_class

        with patch.dict(sys.modules, {"markitdown": fake_module}):
            result = model.process_single("/data/my_paper.pdf")

        assert result.ocr_status == "success"
        assert result.ocr_result_path == str(Path(output_dir) / "my_paper")

    def test_process_single_import_error(self, model):
        """Test handling when markitdown is not installed."""
        # Remove markitdown from sys.modules so import fails
        with patch.dict(sys.modules, {"markitdown": None}):
            result = model.process_single("/data/test.pdf")
            assert "failed" in result.ocr_status
