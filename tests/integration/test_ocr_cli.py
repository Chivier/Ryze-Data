"""Integration tests for OCR CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dirs():
    """Create input/output directories with a dummy PDF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create a dummy PDF file
        pdf = input_dir / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4\ntest content")

        yield {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "config_dir": tmpdir,
        }


@pytest.fixture(autouse=True)
def mock_config_validation():
    """Mock config.validate() so the CLI group callback succeeds."""
    with patch("src.config_manager.ConfigManager.validate", return_value=True):
        yield


class TestListOCRModels:
    """Test the list-ocr-models CLI command."""

    def test_list_ocr_models(self, runner):
        """list-ocr-models should show registered models."""
        result = runner.invoke(cli, ["list-ocr-models"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "marker" in result.output

    def test_list_ocr_models_shows_stubs(self, runner):
        """list-ocr-models should show stub models."""
        result = runner.invoke(cli, ["list-ocr-models"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "deepseek-ocr" in result.output
        assert "markitdown" in result.output
        assert "pdf2md" in result.output


class TestOCRCommand:
    """Test the ocr CLI command with mocked model."""

    @patch("src.ocr.detect_devices")
    @patch("src.ocr.registry.OCRRegistry.get_model")
    def test_ocr_command_runs(self, mock_get_model, mock_detect, runner, temp_dirs):
        """OCR command should invoke model.process_single for each PDF."""
        from src.ocr.base_ocr import OCRResult

        mock_detect.return_value = (0, {0: 2})

        mock_model = MagicMock()
        mock_model.supports_batch.return_value = False
        mock_model.process_single.return_value = OCRResult.make_result(
            pdf_path=str(Path(temp_dirs["input_dir"]) / "sample.pdf"),
            status="success",
            result_path=str(Path(temp_dirs["output_dir"]) / "sample"),
        )
        mock_get_model.return_value = mock_model

        result = runner.invoke(
            cli,
            [
                "--config",
                "config.json",
                "ocr",
                "--input-dir",
                temp_dirs["input_dir"],
                "--output-dir",
                temp_dirs["output_dir"],
                "--ocr-model",
                "marker",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "OCR processing completed" in result.output
        mock_model.process_single.assert_called_once()

    @patch("src.ocr.detect_devices")
    @patch("src.ocr.registry.OCRRegistry.get_model")
    def test_ocr_command_batch_mode(
        self, mock_get_model, mock_detect, runner, temp_dirs
    ):
        """OCR command should use batch mode with multiple GPUs."""
        from src.ocr.base_ocr import OCRResult

        mock_detect.return_value = (2, {0: 2, 1: 2})

        mock_model = MagicMock()
        mock_model.supports_batch.return_value = True
        mock_model.process_batch.return_value = [
            OCRResult.make_result(
                pdf_path=str(Path(temp_dirs["input_dir"]) / "sample.pdf"),
                status="success",
                result_path=str(Path(temp_dirs["output_dir"]) / "sample"),
            )
        ]
        mock_get_model.return_value = mock_model

        result = runner.invoke(
            cli,
            [
                "--config",
                "config.json",
                "ocr",
                "--input-dir",
                temp_dirs["input_dir"],
                "--output-dir",
                temp_dirs["output_dir"],
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_model.process_batch.assert_called_once()

    def test_ocr_command_unknown_model(self, runner, temp_dirs):
        """OCR command should fail with unknown model name."""
        result = runner.invoke(
            cli,
            [
                "--config",
                "config.json",
                "ocr",
                "--input-dir",
                temp_dirs["input_dir"],
                "--output-dir",
                temp_dirs["output_dir"],
                "--ocr-model",
                "nonexistent",
            ],
        )

        assert result.exit_code != 0

    def test_ocr_command_no_pdfs(self, runner):
        """OCR command should fail with empty input directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_input = Path(tmpdir) / "empty"
            empty_input.mkdir()

            result = runner.invoke(
                cli,
                [
                    "--config",
                    "config.json",
                    "ocr",
                    "--input-dir",
                    str(empty_input),
                    "--output-dir",
                    str(Path(tmpdir) / "output"),
                ],
            )
            assert result.exit_code != 0
