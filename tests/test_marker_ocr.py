"""Unit tests for MarkerOCR implementation with mocked subprocess."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ocr.marker_ocr import MarkerOCR


class TestMarkerOCR:
    """Test MarkerOCR with mocked subprocess calls."""

    @pytest.fixture
    def output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model(self, output_dir):
        return MarkerOCR(output_dir=output_dir)

    def test_model_name(self, model):
        assert model.MODEL_NAME == "marker"
        assert model.name == "Marker"

    @patch("shutil.which", return_value="/usr/bin/marker_single")
    def test_is_available_true(self, mock_which):
        assert MarkerOCR.is_available() is True

    @patch("shutil.which", return_value=None)
    def test_is_available_false(self, mock_which):
        assert MarkerOCR.is_available() is False

    @patch("subprocess.run")
    def test_process_single_success(self, mock_run, model, output_dir):
        """Test successful single PDF processing."""
        paper_name = "test_paper"
        pdf_path = f"/data/{paper_name}.pdf"

        # Create expected output
        paper_out = Path(output_dir) / paper_name
        paper_out.mkdir()
        (paper_out / f"{paper_name}.md").write_text("# Test Paper\nContent here.")

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = model.process_single(pdf_path)
        assert result.ocr_status == "success"
        assert result.paper_name == paper_name
        assert result.ocr_result_path == str(paper_out)

    @patch("subprocess.run")
    def test_process_single_failure_nonzero_exit(self, mock_run, model):
        """Test handling of non-zero exit code."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr="marker error", stdout=""
        )

        result = model.process_single("/data/fail.pdf")
        assert "failed" in result.ocr_status
        assert result.paper_name == "fail"

    @patch("subprocess.run")
    def test_process_single_no_output_dir(self, mock_run, model):
        """Test handling when output directory is not created."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        result = model.process_single("/data/missing.pdf")
        assert "failed" in result.ocr_status

    @patch("subprocess.run", side_effect=FileNotFoundError("marker_single not found"))
    def test_process_single_exception(self, mock_run, model):
        """Test handling of subprocess exception."""
        result = model.process_single("/data/error.pdf")
        assert "failed" in result.ocr_status
        assert "marker_single not found" in result.ocr_status

    @patch("shutil.which", return_value="/usr/bin/marker_chunk_convert")
    def test_supports_batch_true(self, mock_which, model):
        assert model.supports_batch() is True

    @patch("shutil.which", return_value=None)
    def test_supports_batch_false(self, mock_which, model):
        assert model.supports_batch() is False

    @patch("subprocess.run")
    @patch("shutil.which", return_value="/usr/bin/marker_chunk_convert")
    def test_process_batch_multi_gpu(self, mock_which, mock_run, model, output_dir):
        """Test batch processing with multiple GPUs."""
        # Create real temp PDF files (shutil.copy2 needs them to exist)
        import tempfile

        with tempfile.TemporaryDirectory() as pdf_dir:
            pdf_a = Path(pdf_dir) / "a.pdf"
            pdf_b = Path(pdf_dir) / "b.pdf"
            pdf_a.write_bytes(b"%PDF-1.4 test a")
            pdf_b.write_bytes(b"%PDF-1.4 test b")

            pdfs = [str(pdf_a), str(pdf_b)]

            # Create expected output
            for name in ["a", "b"]:
                out_dir = Path(output_dir) / name
                out_dir.mkdir()
                (out_dir / f"{name}.md").write_text(f"# {name}")

            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            results = model.process_batch(pdfs, gpu_count=2, workers_per_gpu=2)
            assert len(results) == 2
            assert all(r.ocr_status == "success" for r in results)

    @patch("subprocess.run")
    def test_process_batch_single_gpu_fallback(self, mock_run, model, output_dir):
        """Test batch with single GPU falls back to sequential."""
        pdfs = ["/data/a.pdf"]

        # Create expected output
        out_dir = Path(output_dir) / "a"
        out_dir.mkdir()
        (out_dir / "a.md").write_text("# a")

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        results = model.process_batch(pdfs, gpu_count=1, workers_per_gpu=1)
        assert len(results) == 1

    @patch("shutil.which", return_value="/usr/bin/marker_chunk_convert")
    def test_process_batch_exception(self, mock_which, model):
        """Test batch processing with exception from missing source files."""
        results = model.process_batch(
            ["/nonexistent/a.pdf", "/nonexistent/b.pdf"],
            gpu_count=2,
            workers_per_gpu=1,
        )
        assert len(results) == 2
        assert all("failed" in r.ocr_status for r in results)
