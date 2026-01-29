"""Unit tests for DeepSeek OCR v1 and v2 (mocked -- no GPU needed)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.ocr.deepseek_ocr import DeepSeekOCRv1
from src.ocr.deepseek_ocr_v2 import DeepSeekOCRv2
from src.ocr.registry import OCRRegistry

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_model(cls, tmpdir: str):
    """Instantiate a DeepSeek OCR model pointing at *tmpdir*."""
    return cls(output_dir=tmpdir)


def _setup_mock_inference(model, page_texts: list[str], tmpdir: str):
    """Patch model internals so ``process_single`` runs without GPU.

    Args:
        model: The DeepSeek OCR model instance.
        page_texts: Markdown strings to return per page.
        tmpdir: Path to the output directory.
    """
    model._model = MagicMock()
    model._tokenizer = MagicMock()

    # _ensure_model_loaded is a no-op since _model is already set
    # _pdf_to_images returns fake page paths
    fake_pages = []
    temp_dir = Path(tmpdir) / "fake_paper" / "temp_pages"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for i, _ in enumerate(page_texts):
        p = temp_dir / f"page_{i:04d}.png"
        p.write_bytes(b"fake png")
        fake_pages.append(p)

    model._pdf_to_images = MagicMock(return_value=fake_pages)

    # _infer_single_image returns the text for each page in sequence
    model._infer_single_image = MagicMock(side_effect=page_texts)


# ------------------------------------------------------------------
# TestDeepSeekOCRv1
# ------------------------------------------------------------------


class TestDeepSeekOCRv1:
    """Tests for DeepSeek-OCR v1."""

    def test_model_name(self):
        assert DeepSeekOCRv1.MODEL_NAME == "deepseek-ocr"

    def test_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv1, tmpdir)
            assert model.name == "DeepSeek-OCR v1"

    def test_hf_model_id(self):
        assert DeepSeekOCRv1.HF_MODEL_ID == "deepseek-ai/DeepSeek-OCR"

    def test_image_size(self):
        assert DeepSeekOCRv1.IMAGE_SIZE == 640

    def test_include_test_compress(self):
        assert DeepSeekOCRv1.INCLUDE_TEST_COMPRESS is True

    def test_is_available_returns_bool(self):
        result = DeepSeekOCRv1.is_available()
        assert isinstance(result, bool)

    def test_process_single_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv1, tmpdir)
            _setup_mock_inference(model, ["# Page 1", "# Page 2"], tmpdir)

            pdf_path = str(Path(tmpdir) / "fake_paper.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert result.ocr_status == "success"
            assert result.paper_name == "fake_paper"
            md_path = Path(result.ocr_result_path) / "fake_paper.md"
            assert md_path.exists()
            content = md_path.read_text()
            assert "# Page 1" in content
            assert "# Page 2" in content

    def test_process_single_no_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv1, tmpdir)
            model._model = MagicMock()
            model._tokenizer = MagicMock()
            model._pdf_to_images = MagicMock(return_value=[])

            pdf_path = str(Path(tmpdir) / "empty.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert "no pages" in result.ocr_status

    def test_process_single_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv1, tmpdir)
            model._model = MagicMock()
            model._tokenizer = MagicMock()
            model._pdf_to_images = MagicMock(side_effect=RuntimeError("boom"))

            pdf_path = str(Path(tmpdir) / "bad.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert result.ocr_status.startswith("failed:")
            assert "boom" in result.ocr_status


# ------------------------------------------------------------------
# TestDeepSeekOCRv2
# ------------------------------------------------------------------


class TestDeepSeekOCRv2:
    """Tests for DeepSeek-OCR v2."""

    def test_model_name(self):
        assert DeepSeekOCRv2.MODEL_NAME == "deepseek-ocr-v2"

    def test_name_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv2, tmpdir)
            assert model.name == "DeepSeek-OCR v2"

    def test_hf_model_id(self):
        assert DeepSeekOCRv2.HF_MODEL_ID == "deepseek-ai/DeepSeek-OCR-2"

    def test_image_size(self):
        assert DeepSeekOCRv2.IMAGE_SIZE == 768

    def test_include_test_compress(self):
        assert DeepSeekOCRv2.INCLUDE_TEST_COMPRESS is False

    def test_is_available_returns_bool(self):
        result = DeepSeekOCRv2.is_available()
        assert isinstance(result, bool)

    def test_process_single_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv2, tmpdir)
            _setup_mock_inference(model, ["## Section A", "## Section B"], tmpdir)

            pdf_path = str(Path(tmpdir) / "fake_paper.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert result.ocr_status == "success"
            md_path = Path(result.ocr_result_path) / "fake_paper.md"
            assert md_path.exists()
            content = md_path.read_text()
            assert "## Section A" in content
            assert "## Section B" in content

    def test_process_single_no_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv2, tmpdir)
            model._model = MagicMock()
            model._tokenizer = MagicMock()
            model._pdf_to_images = MagicMock(return_value=[])

            pdf_path = str(Path(tmpdir) / "empty.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert "no pages" in result.ocr_status

    def test_process_single_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv2, tmpdir)
            model._model = MagicMock()
            model._tokenizer = MagicMock()
            model._pdf_to_images = MagicMock(side_effect=RuntimeError("crash"))

            pdf_path = str(Path(tmpdir) / "bad.pdf")
            Path(pdf_path).write_bytes(b"%PDF-1.4")

            result = model.process_single(pdf_path)

            assert "crash" in result.ocr_status


# ------------------------------------------------------------------
# TestDeepSeekInferKwargs
# ------------------------------------------------------------------


class TestDeepSeekInferKwargs:
    """Verify that v1 passes test_compress and v2 does not."""

    def test_v1_passes_test_compress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv1, tmpdir)
            model._model = MagicMock()
            model._model.infer = MagicMock(return_value="md content")
            model._tokenizer = MagicMock()

            img_path = Path(tmpdir) / "page.png"
            img_path.write_bytes(b"fake")
            out_path = Path(tmpdir) / "out"
            out_path.mkdir()

            model._infer_single_image(img_path, out_path)

            call_kwargs = model._model.infer.call_args
            assert call_kwargs.kwargs.get("test_compress") is True

    def test_v2_omits_test_compress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_model(DeepSeekOCRv2, tmpdir)
            model._model = MagicMock()
            model._model.infer = MagicMock(return_value="md content")
            model._tokenizer = MagicMock()

            img_path = Path(tmpdir) / "page.png"
            img_path.write_bytes(b"fake")
            out_path = Path(tmpdir) / "out"
            out_path.mkdir()

            model._infer_single_image(img_path, out_path)

            call_kwargs = model._model.infer.call_args
            assert "test_compress" not in call_kwargs.kwargs


# ------------------------------------------------------------------
# TestRegistration
# ------------------------------------------------------------------


class TestRegistration:
    """Both models must appear in the registry."""

    def test_deepseek_ocr_v1_registered(self):
        assert "deepseek-ocr" in OCRRegistry.list_all()

    def test_deepseek_ocr_v2_registered(self):
        assert "deepseek-ocr-v2" in OCRRegistry.list_all()
