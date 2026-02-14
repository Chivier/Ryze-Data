"""Shared base class for DeepSeek OCR models."""

import shutil
from pathlib import Path

from src.ocr.base_ocr import BaseOCRModel, OCRResult, resize_image


class BaseDeepSeekOCR(BaseOCRModel):
    """Base class with shared logic for DeepSeek-OCR local inference.

    Subclasses must set:
        - MODEL_NAME: Registry key (e.g. ``"deepseek-ocr"``).
        - HF_MODEL_ID: HuggingFace model identifier.
        - IMAGE_SIZE: Input image size for the model (pixels).
        - INCLUDE_TEST_COMPRESS: Whether to pass ``test_compress=True``
          to ``model.infer``.

    The model is loaded lazily on the first call to ``process_single``
    to avoid allocating GPU memory at import time.  Subclasses may
    override ``_ensure_model_loaded`` and ``_infer_single_image`` to
    switch backends (e.g. vLLM instead of transformers).
    """

    HF_MODEL_ID: str = ""
    IMAGE_SIZE: int = 640
    INCLUDE_TEST_COMPRESS: bool = False

    DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    MAX_TOKENS: int = 8192
    DPI = 200
    BASE_SIZE = 448

    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy model loading (overridable by subclasses)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load model on first use.  Override for alternative backends."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.logger.info("Loading %s from %s ...", self.name, self.HF_MODEL_ID)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_MODEL_ID, trust_remote_code=True
        )

        attn_impl = self._resolve_attention_implementation()

        self._model = AutoModelForCausalLM.from_pretrained(
            self.HF_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).cuda()

        self.logger.info(
            "Model loaded (attn=%s, dtype=bfloat16, device=cuda)",
            attn_impl,
        )

    @staticmethod
    def _resolve_attention_implementation() -> str:
        """Return the best available attention implementation."""
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            return "eager"

    # ------------------------------------------------------------------
    # PDF → images (with resize)
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: str, temp_dir: Path) -> list[Path]:
        """Convert a PDF to page images, resizing oversized pages.

        Args:
            pdf_path: Path to the source PDF file.
            temp_dir: Directory to store intermediate page PNGs.

        Returns:
            Sorted list of page image paths.
        """
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, dpi=self.DPI)
        paths: list[Path] = []
        for idx, img in enumerate(images):
            img_path = temp_dir / f"page_{idx:04d}.png"
            img.save(str(img_path), "PNG")
            resized = resize_image(str(img_path))
            paths.append(Path(resized))
        return paths

    # ------------------------------------------------------------------
    # Per-page inference (overridable by subclasses)
    # ------------------------------------------------------------------

    def _infer_single_image(self, image_path: Path, output_path: Path) -> str:
        """Run OCR inference on a single page image.

        Default implementation uses the transformers ``model.infer()``
        API.  Override this method for alternative backends (e.g. vLLM).
        """
        kwargs = dict(
            prompt=self.DEFAULT_PROMPT,
            image_file=str(image_path),
            output_path=str(output_path),
            base_size=self.BASE_SIZE,
            image_size=self.IMAGE_SIZE,
            crop_mode=True,
            save_results=True,
        )
        if self.INCLUDE_TEST_COMPRESS:
            kwargs["test_compress"] = True

        result = self._model.infer(self._tokenizer, **kwargs)
        return result

    # ------------------------------------------------------------------
    # Markdown assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_markdown(page_markdowns: list[str]) -> str:
        """Join per-page markdown into a single document."""
        separator = "\n\n---\n\n"
        return separator.join(page_markdowns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Check whether vLLM or transformers are installed."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF through DeepSeek OCR.

        Pipeline:
            1. Load model (lazy).
            2. Convert PDF pages to images.
            3. Run inference on each page.
            4. Assemble markdown, write to output.
            5. Clean up temp images.
        """
        paper_name = Path(pdf_path).stem
        paper_output_dir = self.get_paper_output_dir(pdf_path)
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = paper_output_dir / "temp_pages"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._ensure_model_loaded()

            page_images = self._pdf_to_images(pdf_path, temp_dir)
            if not page_images:
                return self._make_result(pdf_path, status="failed: no pages extracted")

            page_markdowns: list[str] = []
            for img_path in page_images:
                self.logger.info("Processing page %s", img_path.name)
                md = self._infer_single_image(img_path, paper_output_dir)
                page_markdowns.append(md)

            full_md = self._assemble_markdown(page_markdowns)

            md_path = paper_output_dir / f"{paper_name}.md"
            md_path.write_text(full_md, encoding="utf-8")

            return self._make_result(
                pdf_path,
                status="success",
                result_path=str(paper_output_dir),
            )

        except Exception as e:
            self.logger.error("Failed to process %s: %s", paper_name, e)
            return self._make_result(pdf_path, status=f"failed: {e}")

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
