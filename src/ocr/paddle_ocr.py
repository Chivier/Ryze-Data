"""PaddleOCR implementation (PP-OCRv5 / PP-StructureV3)."""

import glob as globmod
import os
import shutil
import tempfile
from pathlib import Path

from src.ocr.base_ocr import BaseOCRModel, OCRResult, resize_image
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class PaddleOCRModel(BaseOCRModel):
    """OCR model using PaddleOCR.

    Supports two modes:
        - ``"ocr"``       — plain text extraction via PP-OCRv5.
        - ``"structure"`` — layout-aware Markdown via PP-StructureV3.
    """

    MODEL_NAME = "paddleocr"
    MAX_IMAGE_SIZE = (1024, 1024)
    DPI = 200

    def __init__(
        self,
        output_dir: str,
        mode: str = "ocr",
        lang: str = "en",
        device: str = "gpu:0",
    ):
        super().__init__(output_dir)
        self._engine = None
        self._mode = mode  # "ocr" | "structure"
        self._lang = lang
        self._device = device

    @property
    def name(self) -> str:
        return "PaddleOCR"

    # ------------------------------------------------------------------
    # Lazy engine loading
    # ------------------------------------------------------------------

    def _ensure_engine_loaded(self) -> None:
        if self._engine is not None:
            return

        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self._mode == "structure":
            from paddleocr import PPStructureV3

            self.logger.info(
                "Initializing PP-StructureV3 (device=%s) ...", self._device
            )
            self._engine = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=self._device,
            )
        else:
            from paddleocr import PaddleOCR

            self.logger.info(
                "Initializing PaddleOCR (device=%s, lang=%s) ...",
                self._device,
                self._lang,
            )
            self._engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self._device,
                lang=self._lang,
            )
        self.logger.info("PaddleOCR engine ready")

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(res) -> str:
        """Extract plain text from a PaddleOCR result object."""
        texts = (
            res.get("rec_texts")
            if isinstance(res, dict)
            else getattr(res, "rec_texts", None)
        )
        return "\n".join(texts) if texts else ""

    @staticmethod
    def _extract_markdown(res) -> str:
        """Extract markdown from a PP-StructureV3 result object."""
        with tempfile.TemporaryDirectory(prefix="ppstruct_") as tmp_dir:
            res.save_to_markdown(save_path=tmp_dir)
            md_files = sorted(
                globmod.glob(os.path.join(tmp_dir, "**", "*.md"), recursive=True)
            )
            if not md_files:
                return ""
            parts = [Path(f).read_text(encoding="utf-8") for f in md_files]
            return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # PDF → images helper
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: str, temp_dir: Path) -> list[str]:
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, dpi=self.DPI)
        paths: list[str] = []
        for idx, img in enumerate(images):
            img_path = temp_dir / f"page_{idx:04d}.png"
            img.save(str(img_path), "PNG")
            resized = resize_image(str(img_path), self.MAX_IMAGE_SIZE)
            paths.append(resized)
        return paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Check whether paddleocr is installed."""
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF through PaddleOCR."""
        paper_name = Path(pdf_path).stem
        paper_output_dir = self.get_paper_output_dir(pdf_path)
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = paper_output_dir / "temp_pages"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._ensure_engine_loaded()

            image_paths = self._pdf_to_images(pdf_path, temp_dir)
            if not image_paths:
                return self._make_result(
                    pdf_path, status="failed: no pages extracted"
                )

            # PaddleOCR predict accepts a list of image paths.
            results = list(self._engine.predict(input=image_paths))

            if self._mode == "structure":
                page_markdowns = [self._extract_markdown(r) for r in results]
            else:
                page_markdowns = [self._extract_text(r) for r in results]

            full_md = "\n\n---\n\n".join(page_markdowns)

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
