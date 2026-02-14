"""GLM-OCR implementation with vLLM local and Z.AI online API backends."""

import shutil
import tempfile
from pathlib import Path

from src.ocr.base_ocr import BaseOCRModel, OCRResult, resize_image
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class GLMOCRModel(BaseOCRModel):
    """OCR model using ``zai-org/GLM-OCR`` (0.9B parameters).

    Supports two backends:
        - ``"vllm"``: local inference via vLLM.
        - ``"api"``:  Z.AI OpenAI-compatible chat completions endpoint.
    """

    MODEL_NAME = "glm-ocr"
    HF_MODEL_ID = "zai-org/GLM-OCR"
    DEFAULT_PROMPT = "Text Recognition:"
    MAX_TOKENS = 8192
    MAX_IMAGE_SIZE = (1024, 1024)
    DPI = 200

    def __init__(
        self,
        output_dir: str,
        backend: str = "vllm",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        super().__init__(output_dir)
        self._backend = backend  # "vllm" | "api"
        self._api_key = api_key
        self._api_base = api_base or "https://open.bigmodel.cn/api/paas/v4"
        self._model = None
        self._sampling_params = None

    @property
    def name(self) -> str:
        return "GLM-OCR"

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

    def _load_vllm(self) -> None:
        if self._model is not None:
            return

        from vllm import LLM, SamplingParams

        self.logger.info("Loading %s with vLLM ...", self.HF_MODEL_ID)
        self._model = LLM(model=self.HF_MODEL_ID, trust_remote_code=True)
        self._sampling_params = SamplingParams(
            temperature=0.0, max_tokens=self.MAX_TOKENS
        )
        self.logger.info("vLLM model loaded")

    def _infer_vllm(self, image_path: str) -> str:
        from PIL import Image

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        outputs = self._model.generate(
            [
                {
                    "prompt": self.DEFAULT_PROMPT,
                    "multi_modal_data": {"image": image},
                }
            ],
            self._sampling_params,
        )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty output")
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Z.AI API backend
    # ------------------------------------------------------------------

    def _infer_api(self, image_path: str) -> str:
        """Call the Z.AI GLM-OCR API (OpenAI-compatible chat completions)."""
        import base64

        import requests

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime = "image/png" if image_path.endswith(".png") else "image/jpeg"

        resp = requests.post(
            f"{self._api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": "glm-ocr",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}"
                                },
                            },
                            {"type": "text", "text": self.DEFAULT_PROMPT},
                        ],
                    }
                ],
                "max_tokens": self.MAX_TOKENS,
                "temperature": 0.0,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Unified inference
    # ------------------------------------------------------------------

    def _infer(self, image_path: str) -> str:
        resized = resize_image(image_path, self.MAX_IMAGE_SIZE)
        if self._backend == "api":
            return self._infer_api(resized)
        return self._infer_vllm(resized)

    # ------------------------------------------------------------------
    # PDF → images helper
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: str, temp_dir: Path) -> list[Path]:
        from pdf2image import convert_from_path

        images = convert_from_path(pdf_path, dpi=self.DPI)
        paths: list[Path] = []
        for idx, img in enumerate(images):
            img_path = temp_dir / f"page_{idx:04d}.png"
            img.save(str(img_path), "PNG")
            paths.append(img_path)
        return paths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Available when vLLM is installed or an API key is provided."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            pass
        # When used via API the class itself is always importable;
        # actual key check happens at runtime.
        return True

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF through GLM-OCR."""
        paper_name = Path(pdf_path).stem
        paper_output_dir = self.get_paper_output_dir(pdf_path)
        paper_output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = paper_output_dir / "temp_pages"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self._backend == "vllm":
                self._load_vllm()

            page_images = self._pdf_to_images(pdf_path, temp_dir)
            if not page_images:
                return self._make_result(
                    pdf_path, status="failed: no pages extracted"
                )

            page_markdowns: list[str] = []
            for img_path in page_images:
                self.logger.info("Processing page %s", img_path.name)
                md = self._infer(str(img_path))
                page_markdowns.append(md)

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
