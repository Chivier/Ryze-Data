"""DeepSeek-OCR v1 implementation using vLLM inference."""

import os
from pathlib import Path

from src.ocr.deepseek_base import BaseDeepSeekOCR
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class DeepSeekOCRv1(BaseDeepSeekOCR):
    """OCR model using DeepSeek-OCR (v1) via vLLM.

    Uses ``deepseek-ai/DeepSeek-OCR`` with the NGram anti-repetition
    logits processor from the official vLLM recipe.
    """

    MODEL_NAME = "deepseek-ocr"
    HF_MODEL_ID = "deepseek-ai/DeepSeek-OCR"
    IMAGE_SIZE = 640
    INCLUDE_TEST_COMPRESS = True
    LEGACY_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
    MAX_TOKENS = 8192

    def __init__(self, output_dir: str, gpu_id: int | None = None):
        super().__init__(output_dir)
        self._gpu_id = gpu_id
        self._sampling_params = None

    @property
    def name(self) -> str:
        return "DeepSeek-OCR v1"

    # ------------------------------------------------------------------
    # vLLM model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load DeepSeek-OCR v1 via vLLM with NGram logits processor."""
        if self._model is not None:
            return

        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import (
            NGramPerReqLogitsProcessor,
        )

        if self._gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_id)

        self.logger.info("Loading %s with vLLM ...", self.HF_MODEL_ID)

        llm_kwargs = {
            "model": self.HF_MODEL_ID,
            "enable_prefix_caching": False,
            "mm_processor_cache_gb": 0,
            "trust_remote_code": True,
            "logits_processors": [NGramPerReqLogitsProcessor],
        }

        try:
            self._model = LLM(**llm_kwargs)
        except TypeError as e:
            # Compatibility with older vLLM builds missing some kwargs.
            fallback_keys = ("mm_processor_cache_gb",)
            dropped = [k for k in fallback_keys if k in llm_kwargs and k in str(e)]
            if not dropped:
                raise
            for k in dropped:
                llm_kwargs.pop(k, None)
            self.logger.warning(
                "Retrying vLLM load without unsupported kwargs: %s",
                ", ".join(dropped),
            )
            self._model = LLM(**llm_kwargs)

        self._sampling_params = self._build_sampling_params()
        self.logger.info("vLLM model loaded")

    def _build_sampling_params(self):
        """Build vLLM SamplingParams with DeepSeek OCR n-gram controls."""
        from vllm import SamplingParams

        sampling_kwargs = {
            "temperature": 0.0,
            "max_tokens": self.MAX_TOKENS,
            "skip_special_tokens": False,
        }
        ngram_args = {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        }
        try:
            return SamplingParams(extra_args=ngram_args, **sampling_kwargs)
        except TypeError:
            self.logger.warning(
                "This vLLM build does not support SamplingParams.extra_args; "
                "using basic sampling params."
            )
            return SamplingParams(**sampling_kwargs)

    # ------------------------------------------------------------------
    # vLLM inference
    # ------------------------------------------------------------------

    def _infer_single_image(self, image_path: Path, output_path: Path) -> str:
        """Run OCR inference on a single image via vLLM."""
        from PIL import Image

        with Image.open(str(image_path)) as img:
            image = img.convert("RGB")

        model_inputs = [
            {
                "prompt": self.LEGACY_PROMPT,
                "multi_modal_data": {"image": image},
            }
        ]
        outputs = self._model.generate(model_inputs, self._sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty output")
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Check whether vLLM is installed."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False
