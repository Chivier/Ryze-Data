"""QA client for Qwen3-VL-8B supporting vision and text-only modes.

Uses OpenAI-compatible API format (vLLM / DashScope).
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class QwenQAClient:
    """Qwen3-VL-8B client supporting both vision and text-only modes.

    Communicates via an OpenAI-compatible API endpoint.

    Attributes:
        model: Model name for API requests.
        client: OpenAI client instance.
    """

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ):
        """Initialize the QA client.

        Args:
            model: Model name (e.g., "Qwen3-VL-8B").
            api_base: Base URL for the OpenAI-compatible API.
            api_key: API key (use "EMPTY" for local vLLM).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(base_url=api_base, api_key=api_key)

    def answer_with_vision(
        self,
        image_paths: List[str],
        question: str,
        choices: Optional[List[str]] = None,
    ) -> str:
        """Path 0: Send images directly to Qwen3-VL-8B (vision mode).

        Args:
            image_paths: Paths to image files.
            question: The question text.
            choices: Optional MC choices to include in the prompt.

        Returns:
            The model's answer text.
        """
        content: List[Dict[str, Any]] = []

        # Add images as base64
        for img_path in image_paths:
            b64 = self._encode_image(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        # Build question text
        prompt = self._build_prompt(question, choices)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        return self._call_api(messages)

    def answer_with_text(
        self,
        ocr_markdown: str,
        question: str,
        choices: Optional[List[str]] = None,
    ) -> str:
        """Paths 1-4: Send OCR text to Qwen3-VL-8B (text-only mode).

        Args:
            ocr_markdown: OCR-extracted markdown text.
            question: The question text.
            choices: Optional MC choices to include in the prompt.

        Returns:
            The model's answer text.
        """
        prompt = self._build_text_prompt(ocr_markdown, question, choices)
        messages = [{"role": "user", "content": prompt}]

        return self._call_api(messages)

    def _build_prompt(self, question: str, choices: Optional[List[str]]) -> str:
        """Build the question prompt for vision mode.

        Args:
            question: The question text.
            choices: Optional MC options.

        Returns:
            Formatted prompt string.
        """
        if choices:
            options_text = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
            )
            return (
                f"Look at the image(s) and answer the following question.\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Answer with the correct option text only."
            )
        return (
            f"Look at the image(s) and answer the following question.\n\n"
            f"Question: {question}\n\n"
            f"Provide a concise answer."
        )

    def _build_text_prompt(
        self,
        ocr_markdown: str,
        question: str,
        choices: Optional[List[str]],
    ) -> str:
        """Build the question prompt for text-only mode.

        Args:
            ocr_markdown: The OCR-extracted document text.
            question: The question text.
            choices: Optional MC options.

        Returns:
            Formatted prompt string.
        """
        if choices:
            options_text = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
            )
            return (
                f"Based on the following document content, answer the "
                f"question.\n\n"
                f"Document:\n{ocr_markdown}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Answer with the correct option text only."
            )
        return (
            f"Based on the following document content, answer the "
            f"question.\n\n"
            f"Document:\n{ocr_markdown}\n\n"
            f"Question: {question}\n\n"
            f"Provide a concise answer."
        )

    def _call_api(self, messages: List[Dict[str, Any]]) -> str:
        """Make an API call and return the response text.

        Args:
            messages: Chat messages to send.

        Returns:
            The model's response text.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise RuntimeError(f"QA API call failed: {e}") from e

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encode an image file to base64.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64-encoded string.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
