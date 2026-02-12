"""Prompt and multimodal message builders for OCR benchmark requests."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional


@lru_cache(maxsize=8192)
def _encode_image_b64(image_path: str) -> str:
    """Encode image bytes to base64 with process-local memoization."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices))


def build_baseline_text_prompt(question: str, choices: Optional[list[str]]) -> str:
    """Build user text prompt for vision-only baseline."""
    if choices:
        return (
            "Look at the image(s) and answer the following question.\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{_format_choices(choices)}\n\n"
            "Answer with the correct option text only."
        )
    return (
        "Look at the image(s) and answer the following question.\n\n"
        f"Question: {question}\n\n"
        "Provide only the final answer text."
    )


def build_ocr_text_prompt(
    ocr_markdown: str,
    question: str,
    choices: Optional[list[str]],
) -> str:
    """Build user text prompt for OCR-augmented experiments."""
    if choices:
        return (
            "Use both the image(s) and OCR markdown to answer the question.\n\n"
            f"OCR Markdown:\n{ocr_markdown}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{_format_choices(choices)}\n\n"
            "Answer with the correct option text only."
        )
    return (
        "Use both the image(s) and OCR markdown to answer the question.\n\n"
        f"OCR Markdown:\n{ocr_markdown}\n\n"
        f"Question: {question}\n\n"
        "Provide only the final answer text."
    )


def build_multimodal_messages(
    image_paths: list[str],
    text_prompt: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat messages with image_url + text content."""
    content: list[dict[str, Any]] = []
    for path in image_paths:
        b64 = _encode_image_b64(path)
        mime_type = _guess_mime_type(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        )

    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]
