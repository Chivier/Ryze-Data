"""Round-robin OpenAI-compatible client pool for parallel vLLM requests."""

from __future__ import annotations

import itertools
import threading
from typing import Any

import requests


def _normalize_base_url(endpoint: str) -> str:
    base = endpoint.strip().rstrip("/")
    if not base:
        raise ValueError("Endpoint URL cannot be empty.")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _extract_text_content(content: Any) -> str:
    """Extract plain text from OpenAI chat message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


class VLLMClientPool:
    """Thread-safe endpoint scheduler for OpenAI-compatible chat requests."""

    def __init__(
        self,
        endpoints: list[str],
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        normalized = [_normalize_base_url(endpoint) for endpoint in endpoints]
        if not normalized:
            raise ValueError("At least one endpoint is required.")

        self._endpoints = normalized
        self._endpoint_cycle = itertools.cycle(self._endpoints)
        self._lock = threading.Lock()
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _next_endpoint(self) -> str:
        with self._lock:
            return next(self._endpoint_cycle)

    def _chat_url(self, endpoint: str) -> str:
        return f"{endpoint}/chat/completions"

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Send chat completion request to one endpoint with retries.

        Extra ``kwargs`` are merged into the request payload, allowing
        per-request overrides such as ``max_tokens`` or ``response_format``.
        """
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            endpoint = self._next_endpoint()
            url = self._chat_url(endpoint)
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            # Merge extra fields (e.g. response_format) excluding already-handled keys.
            for k, v in kwargs.items():
                if k not in ("max_tokens", "temperature"):
                    payload[k] = v
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]["message"]
                return _extract_text_content(choice.get("content"))
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        assert last_error is not None
        raise RuntimeError(f"Chat completion failed after retries: {last_error}")
