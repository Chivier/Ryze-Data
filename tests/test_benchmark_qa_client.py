"""Unit tests for QwenQAClient with mocked OpenAI API."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark.qa_client import QwenQAClient


class TestQwenQAClient:
    @pytest.fixture
    def mock_openai(self):
        with patch("src.benchmark.qa_client.OpenAI") as mock:
            client = MagicMock()
            mock.return_value = client

            completion = MagicMock()
            completion.choices = [MagicMock(message=MagicMock(content="Test answer"))]
            client.chat.completions.create.return_value = completion

            yield client

    @pytest.fixture
    def qa_client(self, mock_openai):
        return QwenQAClient(
            model="Qwen3-VL-8B",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )

    @pytest.fixture
    def tmp_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            return f.name

    def test_answer_with_text_no_choices(self, qa_client, mock_openai):
        """Test text-only mode without MC choices."""
        result = qa_client.answer_with_text(
            ocr_markdown="# Test Paper\nContent here.",
            question="What is this about?",
        )
        assert result == "Test answer"
        mock_openai.chat.completions.create.assert_called_once()

        # Check message format
        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Test Paper" in messages[0]["content"]

    def test_answer_with_text_with_choices(self, qa_client, mock_openai):
        """Test text-only mode with MC choices."""
        result = qa_client.answer_with_text(
            ocr_markdown="# Paper",
            question="Which is correct?",
            choices=["Option A", "Option B"],
        )
        assert result == "Test answer"

        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "A. Option A" in messages[0]["content"]
        assert "B. Option B" in messages[0]["content"]

    def test_answer_with_vision(self, qa_client, mock_openai, tmp_image):
        """Test vision mode sends images as base64."""
        result = qa_client.answer_with_vision(
            image_paths=[tmp_image],
            question="What does this show?",
        )
        assert result == "Test answer"

        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]

        # Should have image_url and text parts
        assert isinstance(content, list)
        assert any(p["type"] == "image_url" for p in content)
        assert any(p["type"] == "text" for p in content)

    def test_answer_with_vision_choices(self, qa_client, mock_openai, tmp_image):
        """Test vision mode with MC choices."""
        qa_client.answer_with_vision(
            image_paths=[tmp_image],
            question="Which option?",
            choices=["Red", "Blue"],
        )

        call_args = mock_openai.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        text_parts = [p for p in messages[0]["content"] if p["type"] == "text"]
        assert any("A. Red" in p["text"] for p in text_parts)

    def test_api_error(self, qa_client, mock_openai):
        """Test handling of API errors."""
        mock_openai.chat.completions.create.side_effect = Exception("API down")

        with pytest.raises(RuntimeError, match="QA API call failed"):
            qa_client.answer_with_text(
                ocr_markdown="text",
                question="question",
            )

    def test_encode_image(self, tmp_image):
        """Test base64 encoding of image files."""
        b64 = QwenQAClient._encode_image(tmp_image)
        assert isinstance(b64, str)
        assert len(b64) > 0
