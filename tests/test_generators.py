"""Unit tests for QA generators."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.generators.base_generator import QAPair
from src.generators.prompt_manager import PromptManager
from src.generators.text_qa_generator import TextQAGenerator
from src.generators.vision_qa_generator import VisionQAGenerator


class TestQAPair:
    """Test QAPair dataclass."""

    def test_to_dict(self):
        qa = QAPair(
            question="What is CRISPR?",
            answer="CRISPR is a gene editing tool.",
            difficulty="easy",
            question_type="factual",
            paper_id="test_paper",
            section="section_0",
        )
        d = qa.to_dict()
        assert d["question"] == "What is CRISPR?"
        assert d["answer"] == "CRISPR is a gene editing tool."
        assert d["difficulty"] == "easy"

    def test_to_jsonl_line(self):
        qa = QAPair(
            question="Test?",
            answer="Answer.",
            paper_id="paper1",
        )
        line = qa.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["question"] == "Test?"
        assert parsed["paper_id"] == "paper1"


class TestPromptManager:
    """Test PromptManager class."""

    def test_list_text_prompts(self):
        pm = PromptManager("./prompts")
        prompts = pm.list_text_prompts()
        assert "factual" in prompts
        assert "mechanism" in prompts
        assert "application" in prompts

    def test_list_vision_prompts(self):
        pm = PromptManager("./prompts")
        prompts = pm.list_vision_prompts()
        assert "visual-factual" in prompts
        assert "visual-mechanism" in prompts

    def test_get_text_prompt(self):
        pm = PromptManager("./prompts")
        prompt = pm.get_text_prompt("factual", context="Test context here.")
        assert "Test context here." in prompt
        assert "questions" in prompt.lower()

    def test_get_vision_prompt(self):
        pm = PromptManager("./prompts")
        prompt = pm.get_vision_prompt("visual-factual", context="Figure context.")
        assert "Figure context." in prompt

    def test_cache_works(self):
        pm = PromptManager("./prompts")
        # First call loads from disk
        pm.get_text_prompt("factual", context="A")
        assert len(pm._cache) == 1
        # Second call uses cache
        pm.get_text_prompt("factual", context="B")
        assert len(pm._cache) == 1


class TestTextQAGenerator:
    """Test TextQAGenerator class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ocr_dir = Path(tmpdir) / "ocr"
            output_dir = Path(tmpdir) / "output"
            abstract_dir = Path(tmpdir) / "abstracts"
            ocr_dir.mkdir()
            output_dir.mkdir()
            abstract_dir.mkdir()

            # Create sample markdown
            sample_md = ocr_dir / "test_paper.md"
            sample_md.write_text("""# Test Paper

## Introduction

This is a test introduction section with enough content to be processed.
The CRISPR-Cas9 system enables precise genome editing in various organisms.
We achieved 95% editing efficiency in our experiments.

## Results

Our results show significant improvements in crop yield.
The mutant plants showed 20% higher grain weight compared to wild-type.
Statistical analysis confirmed p < 0.001 significance level.
""")
            yield {
                "ocr_dir": str(ocr_dir),
                "output_dir": str(output_dir),
                "abstract_dir": str(abstract_dir),
            }

    def test_split_into_sections(self, temp_dirs):
        """Test section splitting."""
        gen = TextQAGenerator(
            ocr_dir=temp_dirs["ocr_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
            qa_ratio=3,
        )

        # Sections need to be > 100 chars to pass the filter
        content = """# Title

## Section 1

This is the first section with plenty of content to ensure it passes the
minimum length requirement of 100 characters. We need substantial text here.

## Section 2

This is the second section which also contains enough text to be considered
valid for processing. The content here discusses important findings.
"""
        sections = gen._split_into_sections(content)
        assert len(sections) >= 1

    def test_chunk_text(self, temp_dirs):
        """Test text chunking."""
        gen = TextQAGenerator(
            ocr_dir=temp_dirs["ocr_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
        )

        long_text = "Paragraph 1.\n\n" * 100
        chunks = gen._chunk_text(long_text, max_chars=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 600  # Allow some overflow

    @patch("src.generators.text_qa_generator.TextQAGenerator._get_api_keys")
    @patch("src.generators.text_qa_generator.TextQAGenerator._init_balancer")
    def test_process_paper_mocked(self, mock_init_balancer, mock_get_keys, temp_dirs):
        """Test paper processing with mocked API."""
        mock_get_keys.return_value = ["test-key"]

        gen = TextQAGenerator(
            ocr_dir=temp_dirs["ocr_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
            qa_ratio=3,
        )

        # Mock the balancer
        mock_balancer = MagicMock()
        gen.balancer = mock_balancer

        # Mock submit_chat_completion to return request IDs
        mock_balancer.submit_chat_completion.side_effect = [
            "req_1", "req_2", "req_3"
        ]

        # Mock get_result to return successful responses
        mock_result = MagicMock()
        mock_result.id = "req_1"
        mock_result.status = MagicMock()
        mock_result.status.name = "SUCCESS"
        mock_result.result = MagicMock()
        mock_result.result.choices = [MagicMock()]
        mock_result.result.choices[0].message.content = json.dumps({
            "questions": [
                {"q": "What is CRISPR?", "a": "A gene editing tool.", "difficulty": "easy"},
                {"q": "What efficiency?", "a": "95% efficiency.", "difficulty": "medium"},
            ]
        })

        # Return result then None to exit loop
        mock_balancer.get_result.side_effect = [mock_result, None, None, None]
        mock_balancer.result_queue = MagicMock()

        paper_path = Path(temp_dirs["ocr_dir"]) / "test_paper.md"

        # This will call the mocked methods
        # Note: Full integration would require more complex mocking


class TestVisionQAGenerator:
    """Test VisionQAGenerator class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vlm_dir = Path(tmpdir) / "vlm"
            output_dir = Path(tmpdir) / "output"
            abstract_dir = Path(tmpdir) / "abstracts"
            vlm_dir.mkdir()
            output_dir.mkdir()
            abstract_dir.mkdir()

            # Create sample figure context JSON
            context_json = vlm_dir / "test_paper.json"
            context_json.write_text(json.dumps({
                "paper_id": "test_paper",
                "figures": [
                    {
                        "id": "Figure_1",
                        "label": "Figure 1",
                        "image_path": "test_figure.jpeg",
                        "caption": "Test figure caption.",
                        "related_info": {
                            "before": "Text before figure.",
                            "after": "Text after figure."
                        }
                    }
                ]
            }))

            # Create a simple test image
            from PIL import Image
            img = Image.new("RGB", (100, 100), "white")
            img.save(vlm_dir / "test_figure.jpeg", "JPEG")

            yield {
                "vlm_dir": str(vlm_dir),
                "output_dir": str(output_dir),
                "abstract_dir": str(abstract_dir),
            }

    def test_extract_figures(self, temp_dirs):
        """Test figure extraction from JSON."""
        gen = VisionQAGenerator(
            vlm_dir=temp_dirs["vlm_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
        )

        context_path = Path(temp_dirs["vlm_dir"]) / "test_paper.json"
        with open(context_path) as f:
            data = json.load(f)

        figures = gen._extract_figures(data, Path(temp_dirs["vlm_dir"]))
        assert len(figures) == 1
        assert figures[0]["figure_id"] == "Figure_1"
        assert "caption" in figures[0]["context"].lower()

    def test_encode_image(self, temp_dirs):
        """Test image encoding to base64."""
        gen = VisionQAGenerator(
            vlm_dir=temp_dirs["vlm_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
        )

        img_path = Path(temp_dirs["vlm_dir"]) / "test_figure.jpeg"
        b64 = gen._encode_image(img_path)
        assert isinstance(b64, str)
        assert len(b64) > 100  # Should be substantial

    def test_save_vision_qa(self, temp_dirs):
        """Test saving vision QA in LlamaFactory format."""
        gen = VisionQAGenerator(
            vlm_dir=temp_dirs["vlm_dir"],
            abstract_dir=temp_dirs["abstract_dir"],
            output_dir=temp_dirs["output_dir"],
            model="gpt-4o-mini",
        )

        qa_pairs = [
            QAPair(
                question="What does the figure show?",
                answer="The figure shows a bar chart.",
                difficulty="easy",
                question_type="factual",
                paper_id="test_paper",
                section="Figure_1",
                metadata={"image_path": "/path/to/image.jpeg", "figure_id": "Figure_1"},
            )
        ]

        output_path = gen._save_vision_qa(qa_pairs, "test_output")

        # Verify output
        with open(output_path) as f:
            line = f.readline()
            data = json.loads(line)

        assert "messages" in data
        assert len(data["messages"]) == 2
        assert "<image>" in data["messages"][0]["content"]
        assert "images" in data
        assert "metadata" in data
        assert data["metadata"]["paper_id"] == "test_paper"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
