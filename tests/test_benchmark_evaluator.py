"""Integration tests for BenchmarkEvaluator with all external calls mocked."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark.datasets.base import BaseBenchmarkDataset, BenchmarkSample
from src.benchmark.evaluator import BenchmarkEvaluator, BenchmarkReport
from src.benchmark.qa_client import QwenQAClient


class MockDataset(BaseBenchmarkDataset):
    """Mock dataset for testing the evaluator."""

    DATASET_NAME = "mock_dataset"

    def __init__(self, data_dir, samples=None):
        super().__init__(data_dir)
        self._samples = samples or []

    def download(self):
        pass

    def is_downloaded(self):
        return True

    def load_samples(self, max_samples=0):
        if max_samples > 0:
            return self._samples[:max_samples]
        return self._samples


class TestBenchmarkEvaluator:
    @pytest.fixture
    def results_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_images(self, results_dir):
        """Create sample image files for testing."""
        from PIL import Image

        img_dir = Path(results_dir) / "test_images"
        img_dir.mkdir()

        paths = []
        for i in range(2):
            img_path = img_dir / f"sample_{i}.png"
            img = Image.new("RGB", (100, 100), color=(i * 50, 100, 200))
            img.save(str(img_path))
            paths.append(str(img_path))
        return paths

    @pytest.fixture
    def mc_samples(self, sample_images):
        """Multiple-choice benchmark samples."""
        return [
            BenchmarkSample(
                sample_id="mc_0",
                image_paths=[sample_images[0]],
                question="What does this figure show?",
                choices=["Graph", "Table", "Diagram"],
                correct_answer="Graph",
                question_type="multiple_choice",
            ),
            BenchmarkSample(
                sample_id="mc_1",
                image_paths=[sample_images[1]],
                question="What type of chart is this?",
                choices=["Bar", "Line", "Pie"],
                correct_answer="Bar",
                question_type="multiple_choice",
            ),
        ]

    @pytest.fixture
    def ft_samples(self, sample_images):
        """Free-text benchmark samples."""
        return [
            BenchmarkSample(
                sample_id="ft_0",
                image_paths=[sample_images[0]],
                question="Describe the figure.",
                choices=None,
                correct_answer="A bar chart showing sales data",
                question_type="free_text",
            ),
        ]

    @pytest.fixture
    def mock_qa_client(self):
        with patch("src.benchmark.qa_client.OpenAI"):
            client = QwenQAClient(
                model="test-model",
                api_base="http://localhost:8000/v1",
            )
            client.answer_with_vision = MagicMock(return_value="Graph")
            client.answer_with_text = MagicMock(return_value="Graph")
            return client

    def test_run_baseline_only(self, results_dir, mc_samples, mock_qa_client):
        """Test running only baseline (vision mode)."""
        dataset = MockDataset(results_dir, samples=mc_samples)
        evaluator = BenchmarkEvaluator(
            qa_client=mock_qa_client,
            results_dir=results_dir,
        )

        report = evaluator.run(
            dataset=dataset,
            ocr_models=[],
            include_baseline=True,
        )

        assert isinstance(report, BenchmarkReport)
        assert report.dataset_name == "mock_dataset"
        assert report.num_samples == 2
        assert len(report.model_results) == 1
        assert report.model_results[0].model_name == "baseline"
        assert "accuracy" in report.model_results[0].metrics

    def test_run_with_ocr_model(self, results_dir, mc_samples, mock_qa_client):
        """Test running with an OCR model (mocked)."""
        dataset = MockDataset(results_dir, samples=mc_samples)
        evaluator = BenchmarkEvaluator(
            qa_client=mock_qa_client,
            results_dir=results_dir,
        )

        # Mock OCR registry and model
        mock_ocr_model = MagicMock()
        mock_ocr_result = MagicMock()
        mock_ocr_result.ocr_status = "success"
        mock_ocr_model.process_single.return_value = mock_ocr_result

        with patch(
            "src.ocr.registry.OCRRegistry.get_model",
            return_value=mock_ocr_model,
        ):
            # Create fake OCR output
            for sample in mc_samples:
                ocr_dir = Path(results_dir) / "marker" / "ocr_output" / sample.sample_id
                ocr_dir.mkdir(parents=True)
                (ocr_dir / f"{sample.sample_id}.md").write_text("# OCR Output")

            report = evaluator.run(
                dataset=dataset,
                ocr_models=["marker"],
                include_baseline=False,
            )

        assert len(report.model_results) == 1
        assert report.model_results[0].model_name == "marker"

    def test_pdf_caching(self, results_dir, mc_samples, mock_qa_client):
        """Test that PDFs are only created once (cached)."""
        evaluator = BenchmarkEvaluator(
            qa_client=mock_qa_client,
            results_dir=results_dir,
        )

        pdf_cache = evaluator._prepare_pdfs(mc_samples)

        assert len(pdf_cache) == 2
        for sample_id, pdf_path in pdf_cache.items():
            assert Path(pdf_path).exists()

        # Running again should use cache (files already exist)
        pdf_cache_2 = evaluator._prepare_pdfs(mc_samples)
        assert pdf_cache == pdf_cache_2

    def test_qa_cache(self, results_dir):
        """Test QA result caching (JSONL)."""
        with patch("src.benchmark.qa_client.OpenAI"):
            client = QwenQAClient(model="test", api_base="http://localhost:8000/v1")
        evaluator = BenchmarkEvaluator(qa_client=client, results_dir=results_dir)

        cache_path = Path(results_dir) / "test_cache.jsonl"
        evaluator._save_qa_result(cache_path, "s1", "answer1")
        evaluator._save_qa_result(cache_path, "s2", "answer2")

        cached = evaluator._load_qa_cache(cache_path)
        assert cached == {"s1": "answer1", "s2": "answer2"}

    def test_empty_dataset_raises(self, results_dir, mock_qa_client):
        """Test that empty dataset raises ValueError."""
        dataset = MockDataset(results_dir, samples=[])
        evaluator = BenchmarkEvaluator(
            qa_client=mock_qa_client,
            results_dir=results_dir,
        )

        with pytest.raises(ValueError, match="No samples loaded"):
            evaluator.run(dataset=dataset, ocr_models=[])

    def test_free_text_metrics(self, results_dir, ft_samples, mock_qa_client):
        """Test that free-text samples produce EM/BLEU/ROUGE metrics."""
        mock_qa_client.answer_with_vision.return_value = (
            "A bar chart showing sales data"
        )
        dataset = MockDataset(results_dir, samples=ft_samples)
        evaluator = BenchmarkEvaluator(
            qa_client=mock_qa_client,
            results_dir=results_dir,
        )

        report = evaluator.run(
            dataset=dataset,
            ocr_models=[],
            include_baseline=True,
        )

        metrics = report.model_results[0].metrics
        assert "exact_match" in metrics
        assert "bleu_4" in metrics
        assert "rouge_l" in metrics
