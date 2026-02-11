"""Unit tests for benchmark dataset loaders using fixture data."""

import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark.datasets.base import BenchmarkSample


class TestBenchmarkSample:
    def test_creation(self):
        sample = BenchmarkSample(
            sample_id="test_001",
            image_paths=["/tmp/img.png"],
            question="What is this?",
            choices=["A", "B", "C"],
            correct_answer="A",
            question_type="multiple_choice",
        )
        assert sample.sample_id == "test_001"
        assert sample.question_type == "multiple_choice"
        assert sample.metadata == {}

    def test_with_metadata(self):
        sample = BenchmarkSample(
            sample_id="test_002",
            image_paths=[],
            question="Q",
            choices=None,
            correct_answer="A",
            question_type="free_text",
            metadata={"source": "test"},
        )
        assert sample.metadata["source"] == "test"


def _make_datasets_module(mock_load_dataset):
    """Create a fake 'datasets' module with load_dataset."""
    fake = ModuleType("datasets")
    fake.load_dataset = mock_load_dataset
    return fake


class TestArxivQADataset:
    @pytest.fixture
    def data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_dataset_name(self, data_dir):
        from src.benchmark.datasets.arxivqa import ArxivQADataset

        ds = ArxivQADataset(data_dir=data_dir)
        assert ds.DATASET_NAME == "arxivqa"

    def test_is_downloaded_false(self, data_dir):
        from src.benchmark.datasets.arxivqa import ArxivQADataset

        ds = ArxivQADataset(data_dir=data_dir)
        assert ds.is_downloaded() is False

    def test_download(self, data_dir):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load = MagicMock(return_value=mock_dataset)

        fake_ds_module = _make_datasets_module(mock_load)

        with patch.dict(sys.modules, {"datasets": fake_ds_module}):
            from src.benchmark.datasets.arxivqa import ArxivQADataset

            ds = ArxivQADataset(data_dir=data_dir)
            ds.download()

            assert ds.is_downloaded() is True
            mock_load.assert_called_once_with("MMInstruction/ArxivQA", split="train")

    def test_load_samples(self, data_dir):
        """Test loading with mock HuggingFace dataset items."""
        mock_image = MagicMock()
        mock_image.save = MagicMock()

        mock_item = {
            "image": mock_image,
            "question": "What does this figure show?",
            "options": ["Graph", "Table", "Diagram", "Photo"],
            "label": "A",
        }

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_dataset.select = MagicMock(return_value=mock_dataset)
        mock_load = MagicMock(return_value=mock_dataset)

        fake_ds_module = _make_datasets_module(mock_load)

        with patch.dict(sys.modules, {"datasets": fake_ds_module}):
            from src.benchmark.datasets.arxivqa import ArxivQADataset

            ds = ArxivQADataset(data_dir=data_dir)
            samples = ds.load_samples(max_samples=1)

        assert len(samples) == 1
        assert samples[0].question_type == "multiple_choice"
        assert samples[0].correct_answer == "Graph"
        assert samples[0].choices == [
            "Graph",
            "Table",
            "Diagram",
            "Photo",
        ]

    def test_load_samples_label_mapping(self, data_dir):
        """Test that label B maps to second option."""
        mock_image = MagicMock()
        mock_image.save = MagicMock()

        mock_item = {
            "image": mock_image,
            "question": "Q?",
            "options": ["Opt1", "Opt2", "Opt3"],
            "label": "B",
        }

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_dataset.select = MagicMock(return_value=mock_dataset)
        mock_load = MagicMock(return_value=mock_dataset)

        fake_ds_module = _make_datasets_module(mock_load)

        with patch.dict(sys.modules, {"datasets": fake_ds_module}):
            from src.benchmark.datasets.arxivqa import ArxivQADataset

            ds = ArxivQADataset(data_dir=data_dir)
            samples = ds.load_samples(max_samples=1)

        assert samples[0].correct_answer == "Opt2"


class TestSlideVQADataset:
    @pytest.fixture
    def data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_dataset_name(self, data_dir):
        from src.benchmark.datasets.slidevqa import SlideVQADataset

        ds = SlideVQADataset(data_dir=data_dir)
        assert ds.DATASET_NAME == "slidevqa"

    def test_is_downloaded_false(self, data_dir):
        from src.benchmark.datasets.slidevqa import SlideVQADataset

        ds = SlideVQADataset(data_dir=data_dir)
        assert ds.is_downloaded() is False

    def test_download(self, data_dir):
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=50)
        mock_load = MagicMock(return_value=mock_dataset)

        fake_ds_module = _make_datasets_module(mock_load)

        with patch.dict(sys.modules, {"datasets": fake_ds_module}):
            from src.benchmark.datasets.slidevqa import SlideVQADataset

            ds = SlideVQADataset(data_dir=data_dir)
            ds.download()

            assert ds.is_downloaded() is True
            mock_load.assert_called_once_with("NTT-hil-insight/SlideVQA", split="test")

    def test_load_samples(self, data_dir):
        """Test loading with mock slide images."""
        mock_img1 = MagicMock()
        mock_img1.save = MagicMock()
        mock_img2 = MagicMock()
        mock_img2.save = MagicMock()

        mock_item = {
            "images": [mock_img1, mock_img2],
            "question": "What is shown on slide 1?",
            "answer": "A bar chart",
        }

        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_dataset.select = MagicMock(return_value=mock_dataset)
        mock_load = MagicMock(return_value=mock_dataset)

        fake_ds_module = _make_datasets_module(mock_load)

        with patch.dict(sys.modules, {"datasets": fake_ds_module}):
            from src.benchmark.datasets.slidevqa import SlideVQADataset

            ds = SlideVQADataset(data_dir=data_dir)
            samples = ds.load_samples(max_samples=1)

        assert len(samples) == 1
        assert samples[0].question_type == "free_text"
        assert samples[0].correct_answer == "A bar chart"
        assert len(samples[0].image_paths) == 2
