"""Unit tests for OCR base classes, registry, device_utils, and status_tracker."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry
from src.ocr.status_tracker import OCRStatusTracker

# ============== OCRResult Tests ==============


class TestOCRResult:
    """Test OCRResult dataclass."""

    def test_to_dict(self):
        result = OCRResult(
            paper_name="test_paper",
            original_pdf_path="/path/to/test_paper.pdf",
            ocr_status="success",
            ocr_time="2025-01-01 12:00:00",
            ocr_result_path="/output/test_paper",
        )
        d = result.to_dict()
        assert d["paper_name"] == "test_paper"
        assert d["ocr_status"] == "success"
        assert d["original_pdf_path"] == "/path/to/test_paper.pdf"

    def test_to_dict_has_all_csv_fields(self):
        result = OCRResult(
            paper_name="p",
            original_pdf_path="/p.pdf",
            ocr_status="failed",
            ocr_time="2025-01-01",
            ocr_result_path="",
        )
        d = result.to_dict()
        expected_fields = {
            "paper_name",
            "original_pdf_path",
            "ocr_status",
            "ocr_time",
            "ocr_result_path",
        }
        assert set(d.keys()) == expected_fields

    def test_make_result_factory(self):
        result = OCRResult.make_result(
            pdf_path="/data/sample.pdf",
            status="success",
            result_path="/output/sample",
        )
        assert result.paper_name == "sample"
        assert result.original_pdf_path == "/data/sample.pdf"
        assert result.ocr_status == "success"
        assert result.ocr_result_path == "/output/sample"
        assert len(result.ocr_time) > 0

    def test_make_result_defaults_to_failed(self):
        result = OCRResult.make_result(pdf_path="/data/test.pdf")
        assert result.ocr_status == "failed"
        assert result.ocr_result_path == ""


# ============== BaseOCRModel Tests ==============


class _DummyOCR(BaseOCRModel):
    """Concrete implementation for testing the ABC."""

    MODEL_NAME = "dummy"

    @property
    def name(self) -> str:
        return "Dummy"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def process_single(self, pdf_path: str) -> OCRResult:
        return self._make_result(pdf_path, status="success", result_path="/out")


class TestBaseOCRModel:
    """Test BaseOCRModel abstract class through a concrete dummy."""

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "new_subdir"
            model = _DummyOCR(output_dir=str(out))
            assert out.exists()
            assert model.output_dir == out

    def test_process_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            result = model.process_single("/data/paper.pdf")
            assert result.paper_name == "paper"
            assert result.ocr_status == "success"

    def test_supports_batch_default_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            assert model.supports_batch() is False

    def test_process_batch_sequential_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            results = model.process_batch(["/a.pdf", "/b.pdf"])
            assert len(results) == 2
            assert results[0].paper_name == "a"
            assert results[1].paper_name == "b"

    def test_get_paper_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            out = model.get_paper_output_dir("/data/my_paper.pdf")
            assert out == Path(tmpdir) / "my_paper"

    def test_has_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            assert model.logger is not None
            assert model.logger.name == "_DummyOCR"

    def test_create_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _DummyOCR(output_dir=tmpdir)
            progress = model.create_progress()
            assert progress is not None


# ============== OCRRegistry Tests ==============


class TestOCRRegistry:
    """Test OCRRegistry decorator and lookup methods."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved = dict(OCRRegistry._models)
        OCRRegistry.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        OCRRegistry._models = self._saved

    def test_register_decorator(self):
        @OCRRegistry.register
        class _TestModel(BaseOCRModel):
            MODEL_NAME = "test-model"

            @property
            def name(self):
                return "Test"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        assert "test-model" in OCRRegistry.list_all()

    def test_register_duplicate_raises(self):
        @OCRRegistry.register
        class _ModelA(BaseOCRModel):
            MODEL_NAME = "dup"

            @property
            def name(self):
                return "A"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        with pytest.raises(ValueError, match="already registered"):

            @OCRRegistry.register
            class _ModelB(BaseOCRModel):
                MODEL_NAME = "dup"

                @property
                def name(self):
                    return "B"

                @classmethod
                def is_available(cls):
                    return True

                def process_single(self, pdf_path):
                    return self._make_result(pdf_path)

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty MODEL_NAME"):

            @OCRRegistry.register
            class _NoName(BaseOCRModel):
                MODEL_NAME = ""

                @property
                def name(self):
                    return ""

                @classmethod
                def is_available(cls):
                    return True

                def process_single(self, pdf_path):
                    return self._make_result(pdf_path)

    def test_get_model_class(self):
        @OCRRegistry.register
        class _GetTest(BaseOCRModel):
            MODEL_NAME = "get-test"

            @property
            def name(self):
                return "GetTest"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        assert OCRRegistry.get_model_class("get-test") is _GetTest
        assert OCRRegistry.get_model_class("nonexistent") is None

    def test_get_model_instantiates(self):
        @OCRRegistry.register
        class _InstTest(BaseOCRModel):
            MODEL_NAME = "inst-test"

            @property
            def name(self):
                return "InstTest"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = OCRRegistry.get_model("inst-test", output_dir=tmpdir)
            assert isinstance(model, _InstTest)

    def test_get_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown OCR model"):
            OCRRegistry.get_model("no-such-model", output_dir="/tmp")

    def test_get_model_unavailable_raises(self):
        @OCRRegistry.register
        class _Unavail(BaseOCRModel):
            MODEL_NAME = "unavail"

            @property
            def name(self):
                return "Unavail"

            @classmethod
            def is_available(cls):
                return False

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        with pytest.raises(RuntimeError, match="not installed"):
            OCRRegistry.get_model("unavail", output_dir="/tmp")

    def test_list_available_filters(self):
        @OCRRegistry.register
        class _Avail(BaseOCRModel):
            MODEL_NAME = "avail"

            @property
            def name(self):
                return "Avail"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        @OCRRegistry.register
        class _NotAvail(BaseOCRModel):
            MODEL_NAME = "not-avail"

            @property
            def name(self):
                return "NotAvail"

            @classmethod
            def is_available(cls):
                return False

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        assert "avail" in OCRRegistry.list_available()
        assert "not-avail" not in OCRRegistry.list_available()
        assert "not-avail" in OCRRegistry.list_all()

    def test_list_models_with_status(self):
        @OCRRegistry.register
        class _StatusTest(BaseOCRModel):
            MODEL_NAME = "status-test"

            @property
            def name(self):
                return "StatusTest"

            @classmethod
            def is_available(cls):
                return True

            def process_single(self, pdf_path):
                return self._make_result(pdf_path)

        entries = OCRRegistry.list_models_with_status()
        assert len(entries) == 1
        assert entries[0]["name"] == "status-test"
        assert entries[0]["status"] == "available"


# ============== OCRStatusTracker Tests ==============


class TestOCRStatusTracker:
    """Test OCRStatusTracker CSV writing and status reporting."""

    def test_record_result_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir)
            tracker.total_files = 3

            tracker.record_result(
                OCRResult.make_result("/a.pdf", status="success", result_path="/out/a")
            )
            assert tracker.completed_files == 1
            assert tracker.failed_files == 0

            tracker.record_result(
                OCRResult.make_result("/b.pdf", status="failed: error")
            )
            assert tracker.completed_files == 1
            assert tracker.failed_files == 1

    def test_flush_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir)
            tracker.total_files = 1

            tracker.record_result(
                OCRResult.make_result("/a.pdf", status="success", result_path="/out/a")
            )
            tracker.flush()

            csv_path = Path(tmpdir) / "ocr_status.csv"
            assert csv_path.exists()

            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]["paper_name"] == "a"
                assert rows[0]["ocr_status"] == "success"

    def test_flush_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir)
            tracker.total_files = 2

            tracker.record_result(
                OCRResult.make_result("/a.pdf", status="success", result_path="/out/a")
            )
            tracker.flush()

            tracker.record_result(
                OCRResult.make_result("/b.pdf", status="success", result_path="/out/b")
            )
            tracker.flush()

            csv_path = Path(tmpdir) / "ocr_status.csv"
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2

    def test_progress_percentage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir)
            tracker.total_files = 4
            tracker.completed_files = 2
            tracker.failed_files = 1
            assert tracker.progress_percentage == 75.0

    def test_progress_percentage_zero_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir)
            assert tracker.progress_percentage == 0.0

    def test_get_status_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = OCRStatusTracker(output_dir=tmpdir, task_name="test_task")
            tracker.total_files = 10
            tracker.completed_files = 5
            tracker.failed_files = 2

            status = tracker.get_status_dict()
            assert status["task_name"] == "test_task"
            assert status["total_files"] == 10
            assert status["completed_files"] == 5
            assert status["failed_files"] == 2
            assert status["progress_percentage"] == 70.0


# ============== device_utils Tests ==============


class TestDetectDevices:
    """Test detect_devices utility."""

    def test_with_gpus(self):
        from src.ocr.device_utils import detect_devices

        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 2
        props_0 = MagicMock()
        props_0.total_memory = 8 * 1024 * 1024 * 1024  # 8 GB
        props_1 = MagicMock()
        props_1.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_torch.cuda.get_device_properties.side_effect = [props_0, props_1]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            gpu_count, workers = detect_devices()
        assert gpu_count == 2
        assert workers[0] == 2  # 8 / 3.5 = 2
        assert workers[1] == 4  # 16 / 3.5 = 4

    def test_no_gpus(self):
        from src.ocr.device_utils import detect_devices

        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 0

        mock_psutil = MagicMock()
        mock_psutil.cpu_count.return_value = 8

        with patch.dict("sys.modules", {"torch": mock_torch, "psutil": mock_psutil}):
            gpu_count, workers = detect_devices()
        assert gpu_count == 0
        assert workers[0] == 4  # min(8, 4) = 4
