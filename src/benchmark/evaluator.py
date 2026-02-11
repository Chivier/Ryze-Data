"""Benchmark evaluator orchestrating OCR + QA pipeline evaluation.

Runs 5 paths (baseline + 4 OCR models) and collects metrics.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.benchmark.datasets.base import BaseBenchmarkDataset, BenchmarkSample
from src.benchmark.image_utils import images_to_pdf
from src.benchmark.metrics import compute_all_metrics
from src.benchmark.qa_client import QwenQAClient

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results for a single OCR model on a dataset.

    Attributes:
        model_name: Name of the OCR model (or "baseline").
        predictions: List of predicted answers.
        references: List of reference answers.
        ocr_times: OCR processing time per sample (seconds).
        metrics: Computed metric scores.
    """

    model_name: str
    predictions: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    ocr_times: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full benchmark report across all models and datasets.

    Attributes:
        dataset_name: Name of the benchmark dataset.
        num_samples: Number of samples evaluated.
        question_type: "multiple_choice" or "free_text".
        model_results: Per-model results.
    """

    dataset_name: str
    num_samples: int
    question_type: str
    model_results: List[ModelResult] = field(default_factory=list)


class BenchmarkEvaluator:
    """Main orchestrator for benchmark evaluation.

    Manages the flow: load samples -> convert images -> run OCR ->
    query QA model -> compute metrics -> generate report.

    Attributes:
        qa_client: QwenQAClient for VLM queries.
        results_dir: Directory for caching intermediate results.
    """

    def __init__(
        self,
        qa_client: QwenQAClient,
        results_dir: str = "data/benchmark_results",
    ):
        """Initialize the evaluator.

        Args:
            qa_client: Initialized QwenQAClient instance.
            results_dir: Directory for caching OCR and QA results.
        """
        self.qa_client = qa_client
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        dataset: BaseBenchmarkDataset,
        ocr_models: List[str],
        max_samples: int = 0,
        include_baseline: bool = True,
    ) -> BenchmarkReport:
        """Run the full benchmark evaluation.

        Args:
            dataset: Benchmark dataset to evaluate on.
            ocr_models: List of OCR model names to evaluate.
            max_samples: Max samples to evaluate (0 = all).
            include_baseline: Whether to include the vision baseline.

        Returns:
            BenchmarkReport with all results.
        """
        # 1. Load samples
        logger.info(f"Loading {dataset.DATASET_NAME} samples...")
        samples = dataset.load_samples(max_samples)
        if not samples:
            raise ValueError("No samples loaded from dataset")

        logger.info(f"Loaded {len(samples)} samples")
        question_type = samples[0].question_type

        # 2. Convert images to PDFs (cached)
        logger.info("Converting images to PDFs...")
        pdf_cache = self._prepare_pdfs(samples)

        report = BenchmarkReport(
            dataset_name=dataset.DATASET_NAME,
            num_samples=len(samples),
            question_type=question_type,
        )

        # 3. Run baseline (vision mode)
        if include_baseline:
            logger.info("Running baseline (vision mode)...")
            baseline_result = self._run_baseline(samples)
            report.model_results.append(baseline_result)

        # 4. Run each OCR model
        for model_name in ocr_models:
            logger.info(f"Running OCR model: {model_name}")
            model_result = self._run_ocr_pipeline(model_name, samples, pdf_cache)
            report.model_results.append(model_result)

        return report

    def _prepare_pdfs(self, samples: List[BenchmarkSample]) -> Dict[str, str]:
        """Convert sample images to PDFs, with caching.

        Args:
            samples: List of benchmark samples.

        Returns:
            Mapping of sample_id to PDF path.
        """
        pdf_dir = self.results_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_cache = {}
        for sample in samples:
            pdf_path = str(pdf_dir / f"{sample.sample_id}.pdf")
            if not Path(pdf_path).exists():
                images_to_pdf(sample.image_paths, pdf_path)
            pdf_cache[sample.sample_id] = pdf_path

        return pdf_cache

    def _run_baseline(self, samples: List[BenchmarkSample]) -> ModelResult:
        """Run baseline evaluation using vision mode.

        Args:
            samples: Benchmark samples.

        Returns:
            ModelResult for the baseline.
        """
        result = ModelResult(model_name="baseline")
        cache_path = self.results_dir / "baseline" / "qa_results.jsonl"
        cached = self._load_qa_cache(cache_path)

        for sample in samples:
            if sample.sample_id in cached:
                prediction = cached[sample.sample_id]
            else:
                try:
                    prediction = self.qa_client.answer_with_vision(
                        image_paths=sample.image_paths,
                        question=sample.question,
                        choices=sample.choices,
                    )
                except Exception as e:
                    logger.warning(f"Baseline failed for {sample.sample_id}: {e}")
                    prediction = ""

                self._save_qa_result(cache_path, sample.sample_id, prediction)

            result.predictions.append(prediction)
            result.references.append(sample.correct_answer)

        result.metrics = compute_all_metrics(
            result.predictions,
            result.references,
            question_type=samples[0].question_type,
        )
        return result

    def _run_ocr_pipeline(
        self,
        model_name: str,
        samples: List[BenchmarkSample],
        pdf_cache: Dict[str, str],
    ) -> ModelResult:
        """Run a single OCR model pipeline.

        Args:
            model_name: OCR model name (e.g., "marker").
            samples: Benchmark samples.
            pdf_cache: Mapping of sample_id to PDF path.

        Returns:
            ModelResult for this OCR model.
        """
        from src.ocr.registry import OCRRegistry

        result = ModelResult(model_name=model_name)

        # Set up OCR output and QA cache
        ocr_output_dir = self.results_dir / model_name / "ocr_output"
        ocr_output_dir.mkdir(parents=True, exist_ok=True)

        qa_cache_path = self.results_dir / model_name / "qa_results.jsonl"
        qa_cached = self._load_qa_cache(qa_cache_path)

        # Instantiate OCR model
        ocr_model = OCRRegistry.get_model(model_name, output_dir=str(ocr_output_dir))

        for sample in samples:
            pdf_path = pdf_cache[sample.sample_id]

            # Run OCR (check cache first)
            md_path = ocr_output_dir / sample.sample_id / f"{sample.sample_id}.md"

            if md_path.exists():
                ocr_time = 0.0
            else:
                start = time.time()
                ocr_result = ocr_model.process_single(pdf_path)
                ocr_time = time.time() - start

                if "success" not in ocr_result.ocr_status:
                    logger.warning(
                        f"OCR failed for {sample.sample_id}: "
                        f"{ocr_result.ocr_status}"
                    )
                    result.predictions.append("")
                    result.references.append(sample.correct_answer)
                    result.ocr_times.append(ocr_time)
                    continue

            result.ocr_times.append(ocr_time)

            # Read OCR markdown
            try:
                ocr_markdown = md_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.warning(f"OCR output missing for {sample.sample_id}")
                result.predictions.append("")
                result.references.append(sample.correct_answer)
                continue

            # Run QA
            if sample.sample_id in qa_cached:
                prediction = qa_cached[sample.sample_id]
            else:
                try:
                    prediction = self.qa_client.answer_with_text(
                        ocr_markdown=ocr_markdown,
                        question=sample.question,
                        choices=sample.choices,
                    )
                except Exception as e:
                    logger.warning(f"QA failed for {sample.sample_id}: {e}")
                    prediction = ""

                self._save_qa_result(qa_cache_path, sample.sample_id, prediction)

            result.predictions.append(prediction)
            result.references.append(sample.correct_answer)

        result.metrics = compute_all_metrics(
            result.predictions,
            result.references,
            question_type=samples[0].question_type,
            ocr_times=result.ocr_times if result.ocr_times else None,
        )
        return result

    def _load_qa_cache(self, cache_path: Path) -> Dict[str, str]:
        """Load cached QA results from a JSONL file.

        Args:
            cache_path: Path to the JSONL cache file.

        Returns:
            Mapping of sample_id to predicted answer.
        """
        cached = {}
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        cached[entry["sample_id"]] = entry["prediction"]
        return cached

    def _save_qa_result(
        self, cache_path: Path, sample_id: str, prediction: str
    ) -> None:
        """Append a QA result to the JSONL cache.

        Args:
            cache_path: Path to the JSONL cache file.
            sample_id: The sample identifier.
            prediction: The predicted answer.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            entry = {"sample_id": sample_id, "prediction": prediction}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
