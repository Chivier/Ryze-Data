"""Marker OCR implementation using the marker CLI tools."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class MarkerOCR(BaseOCRModel):
    """OCR model using the Marker library (marker_single / marker_chunk_convert).

    This wraps the existing marker CLI tools for PDF-to-Markdown conversion.
    Supports both single-file and batch processing modes.
    """

    MODEL_NAME = "marker"

    @property
    def name(self) -> str:
        return "Marker"

    @classmethod
    def is_available(cls) -> bool:
        """Check if the marker CLI is installed."""
        return shutil.which("marker_single") is not None

    def process_single(self, pdf_path: str) -> OCRResult:
        """Process a single PDF using marker_single.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            OCRResult with processing outcome.
        """
        paper_name = Path(pdf_path).stem
        paper_output_dir = self.output_dir / paper_name

        try:
            cmd = [
                "marker_single",
                str(pdf_path),
                str(paper_output_dir),
                "--output_format",
                "markdown",
            ]

            self.logger.info(f"Processing {paper_name} with marker_single")
            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode == 0 and paper_output_dir.exists():
                md_path = paper_output_dir / f"{paper_name}.md"
                if md_path.exists():
                    return self._make_result(
                        pdf_path,
                        status="success",
                        result_path=str(paper_output_dir),
                    )

            stderr = process.stderr.strip() if process.stderr else "output not found"
            return self._make_result(pdf_path, status=f"failed: {stderr}")

        except Exception as e:
            self.logger.error(f"Failed to process {paper_name}: {e}")
            return self._make_result(pdf_path, status=f"failed: {e}")

    def supports_batch(self) -> bool:
        """Marker supports batch processing via marker_chunk_convert."""
        return shutil.which("marker_chunk_convert") is not None

    def process_batch(
        self,
        pdf_paths: List[str],
        gpu_count: int = 1,
        workers_per_gpu: int = 1,
    ) -> List[OCRResult]:
        """Process PDFs using marker_chunk_convert for multi-GPU batches.

        Falls back to sequential processing for single GPU.

        Args:
            pdf_paths: List of absolute paths to PDF files.
            gpu_count: Number of GPUs available.
            workers_per_gpu: Workers to use per GPU.

        Returns:
            List of OCRResult for each PDF.
        """
        if gpu_count <= 1 or not self.supports_batch():
            return super().process_batch(pdf_paths, gpu_count, workers_per_gpu)

        return self._run_batch(pdf_paths, gpu_count, workers_per_gpu)

    def _run_batch(
        self,
        pdf_paths: List[str],
        gpu_count: int,
        workers_per_gpu: int,
    ) -> List[OCRResult]:
        """Run marker_chunk_convert on a batch of PDFs.

        Args:
            pdf_paths: List of PDF file paths.
            gpu_count: Number of GPUs.
            workers_per_gpu: Workers per GPU.

        Returns:
            List of OCRResult for each PDF.
        """
        temp_input_dir = self.output_dir.parent / "temp_pdf_batch"
        temp_input_dir.mkdir(exist_ok=True)

        try:
            for pdf_path in pdf_paths:
                shutil.copy2(pdf_path, temp_input_dir)

            env = os.environ.copy()
            env["NUM_DEVICES"] = str(max(1, gpu_count))
            env["NUM_WORKERS"] = str(workers_per_gpu)

            cmd = [
                "marker_chunk_convert",
                str(temp_input_dir),
                str(self.output_dir),
            ]

            self.logger.info(
                f"Running marker_chunk_convert with {gpu_count} GPUs, "
                f"{workers_per_gpu} workers/GPU"
            )
            process = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if process.returncode != 0:
                self.logger.error(f"marker_chunk_convert failed: {process.stderr}")

            return self._collect_batch_results(pdf_paths)

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return [self._make_result(p, status=f"failed: {e}") for p in pdf_paths]

        finally:
            if temp_input_dir.exists():
                shutil.rmtree(temp_input_dir)

    def _collect_batch_results(self, pdf_paths: List[str]) -> List[OCRResult]:
        """Check output directories after batch processing.

        Args:
            pdf_paths: The original list of PDF paths.

        Returns:
            List of OCRResult based on output file presence.
        """
        results = []
        for pdf_path in pdf_paths:
            paper_name = Path(pdf_path).stem
            paper_output_dir = self.output_dir / paper_name
            md_path = paper_output_dir / f"{paper_name}.md"

            if paper_output_dir.exists() and md_path.exists():
                results.append(
                    self._make_result(
                        pdf_path,
                        status="success",
                        result_path=str(paper_output_dir),
                    )
                )
            else:
                results.append(
                    self._make_result(pdf_path, status="failed: output not found")
                )
        return results
