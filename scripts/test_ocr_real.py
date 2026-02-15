#!/usr/bin/env python3
"""Real-file OCR test runner.

Tests a single OCR model against real ArxivQA PDFs.

Usage (run from repo root with PYTHONPATH=.):
    PYTHONPATH=. scripts/utils/markitdown/.venv/bin/python scripts/test_ocr_real.py markitdown
    PYTHONPATH=. scripts/utils/marker/.venv/bin/python scripts/test_ocr_real.py marker --gpu 3
    PYTHONPATH=. scripts/utils/paddleocr/.venv/bin/python scripts/test_ocr_real.py paddleocr --gpu 3
    PYTHONPATH=. scripts/utils/deepseek_ocr_v1/.venv/bin/python scripts/test_ocr_real.py deepseek-ocr --gpu 3
    PYTHONPATH=. scripts/utils/glm_ocr/.venv/bin/python scripts/test_ocr_real.py glm-ocr --gpu 3
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Test PDFs (small / medium / large)
# ---------------------------------------------------------------------------
PDF_DIR = Path("data/benchmark_results/run_baseline/pdfs")
TEST_PDFS = [
    PDF_DIR / "arxivqa_1000.pdf",  # ~119 KB (small-medium)
    PDF_DIR / "arxivqa_0.pdf",  # ~183 KB (medium)
    PDF_DIR / "arxivqa_384.pdf",  # ~1.5 MB (large)
]

# ---------------------------------------------------------------------------
# Model → constructor kwargs mapping
# ---------------------------------------------------------------------------
MODEL_KWARGS = {
    "markitdown": lambda args: {},
    "marker": lambda args: {},
    # PaddleOCR: CUDA_VISIBLE_DEVICES is already set, so always use gpu:0.
    "paddleocr": lambda args: {"device": "gpu:0"},
    "deepseek-ocr": lambda args: {},
    "glm-ocr": lambda args: {"backend": args.backend},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a single OCR model on real PDFs.")
    parser.add_argument("model", help="OCR model name (e.g. markitdown, marker, paddleocr, deepseek-ocr, glm-ocr)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--backend", default="vllm", help="Backend for GLM-OCR (vllm or api)")
    return parser.parse_args()


# Models that extract embedded text (not OCR) — expected to produce
# little/no output on image-only PDFs like ArxivQA.
TEXT_EXTRACTORS = {"markitdown", "marker"}


def validate_output(paper_name: str, output_dir: Path, model_name: str = "") -> tuple[str, str]:
    """Validate OCR output for a single paper.

    Returns:
        (status, message) where status is PASS, WARN, or FAIL.
    """
    md_path = output_dir / paper_name / f"{paper_name}.md"
    temp_dir = output_dir / paper_name / "temp_pages"

    if not md_path.exists():
        return "FAIL", f"Output file not found: {md_path}"

    content = md_path.read_text(encoding="utf-8")

    # Text extractors on image-only PDFs: just verify file was created.
    if model_name in TEXT_EXTRACTORS:
        return "PASS", f"OK ({len(content)} chars)"

    if not content:
        return "FAIL", "Output file is empty"

    if len(content) < 50:
        return "WARN", f"Output is very short ({len(content)} chars)"

    error_markers = ["error", "traceback", "exception"]
    content_lower = content.lower()
    for marker in error_markers:
        if marker in content_lower and len(content) < 500:
            return "WARN", f"Output may contain error text (found '{marker}')"

    if temp_dir.exists():
        return "WARN", "temp_pages directory was not cleaned up"

    return "PASS", f"OK ({len(content)} chars)"


def main():
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES before any GPU library is imported.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[env] CUDA_VISIBLE_DEVICES={args.gpu}")

    # Now import OCR modules (which may import torch/vllm).
    from src.ocr import OCRRegistry

    print(f"\n{'='*60}")
    print(f"  Testing OCR model: {args.model}")
    print(f"{'='*60}\n")

    # Verify model is registered and available.
    model_class = OCRRegistry.get_model_class(args.model)
    if model_class is None:
        print(f"FAIL: Model '{args.model}' is not registered.")
        print(f"  Available: {OCRRegistry.list_all()}")
        sys.exit(1)

    if not model_class.is_available():
        print(f"FAIL: Model '{args.model}' dependencies are not available.")
        sys.exit(1)

    print(f"[info] Model '{args.model}' is registered and available")

    # Create a temp output directory for this test run.
    output_dir = Path(tempfile.mkdtemp(prefix=f"ocr_test_{args.model}_"))
    print(f"[info] Output dir: {output_dir}")

    # Verify test PDFs exist.
    available_pdfs = [p for p in TEST_PDFS if p.exists()]
    if not available_pdfs:
        print("FAIL: No test PDFs found. Expected them in:")
        for p in TEST_PDFS:
            print(f"  {p}")
        sys.exit(1)
    print(f"[info] Testing with {len(available_pdfs)} PDFs")

    # Get model kwargs.
    kwargs_fn = MODEL_KWARGS.get(args.model, lambda a: {})
    kwargs = kwargs_fn(args)
    if kwargs:
        print(f"[info] Model kwargs: {kwargs}")

    # Instantiate the model.
    model = OCRRegistry.get_model(args.model, output_dir=str(output_dir), **kwargs)

    # Run tests.
    results = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for pdf_path in available_pdfs:
        paper_name = pdf_path.stem
        pdf_size = pdf_path.stat().st_size
        print(f"\n--- {paper_name} ({pdf_size:,} bytes) ---")

        t0 = time.time()
        try:
            result = model.process_single(str(pdf_path.resolve()))
            elapsed = time.time() - t0

            print(f"  ocr_status: {result.ocr_status}")
            print(f"  time: {elapsed:.1f}s")

            if result.ocr_status != "success":
                status, msg = "FAIL", f"OCR status: {result.ocr_status}"
            else:
                status, msg = validate_output(paper_name, output_dir, model_name=args.model)

            print(f"  result: {status} - {msg}")
            results[status] += 1

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  EXCEPTION after {elapsed:.1f}s: {e}")
            results["FAIL"] += 1

    # Summary.
    print(f"\n{'='*60}")
    print(f"  Summary [{args.model}]: {results['PASS']} PASS, {results['WARN']} WARN, {results['FAIL']} FAIL")
    print(f"{'='*60}\n")

    # Clean up.
    shutil.rmtree(output_dir, ignore_errors=True)

    # Exit code: non-zero if any failures.
    sys.exit(1 if results["FAIL"] > 0 else 0)


if __name__ == "__main__":
    main()
