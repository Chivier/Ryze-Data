#!/usr/bin/env python3
"""Standalone OCR script using Marker.

Uses legacy chunked-ocr execution style internally:
1) detect devices and derive worker counts
2) create PDF cache on demand
3) run marker_single per sample in parallel workers

Output contract matches other OCR scripts:
- Markdown files under {output_dir}/{sample_id}/{sample_id}.md
- summary logs with success / failed / skipped counters
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared.dataset_loader import load_dataset_samples
from _shared.image_utils import images_to_pdf

logger = logging.getLogger(__name__)

MODEL_NAME = "marker"


def resolve_cli_bin(cli_name: str) -> str:
    """Resolve a CLI path from current venv first, then fallback to PATH."""
    candidates = [
        Path(sys.executable).parent / cli_name,
        Path(sys.executable).resolve().parent / cli_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return cli_name


def _pick_writable_cache_root(primary: Path) -> Path:
    """Pick a writable cache root, with safe fallbacks."""
    candidates = [
        primary,
        Path(__file__).resolve().parent / ".cache",
        Path("/tmp/marker_cache"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"No writable cache directory available. Tried: {', '.join(str(p) for p in candidates)}"
    )


def configure_cache_env(cache_dir: Path) -> None:
    """Route third-party caches into the project cache directory."""
    third_party_cache = _pick_writable_cache_root(cache_dir / "third_party_cache")
    env_defaults = {
        "XDG_CACHE_HOME": str(third_party_cache),
        "HF_HOME": str(third_party_cache / "huggingface"),
        "HF_HUB_CACHE": str(third_party_cache / "huggingface" / "hub"),
        "HF_DATASETS_CACHE": str(third_party_cache / "huggingface" / "datasets"),
        "TORCH_HOME": str(third_party_cache / "torch"),
        "MODEL_CACHE_DIR": str(third_party_cache / "datalab" / "models"),
    }

    for env_name, env_value in env_defaults.items():
        if env_name not in os.environ:
            os.environ[env_name] = env_value
            Path(env_value).mkdir(parents=True, exist_ok=True)


def detect_devices() -> tuple[int, dict[int, int]]:
    """Detect GPUs and derive worker counts (legacy strategy: 3.5GB/worker)."""
    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except ImportError:
        gpu_count = 0

    workers: dict[int, int] = {}
    logger.info("GPU numbers: %d", gpu_count)

    if gpu_count > 0:
        import torch

        for idx in range(gpu_count):
            gpu_mem_gb = (
                torch.cuda.get_device_properties(idx).total_memory / 1024 / 1024 / 1024
            )
            workers_per_gpu = max(1, int(gpu_mem_gb / 3.5))
            workers[idx] = workers_per_gpu
            logger.info("GPU %d: %.2f GB, Workers: %d", idx, gpu_mem_gb, workers_per_gpu)
    else:
        try:
            import psutil

            cpu_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            cpu_cores = os.cpu_count() or 4
        workers[0] = min(cpu_cores, 4)
        logger.info("No GPUs detected, using CPU mode (%d workers)", workers[0])

    return gpu_count, workers


def process_pdf_single(pdf_path: str, output_dir: Path) -> bool:
    """Process a single PDF with marker_single."""
    paper_name = Path(pdf_path).stem
    paper_output_dir = output_dir / paper_name
    md_path = paper_output_dir / f"{paper_name}.md"

    try:
        legacy_cmd = [
            resolve_cli_bin("marker_single"),
            pdf_path,
            str(paper_output_dir),
            "--output_format",
            "markdown",
        ]
        process = subprocess.run(legacy_cmd, capture_output=True, text=True)

        # Keep legacy command shape first, then fallback to current Marker CLI
        # (`--output_dir`) if the old positional form is rejected.
        if process.returncode != 0 and "unexpected extra argument" in (
            process.stderr or ""
        ).lower():
            modern_cmd = [
                resolve_cli_bin("marker_single"),
                pdf_path,
                "--output_dir",
                str(paper_output_dir),
                "--output_format",
                "markdown",
            ]
            process = subprocess.run(modern_cmd, capture_output=True, text=True)

        if process.returncode == 0 and paper_output_dir.exists():
            if not md_path.exists():
                md_candidates = list(paper_output_dir.glob("**/*.md"))
                if md_candidates:
                    md_candidates[0].rename(md_path)
            if md_path.exists():
                return True

        stderr = process.stderr.strip() if process.stderr else "output not found"
        logger.error("marker_single failed for %s: %s", paper_name, stderr)
        return False

    except Exception as exc:
        logger.error("marker_single exception for %s: %s", paper_name, exc)
        return False


def process_sample(
    sample_id: str,
    image_paths: list[str],
    output_dir: Path,
    pdf_cache_dir: Path,
) -> str:
    """Run one pipelined unit: image->PDF (if needed) and OCR.

    Returns:
        One of "success", "failed", or "skipped".
    """
    md_path = output_dir / sample_id / f"{sample_id}.md"
    if md_path.exists():
        return "skipped"

    pdf_path = pdf_cache_dir / f"{sample_id}.pdf"
    if not pdf_path.exists():
        try:
            images_to_pdf(image_paths, str(pdf_path))
        except Exception as exc:
            logger.error("Failed to create PDF for %s: %s", sample_id, exc)
            return "failed"

    ok = process_pdf_single(str(pdf_path), output_dir=output_dir)
    return "success" if ok else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Marker OCR on HF datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxivqa", "slidevqa"],
        help="Dataset to process",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/ocr_precompute/marker/{dataset})",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/benchmark_data",
        help="Shared image cache directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to process (0=all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Pipeline worker count (0=auto from device detection)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help=(
            "GPU selection for Marker via CUDA_VISIBLE_DEVICES, "
            "e.g. '0' or '0,1'. Use 'cpu' to force CPU mode."
        ),
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional HuggingFace endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        logger.info("Using HuggingFace endpoint: %s", args.hf_endpoint)

    if args.gpu is not None:
        gpu_value = str(args.gpu).strip()
        if gpu_value.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("GPU selection: forcing CPU mode (CUDA disabled)")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_value
            logger.info(
                "GPU selection: CUDA_VISIBLE_DEVICES=%s",
                os.environ["CUDA_VISIBLE_DEVICES"],
            )

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = (project_root / args.cache_dir).resolve()
    configure_cache_env(cache_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "data" / "ocr_precompute" / MODEL_NAME / args.dataset).resolve()

    pdf_cache_dir = project_root / "data" / "benchmark_data" / "ocr_pdfs" / args.dataset
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Marker OCR — dataset=%s, output=%s", args.dataset, output_dir)

    start_time = time.time()
    success = 0
    failed = 0
    skipped = 0
    completed = 0

    try:
        samples = load_dataset_samples(args.dataset, str(cache_dir), args.max_samples)
    except Exception as exc:
        logger.error("Dataset loading failed: %s", exc)
        logger.error(
            "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
        )
        return 1

    gpu_count, device_workers = detect_devices()
    total_workers = sum(device_workers.values())
    worker_count = args.workers if args.workers > 0 else max(1, total_workers)
    max_in_flight = max(worker_count * 4, 16)
    logger.info(
        "Using %d workers across %d devices (max in-flight=%d)",
        worker_count,
        len(device_workers),
        max_in_flight,
    )

    def consume_done(
        done_futures: set[Future[str]],
        progress: Progress,
        task_id: int,
    ) -> None:
        nonlocal success, failed, skipped, completed
        for fut in done_futures:
            try:
                status = fut.result()
            except Exception as exc:
                logger.error("Pipeline worker crashed: %s", exc)
                status = "failed"

            if status == "success":
                success += 1
            elif status == "failed":
                failed += 1
            else:
                skipped += 1
            completed += 1

            progress.update(
                task_id,
                completed=completed,
                description=(
                    f"Processing samples (ok={success}, fail={failed}, skip={skipped})"
                ),
            )

            if (success + failed) > 0 and (success + failed) % 50 == 0:
                logger.info(
                    "Progress: %d success, %d failed, %d skipped",
                    success,
                    failed,
                    skipped,
                )

    try:
        in_flight: set[Future[str]] = set()
        submitted = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                "Processing samples (ok=0, fail=0, skip=0)",
                total=0,
                start=False,
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="marker-pipeline",
            ) as executor:
                for sample in samples:
                    fut = executor.submit(
                        process_sample,
                        sample.sample_id,
                        sample.image_paths,
                        output_dir,
                        pdf_cache_dir,
                    )
                    in_flight.add(fut)
                    submitted += 1
                    progress.update(task_id, total=submitted)
                    if submitted == 1:
                        progress.start_task(task_id)

                    if len(in_flight) >= max_in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        consume_done(done, progress, task_id)

                if in_flight:
                    done, _ = wait(in_flight)
                    consume_done(done, progress, task_id)

            if submitted == 0:
                progress.update(
                    task_id,
                    total=1,
                    completed=1,
                    description="No samples found",
                )
            else:
                progress.update(task_id, total=submitted, completed=completed)
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc)
        logger.error(
            "If HuggingFace is unreachable, retry with --hf-endpoint https://hf-mirror.com"
        )
        return 1

    elapsed = time.time() - start_time
    logger.info(
        "Done in %.1fs — %d success, %d failed, %d skipped",
        elapsed,
        success,
        failed,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
