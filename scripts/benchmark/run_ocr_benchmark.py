#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Run OCR-precompute benchmark against a vLLM endpoint pool.

This runner evaluates up to 5 experiments:
1) baseline  : image + prompt
2) baseline1 : image + OCR(v1) + prompt
3) baseline2 : image + OCR(v2) + prompt
4) baseline3 : image + OCR(markitdown) + prompt
5) us        : image + OCR(marker) + prompt
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.benchmark.client_pool import VLLMClientPool
from scripts.utils.benchmark.dataset_adapter import BenchmarkRawSample, load_benchmark_samples
from scripts.utils.benchmark.metrics_ext import (
    compute_free_text_metrics,
    compute_multiple_choice_metrics,
)
from scripts.utils.benchmark.ocr_resolver import (
    parse_experiments,
    read_ocr_markdown,
    resolve_ocr_dir,
)
from scripts.utils.benchmark.prompt_builder import (
    build_baseline_text_prompt,
    build_multimodal_messages,
    build_ocr_text_prompt,
)

logger = logging.getLogger(__name__)


def _read_endpoint_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    tokens = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(part.strip() for part in line.split(",") if part.strip())
    return tokens


def _resolve_endpoints(endpoints_arg: Optional[str], endpoints_file: Path) -> list[str]:
    if endpoints_arg:
        parsed = [token.strip() for token in endpoints_arg.split(",") if token.strip()]
        if parsed:
            return parsed

    from_file = _read_endpoint_file(endpoints_file)
    if from_file:
        return from_file

    return ["http://localhost:8000/v1"]


def _load_answer_cache(path: Path) -> dict[str, str]:
    cached: dict[str, str] = {}
    if not path.exists():
        return cached

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            sample_id = str(item.get("sample_id", "")).strip()
            if not sample_id:
                continue
            cached[sample_id] = str(item.get("answer", ""))
    return cached


def _append_answer_record(
    path: Path,
    sample: BenchmarkRawSample,
    answer: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "sample_id": sample.sample_id,
        "answer": answer,
        "reference": sample.reference,
        "question_type": sample.question_type,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_messages(
    sample: BenchmarkRawSample,
    experiment: str,
    ocr_markdown: Optional[str],
) -> list[dict]:
    if experiment == "baseline":
        text_prompt = build_baseline_text_prompt(
            question=sample.question,
            choices=sample.choices,
        )
    else:
        text_prompt = build_ocr_text_prompt(
            ocr_markdown=ocr_markdown or "",
            question=sample.question,
            choices=sample.choices,
        )
    return build_multimodal_messages(
        image_paths=sample.image_paths,
        text_prompt=text_prompt,
    )


def _infer_one(
    pool: VLLMClientPool,
    sample: BenchmarkRawSample,
    experiment: str,
    ocr_markdown: Optional[str],
) -> str:
    messages = _build_messages(sample=sample, experiment=experiment, ocr_markdown=ocr_markdown)
    return pool.chat(messages=messages).strip()


def _compute_metrics(
    question_type: str,
    predictions: list[str],
    references: list[str],
    *,
    num_missing_ocr: int,
    num_cached: int,
    num_new_requests: int,
) -> dict[str, float]:
    extra = {
        "num_missing_ocr": float(num_missing_ocr),
        "num_cached": float(num_cached),
        "num_new_requests": float(num_new_requests),
    }
    if question_type == "multiple_choice":
        return compute_multiple_choice_metrics(predictions, references, extra=extra)
    return compute_free_text_metrics(predictions, references, extra=extra)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary_columns(question_type: str) -> list[str]:
    base = ["dataset", "question_type", "experiment"]
    tail = ["num_samples", "num_missing_ocr", "num_cached", "num_new_requests"]
    if question_type == "multiple_choice":
        return base + ["accuracy", "macro_precision", "macro_recall", "macro_f1"] + tail
    return base + [
        "exact_match",
        "bleu_4",
        "rouge_l",
        "token_precision",
        "token_recall",
        "token_f1",
    ] + tail


def _write_summary_csv(path: Path, rows: list[dict], question_type: str) -> None:
    columns = _summary_columns(question_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_experiment(
    *,
    experiment: str,
    samples: list[BenchmarkRawSample],
    ocr_dir: Optional[Path],
    pool: VLLMClientPool,
    output_dir: Path,
    fail_on_missing_ocr: bool,
    concurrency: int,
) -> dict:
    exp_dir = output_dir / experiment
    answers_path = exp_dir / "answers.jsonl"
    metrics_path = exp_dir / "metrics.json"

    cached_answers = _load_answer_cache(answers_path)
    predictions_by_id: dict[str, str] = dict(cached_answers)
    num_cached = len(cached_answers)
    num_missing_ocr = 0

    pending: list[tuple[BenchmarkRawSample, Optional[str]]] = []
    for sample in samples:
        if sample.sample_id in cached_answers:
            continue

        ocr_markdown: Optional[str] = None
        if experiment != "baseline":
            if ocr_dir is None:
                raise ValueError(f"OCR directory is required for experiment '{experiment}'.")

            ocr_markdown = read_ocr_markdown(ocr_dir=ocr_dir, sample_id=sample.sample_id)
            if ocr_markdown is None:
                if fail_on_missing_ocr:
                    raise FileNotFoundError(
                        f"Missing OCR markdown for {sample.sample_id} in {ocr_dir}"
                    )
                num_missing_ocr += 1
                predictions_by_id[sample.sample_id] = ""
                _append_answer_record(path=answers_path, sample=sample, answer="")
                continue

        pending.append((sample, ocr_markdown))

    logger.info(
        "Experiment %s: %d cached, %d pending, %d missing OCR",
        experiment,
        num_cached,
        len(pending),
        num_missing_ocr,
    )

    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map: dict[Future[str], BenchmarkRawSample] = {}
            for sample, ocr_markdown in pending:
                future = executor.submit(
                    _infer_one,
                    pool,
                    sample,
                    experiment,
                    ocr_markdown,
                )
                future_map[future] = sample

            completed = 0
            for future in as_completed(future_map):
                sample = future_map[future]
                answer = ""
                try:
                    answer = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Inference failed: experiment=%s sample=%s error=%s",
                        experiment,
                        sample.sample_id,
                        exc,
                    )

                predictions_by_id[sample.sample_id] = answer
                _append_answer_record(path=answers_path, sample=sample, answer=answer)
                completed += 1
                if completed % 50 == 0:
                    logger.info(
                        "Experiment %s progress: %d/%d",
                        experiment,
                        completed,
                        len(pending),
                    )

    predictions = [predictions_by_id.get(sample.sample_id, "") for sample in samples]
    references = [sample.reference for sample in samples]
    question_type = samples[0].question_type

    metrics = _compute_metrics(
        question_type=question_type,
        predictions=predictions,
        references=references,
        num_missing_ocr=num_missing_ocr,
        num_cached=num_cached,
        num_new_requests=len(pending),
    )

    payload = {
        "dataset": samples[0].dataset,
        "question_type": question_type,
        "experiment": experiment,
        "metrics": metrics,
    }
    _write_json(metrics_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR-precompute benchmark")
    parser.add_argument("--dataset", required=True, choices=["arxivqa", "slidevqa"])
    parser.add_argument(
        "--ocr-root",
        default="data/ocr_precompute",
        help="Root path for precomputed OCR outputs",
    )
    parser.add_argument(
        "--data-dir",
        default="data/benchmark_data",
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_results/ocr_bench",
        help="Output directory for answers and metrics",
    )
    parser.add_argument(
        "--experiments",
        default="baseline,baseline1,baseline2,baseline3,us",
        help="Comma-separated experiments subset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to evaluate (0 means all)",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Comma-separated vLLM API base URLs (OpenAI-compatible)",
    )
    parser.add_argument(
        "--endpoints-file",
        default="logs/benchmark/latest/vllm_pool_endpoints.txt",
        help="Fallback endpoint file when --endpoints is omitted",
    )
    parser.add_argument("--model", default="Qwen3-VL-8B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)

    missing_group = parser.add_mutually_exclusive_group()
    missing_group.add_argument(
        "--fail-on-missing-ocr",
        dest="fail_on_missing_ocr",
        action="store_true",
        help="Fail if an OCR sample is missing markdown",
    )
    missing_group.add_argument(
        "--allow-missing-ocr",
        dest="fail_on_missing_ocr",
        action="store_false",
        help="Continue when OCR markdown is missing (stores empty answer)",
    )
    parser.set_defaults(fail_on_missing_ocr=True)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    endpoints = _resolve_endpoints(
        endpoints_arg=args.endpoints,
        endpoints_file=Path(args.endpoints_file),
    )
    experiments = parse_experiments(args.experiments)

    logger.info("Dataset: %s", args.dataset)
    logger.info("Experiments: %s", experiments)
    logger.info("Endpoints: %s", endpoints)

    samples = load_benchmark_samples(
        dataset_name=args.dataset,
        cache_dir=args.data_dir,
        max_samples=args.max_samples,
    )
    logger.info("Loaded %d samples", len(samples))

    pool = VLLMClientPool(
        endpoints=endpoints,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    dataset_output_dir = Path(args.output_dir).resolve() / args.dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for experiment in experiments:
        ocr_dir = resolve_ocr_dir(
            ocr_root=args.ocr_root,
            dataset_name=args.dataset,
            experiment=experiment,
        )
        if ocr_dir is not None and not ocr_dir.exists():
            raise FileNotFoundError(
                f"OCR directory does not exist for experiment '{experiment}': {ocr_dir}"
            )

        result = _evaluate_experiment(
            experiment=experiment,
            samples=samples,
            ocr_dir=ocr_dir,
            pool=pool,
            output_dir=dataset_output_dir,
            fail_on_missing_ocr=args.fail_on_missing_ocr,
            concurrency=args.concurrency,
        )

        flattened = {
            "dataset": result["dataset"],
            "question_type": result["question_type"],
            "experiment": result["experiment"],
        }
        flattened.update(result["metrics"])
        summary_rows.append(flattened)

    summary_json_path = dataset_output_dir / "summary.json"
    summary_csv_path = dataset_output_dir / "summary.csv"
    _write_json(summary_json_path, {"results": summary_rows})
    _write_summary_csv(
        path=summary_csv_path,
        rows=summary_rows,
        question_type=samples[0].question_type,
    )

    logger.info("Summary JSON: %s", summary_json_path)
    logger.info("Summary CSV : %s", summary_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
