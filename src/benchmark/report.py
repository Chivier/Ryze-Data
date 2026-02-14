"""Benchmark report generation in table, CSV, and Markdown formats.

Uses rich.table for console output and plain text for file formats.
"""

import csv
import io
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from src.benchmark.evaluator import BenchmarkReport


def generate_rich_table(report: BenchmarkReport) -> Table:
    """Generate a rich Table for console display.

    Args:
        report: BenchmarkReport to render.

    Returns:
        Rich Table object.
    """
    table = Table(
        title=f"Benchmark Results: {report.dataset_name} "
        f"({report.num_samples} samples)",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Model", style="bold")

    # Add metric columns based on question type
    if report.question_type == "multiple_choice":
        table.add_column("Accuracy", justify="right")
    else:
        table.add_column("EM", justify="right")
        table.add_column("BLEU-4", justify="right")
        table.add_column("ROUGE-L", justify="right")
        table.add_column("Token F1", justify="right")

    table.add_column("Avg OCR Time (s)", justify="right")

    for result in report.model_results:
        row = [_format_model_name(result.model_name)]

        if report.question_type == "multiple_choice":
            row.append(_fmt(result.metrics.get("accuracy")))
        else:
            row.append(_fmt(result.metrics.get("exact_match")))
            row.append(_fmt(result.metrics.get("bleu_4")))
            row.append(_fmt(result.metrics.get("rouge_l")))
            row.append(_fmt(result.metrics.get("token_f1")))

        ocr_time = result.metrics.get("avg_ocr_time")
        row.append(_fmt(ocr_time) if ocr_time is not None else "-")

        table.add_row(*row)

    return table


def print_report(report: BenchmarkReport) -> None:
    """Print the benchmark report to the console.

    Args:
        report: BenchmarkReport to display.
    """
    console = Console()
    table = generate_rich_table(report)
    console.print(table)


def to_csv(report: BenchmarkReport) -> str:
    """Convert benchmark report to CSV string.

    Args:
        report: BenchmarkReport to convert.

    Returns:
        CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = ["model"]
    if report.question_type == "multiple_choice":
        header.append("accuracy")
    else:
        header.extend(["exact_match", "bleu_4", "rouge_l", "token_f1"])
    header.append("avg_ocr_time")
    writer.writerow(header)

    # Data rows
    for result in report.model_results:
        row = [result.model_name]
        if report.question_type == "multiple_choice":
            row.append(_raw(result.metrics.get("accuracy")))
        else:
            row.append(_raw(result.metrics.get("exact_match")))
            row.append(_raw(result.metrics.get("bleu_4")))
            row.append(_raw(result.metrics.get("rouge_l")))
            row.append(_raw(result.metrics.get("token_f1")))
        row.append(_raw(result.metrics.get("avg_ocr_time")))
        writer.writerow(row)

    return output.getvalue()


def to_markdown(report: BenchmarkReport) -> str:
    """Convert benchmark report to Markdown table.

    Args:
        report: BenchmarkReport to convert.

    Returns:
        Markdown formatted table string.
    """
    lines = []

    # Header
    header = ["Model"]
    if report.question_type == "multiple_choice":
        header.append("Accuracy")
    else:
        header.extend(["EM", "BLEU-4", "ROUGE-L", "Token F1"])
    header.append("Avg OCR Time (s)")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Data rows
    for result in report.model_results:
        row = [_format_model_name(result.model_name)]
        if report.question_type == "multiple_choice":
            row.append(_fmt(result.metrics.get("accuracy")))
        else:
            row.append(_fmt(result.metrics.get("exact_match")))
            row.append(_fmt(result.metrics.get("bleu_4")))
            row.append(_fmt(result.metrics.get("rouge_l")))
            row.append(_fmt(result.metrics.get("token_f1")))

        ocr_time = result.metrics.get("avg_ocr_time")
        row.append(_fmt(ocr_time) if ocr_time is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def save_report(
    report: BenchmarkReport,
    output_dir: str,
    formats: Optional[List[str]] = None,
) -> List[str]:
    """Save the report in multiple formats.

    Args:
        report: BenchmarkReport to save.
        output_dir: Directory to save report files.
        formats: List of formats ("csv", "markdown", "table").
            Defaults to all.

    Returns:
        List of saved file paths.
    """
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["csv", "markdown"]

    saved = []

    if "csv" in formats:
        csv_path = output / f"{report.dataset_name}_results.csv"
        csv_path.write_text(to_csv(report), encoding="utf-8")
        saved.append(str(csv_path))

    if "markdown" in formats:
        md_path = output / f"{report.dataset_name}_results.md"
        md_path.write_text(to_markdown(report), encoding="utf-8")
        saved.append(str(md_path))

    return saved


def _format_model_name(name: str) -> str:
    """Format model name for display.

    Args:
        name: Raw model name.

    Returns:
        Formatted display name.
    """
    display_names = {
        "baseline": "Baseline (VLM)",
        "marker": "Marker (Ours)",
        "deepseek-ocr": "DeepSeek v1",
        "glm-ocr": "GLM-OCR",
        "markitdown": "MarkItDown",
        "paddleocr": "PaddleOCR",
    }
    return display_names.get(name, name)


def _fmt(value: Optional[float]) -> str:
    """Format a float for display.

    Args:
        value: Float value or None.

    Returns:
        Formatted string with 4 decimal places, or "-".
    """
    if value is None:
        return "-"
    return f"{value:.4f}"


def _raw(value: Optional[float]) -> str:
    """Format a float for CSV output.

    Args:
        value: Float value or None.

    Returns:
        String representation or empty string.
    """
    if value is None:
        return ""
    return f"{value:.6f}"
