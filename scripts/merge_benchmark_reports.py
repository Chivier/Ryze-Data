#!/usr/bin/env python3
"""Merge per-model benchmark CSV reports into a unified report.

Scans data/benchmark_results/run_*/{dataset}_results.csv and combines them
into a single CSV + Markdown report ordered by the canonical model sequence.

Usage:
    python scripts/merge_benchmark_reports.py \
        --results-base data/benchmark_results \
        --dataset arxivqa \
        --output-dir data/benchmark_results
"""

import argparse
import csv
import io
import sys
from pathlib import Path

# Canonical model ordering
MODEL_ORDER = [
    "baseline",
    "deepseek-ocr",
    "deepseek-ocr-v2",
    "markitdown",
    "marker",
]

MODEL_DISPLAY = {
    "baseline": "Baseline (VLM)",
    "deepseek-ocr": "DeepSeek v1",
    "deepseek-ocr-v2": "DeepSeek v2",
    "markitdown": "MarkItDown",
    "marker": "Marker (Ours)",
}


def find_model_csvs(
    results_base: Path, dataset: str
) -> dict[str, Path]:
    """Find per-model CSV files under run_*/ directories.

    Args:
        results_base: Base directory containing run_*/ subdirectories.
        dataset: Dataset name (e.g., "arxivqa").

    Returns:
        Mapping of model_name to CSV path.
    """
    found = {}
    for run_dir in sorted(results_base.glob("run_*")):
        if not run_dir.is_dir():
            continue
        model_name = run_dir.name.removeprefix("run_")
        csv_path = run_dir / f"{dataset}_results.csv"
        if csv_path.exists():
            found[model_name] = csv_path
    return found


def read_model_row(csv_path: Path) -> dict[str, str] | None:
    """Read the single data row from a per-model CSV report.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Dictionary of column_name -> value, or None if empty.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return None


def merge_reports(
    results_base: Path, dataset: str
) -> tuple[list[str], list[dict[str, str]]]:
    """Merge all per-model CSV reports for a dataset.

    Args:
        results_base: Base directory containing run_*/ subdirectories.
        dataset: Dataset name.

    Returns:
        Tuple of (header_columns, rows) where each row is a dict.
    """
    model_csvs = find_model_csvs(results_base, dataset)

    if not model_csvs:
        print(f"[warn] No CSV files found for dataset={dataset}", file=sys.stderr)
        return [], []

    # Determine header from the first available CSV
    header = None
    rows = []

    for model_name in MODEL_ORDER:
        if model_name not in model_csvs:
            print(
                f"[warn] Missing results for model={model_name}, skipping",
                file=sys.stderr,
            )
            continue

        row = read_model_row(model_csvs[model_name])
        if row is None:
            print(
                f"[warn] Empty CSV for model={model_name}, skipping",
                file=sys.stderr,
            )
            continue

        # Use the header from the first valid CSV
        if header is None:
            header = list(row.keys())

        # Ensure model column uses the canonical name
        row["model"] = model_name
        rows.append(row)

    # Check for models not in canonical order (unexpected)
    for model_name in model_csvs:
        if model_name not in MODEL_ORDER:
            print(
                f"[warn] Unknown model={model_name} found, appending at end",
                file=sys.stderr,
            )
            row = read_model_row(model_csvs[model_name])
            if row:
                if header is None:
                    header = list(row.keys())
                row["model"] = model_name
                rows.append(row)

    return header or [], rows


def write_csv(header: list[str], rows: list[dict[str, str]]) -> str:
    """Generate CSV string from merged data.

    Args:
        header: Column names.
        rows: List of row dictionaries.

    Returns:
        CSV formatted string.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def write_markdown(
    header: list[str], rows: list[dict[str, str]], dataset: str
) -> str:
    """Generate Markdown table from merged data.

    Args:
        header: Column names.
        rows: List of row dictionaries.
        dataset: Dataset name for the title.

    Returns:
        Markdown formatted string.
    """
    lines = [f"# Benchmark Results: {dataset}", ""]

    # Build display header
    display_header = []
    for col in header:
        if col == "model":
            display_header.append("Model")
        elif col == "accuracy":
            display_header.append("Accuracy")
        elif col == "exact_match":
            display_header.append("EM")
        elif col == "bleu_4":
            display_header.append("BLEU-4")
        elif col == "rouge_l":
            display_header.append("ROUGE-L")
        elif col == "token_f1":
            display_header.append("Token F1")
        elif col == "avg_ocr_time":
            display_header.append("Avg OCR Time (s)")
        else:
            display_header.append(col)

    lines.append("| " + " | ".join(display_header) + " |")
    lines.append("| " + " | ".join("---" for _ in display_header) + " |")

    for row in rows:
        cells = []
        for col in header:
            val = row.get(col, "")
            if col == "model":
                cells.append(MODEL_DISPLAY.get(val, val))
            elif val == "":
                cells.append("-")
            else:
                try:
                    cells.append(f"{float(val):.4f}")
                except ValueError:
                    cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def print_rich_table(
    header: list[str], rows: list[dict[str, str]], dataset: str
) -> None:
    """Print a formatted table to the terminal using rich (with fallback).

    Args:
        header: Column names.
        rows: List of row dictionaries.
        dataset: Dataset name for the title.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title=f"Merged Benchmark Results: {dataset}",
            show_header=True,
            header_style="bold cyan",
        )

        for col in header:
            if col == "model":
                table.add_column("Model", style="bold")
            elif col in ("accuracy", "exact_match", "bleu_4", "rouge_l", "token_f1"):
                display = {
                    "accuracy": "Accuracy",
                    "exact_match": "EM",
                    "bleu_4": "BLEU-4",
                    "rouge_l": "ROUGE-L",
                    "token_f1": "Token F1",
                }.get(col, col)
                table.add_column(display, justify="right")
            elif col == "avg_ocr_time":
                table.add_column("Avg OCR Time (s)", justify="right")
            else:
                table.add_column(col, justify="right")

        for row in rows:
            cells = []
            for col in header:
                val = row.get(col, "")
                if col == "model":
                    cells.append(MODEL_DISPLAY.get(val, val))
                elif val == "":
                    cells.append("-")
                else:
                    try:
                        cells.append(f"{float(val):.4f}")
                    except ValueError:
                        cells.append(val)
            table.add_row(*cells)

        Console().print(table)

    except ImportError:
        # Fallback: plain text
        print(f"\n=== Merged Benchmark Results: {dataset} ===")
        col_widths = [max(len(col), 12) for col in header]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*header))
        print(fmt.format(*("-" * w for w in col_widths)))
        for row in rows:
            vals = []
            for col in header:
                val = row.get(col, "")
                if col == "model":
                    vals.append(MODEL_DISPLAY.get(val, val))
                else:
                    vals.append(val if val else "-")
            print(fmt.format(*vals))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-model benchmark CSV reports into unified output"
    )
    parser.add_argument(
        "--results-base",
        default="data/benchmark_results",
        help="Base directory containing run_*/ subdirectories",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., arxivqa, slidevqa)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_results",
        help="Directory for merged output files",
    )
    args = parser.parse_args()

    results_base = Path(args.results_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header, rows = merge_reports(results_base, args.dataset)

    if not rows:
        print(f"[error] No results to merge for dataset={args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Print to terminal
    print_rich_table(header, rows, args.dataset)

    # Save CSV
    csv_path = output_dir / f"{args.dataset}_results.csv"
    csv_path.write_text(write_csv(header, rows), encoding="utf-8")
    print(f"[saved] CSV  -> {csv_path}")

    # Save Markdown
    md_path = output_dir / f"{args.dataset}_results.md"
    md_path.write_text(
        write_markdown(header, rows, args.dataset), encoding="utf-8"
    )
    print(f"[saved] Markdown -> {md_path}")


if __name__ == "__main__":
    main()
