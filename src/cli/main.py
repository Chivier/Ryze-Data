#!/usr/bin/env python3
"""
Ryze-Data CLI: Comprehensive data processing framework for scientific papers
"""

import sys
from pathlib import Path

import click

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.cli.data_inspector import inspect_command
from src.config_manager import config


@click.group()
@click.option(
    "--config",
    "-c",
    "config_file",
    default="config.json",
    help="Path to configuration file",
)
@click.option("--env", "-e", default=".env", help="Path to environment file")
@click.pass_context
def cli(ctx, config_file, env):
    """Ryze-Data: Comprehensive data processing framework for scientific papers"""
    ctx.ensure_object(dict)

    # Load configuration
    config.load(config_file)

    # Validate configuration
    if not config.validate():
        click.echo(
            "Configuration validation failed. Please check your settings.", err=True
        )
        sys.exit(1)

    ctx.obj["config"] = config
    click.echo(f"Loaded configuration from {config_file}")


@cli.command()
@click.pass_context
def scrape(ctx):
    """Scrape Nature articles metadata"""
    from src.scrapers.nature_scraper import NatureScraper

    cfg = ctx.obj["config"]
    click.echo("Starting Nature article scraping...")

    scraper = NatureScraper(output_dir=cfg.paths.nature_data)
    scraper.run()

    click.echo("Scraping completed!")


@cli.command()
@click.option("--workers", "-w", default=4, help="Number of parallel workers")
@click.option("--servers", help="Comma-separated list of download servers")
@click.pass_context
def download(ctx, workers, servers):
    """Download PDFs from scraped metadata"""
    from src.downloaders.pdf_downloader import PDFDownloader

    cfg = ctx.obj["config"]
    click.echo(f"Starting PDF downloads with {workers} workers...")

    downloader = PDFDownloader(
        metadata_dir=cfg.paths.nature_data,
        output_dir=cfg.paths.pdf_dir,
        workers=workers,
        servers=(
            servers.split(",") if servers else cfg.servers.get("download_servers", [])
        ),
    )
    downloader.run()

    click.echo("Download completed!")


@cli.command()
@click.option("--input-dir", help="Directory containing PDFs")
@click.option("--output-dir", help="Directory for OCR output")
@click.option("--batch-size", default=10, help="Batch size for processing")
@click.option(
    "--ocr-model",
    default="marker",
    help="OCR model to use (see list-ocr-models)",
)
@click.pass_context
def ocr(ctx, input_dir, output_dir, batch_size, ocr_model):
    """Run OCR on PDF files"""
    import time

    from src.ocr import OCRRegistry, OCRStatusTracker, detect_devices

    cfg = ctx.obj["config"]

    input_path = Path(input_dir or cfg.paths.pdf_dir)
    output_path = Path(output_dir or cfg.paths.ocr_output)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Starting OCR processing with model: {ocr_model}")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")

    # Collect PDF files
    pdf_files = [
        str(input_path / f)
        for f in sorted(input_path.iterdir())
        if f.is_file() and f.suffix == ".pdf"
    ]
    if not pdf_files:
        click.echo("No PDF files found in input directory.", err=True)
        sys.exit(1)
    click.echo(f"Found {len(pdf_files)} PDF files")

    # Instantiate the OCR model
    try:
        model = OCRRegistry.get_model(ocr_model, output_dir=str(output_path))
    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Detect devices
    gpu_count, device_workers = detect_devices()
    total_workers = sum(device_workers.values())
    click.echo(f"Using {total_workers} workers across {len(device_workers)} devices")

    # Set up status tracking
    tracker = OCRStatusTracker(
        output_dir=str(output_path),
        task_name="ocr_processing",
        metrics_port=cfg.monitoring.metrics_port,
    )
    tracker.total_files = len(pdf_files)
    tracker.start_server()

    # Process PDFs
    start_time = time.time()

    if model.supports_batch() and gpu_count > 1:
        avg_workers = total_workers // max(1, gpu_count)
        results = model.process_batch(pdf_files, gpu_count, avg_workers)
        for result in results:
            tracker.record_result(result)
    else:
        for pdf_path in pdf_files:
            result = model.process_single(pdf_path)
            tracker.record_result(result)

    tracker.flush()
    elapsed = time.time() - start_time

    # Summary
    click.echo(f"\n{'=' * 50}")
    click.echo("OCR Processing Summary")
    click.echo(f"{'=' * 50}")
    click.echo(f"Total files: {tracker.total_files}")
    click.echo(f"Successful: {tracker.completed_files}")
    click.echo(f"Failed: {tracker.failed_files}")
    click.echo(f"Total time: {elapsed:.2f} seconds")
    if tracker.total_files > 0:
        click.echo(
            f"Average time per file: {elapsed / tracker.total_files:.2f} seconds"
        )
    click.echo(
        f"\nStatus available at: "
        f"http://localhost:{cfg.monitoring.metrics_port}/status"
    )
    click.echo(f"Results saved to: {tracker.csv_path}")

    tracker.stop_server()
    click.echo("OCR processing completed!")


@cli.command(name="list-ocr-models")
def list_ocr_models():
    """List available OCR models and their install status"""
    from src.ocr import OCRRegistry

    models = OCRRegistry.list_models_with_status()
    if not models:
        click.echo("No OCR models registered.")
        return

    click.echo("\nRegistered OCR models:\n")
    for entry in models:
        status_str = (
            click.style(f"[{entry['status']}]", fg="green")
            if entry["status"] == "available"
            else click.style(f"[{entry['status']}]", fg="yellow")
        )
        click.echo(f"  {entry['name']:<14} {status_str}")
    click.echo()


@cli.command()
@click.option("--threads", "-t", default=8, help="Number of threads for processing")
@click.pass_context
def extract(ctx, threads):
    """Extract figures and text from OCR results"""
    from src.processors.figure_extractor import FigureExtractor

    cfg = ctx.obj["config"]
    click.echo(f"Starting figure extraction with {threads} threads...")

    extractor = FigureExtractor(
        input_dir=cfg.paths.ocr_output,
        output_dir=cfg.paths.vlm_preprocessing,
        threads=threads,
    )
    extractor.run()

    click.echo("Figure extraction completed!")


@cli.command(name="generate-qa")
@click.option(
    "--mode",
    type=click.Choice(["text", "vision", "both"]),
    default="both",
    help="QA generation mode",
)
@click.option("--model", help="Model to use for generation")
@click.option(
    "--workers", "-w", default=4, help="Number of parallel workers (vision mode)"
)
@click.option("--qa-ratio", default=8, help="Number of QA pairs per figure/section")
@click.option(
    "--quality-filter/--no-quality-filter",
    default=False,
    help="Enable quality filtering (adds API cost)",
)
@click.pass_context
def generate_qa(ctx, mode, model, workers, qa_ratio, quality_filter):
    """Generate QA pairs from processed data"""
    cfg = ctx.obj["config"]
    quality_threshold = cfg.processing.quality_threshold

    if quality_filter:
        click.echo(f"Quality filtering enabled (threshold: {quality_threshold})")

    if mode in ["text", "both"]:
        click.echo("Generating text QA pairs...")
        from src.generators.text_qa_generator import TextQAGenerator

        text_gen = TextQAGenerator(
            ocr_dir=cfg.paths.ocr_output,
            abstract_dir=cfg.paths.abstract_dir,
            output_dir=cfg.paths.sft_data,
            model=model or cfg.qa_generation_model.model,
            qa_ratio=qa_ratio,
            quality_filter=quality_filter,
            quality_threshold=quality_threshold,
        )
        text_gen.run()
        click.echo("Text QA generation completed!")

    if mode in ["vision", "both"]:
        click.echo(f"Generating vision QA pairs with {workers} workers...")
        from src.generators.vision_qa_generator import VisionQAGenerator

        vision_gen = VisionQAGenerator(
            vlm_dir=cfg.paths.vlm_preprocessing,
            abstract_dir=cfg.paths.abstract_dir,
            output_dir=cfg.paths.vlm_sft_data,
            model=model or cfg.vision_model.model,
            workers=workers,
            qa_ratio=qa_ratio,
            quality_filter=quality_filter,
            quality_threshold=quality_threshold,
        )
        vision_gen.run()
        click.echo("Vision QA generation completed!")


@cli.command()
@click.option(
    "--stages",
    "-s",
    multiple=True,
    type=click.Choice(["scrape", "download", "ocr", "extract", "qa"]),
    help="Stages to run (default: all)",
)
@click.option("--workers", "-w", default=4, help="Number of parallel workers")
@click.option(
    "--quality-filter/--no-quality-filter",
    default=False,
    help="Enable QA quality filtering",
)
@click.pass_context
def pipeline(ctx, stages, workers, quality_filter):
    """Run complete processing pipeline"""
    # If no stages specified, run all
    if not stages:
        stages = ["scrape", "download", "ocr", "extract", "qa"]

    click.echo(f"Running pipeline stages: {', '.join(stages)}")

    if "scrape" in stages:
        ctx.invoke(scrape)

    if "download" in stages:
        ctx.invoke(download, workers=workers)

    if "ocr" in stages:
        ctx.invoke(ocr)

    if "extract" in stages:
        ctx.invoke(extract)

    if "qa" in stages:
        ctx.invoke(
            generate_qa, mode="both", workers=workers, quality_filter=quality_filter
        )

    click.echo("Pipeline completed successfully!")


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    cfg = ctx.obj["config"]

    click.echo("\n=== Current Configuration ===\n")
    click.echo(f"Project: {cfg.project.name} v{cfg.project.version}")
    click.echo(f"Environment: {cfg.project.environment}\n")

    click.echo("Paths:")
    for key, value in cfg.paths.__dict__.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nProcessing:")
    click.echo(f"  Workers: {cfg.processing.parallel_workers}")
    click.echo(f"  QA Ratio: {cfg.processing.qa_ratio}")
    click.echo(f"  Quality Threshold: {cfg.processing.quality_threshold}")

    click.echo("\nModels:")
    click.echo(f"  LLM: {cfg.qa_generation_model.model}")
    click.echo(f"  Vision: {cfg.vision_model.model}")

    click.echo("\nMonitoring:")
    click.echo(f"  Status Port: {cfg.monitoring.metrics_port}")
    click.echo(f"  Log Level: {cfg.monitoring.log_level}")


# Add the inspect command
cli.add_command(inspect_command)


# ============== Benchmark Commands ==============


@cli.group()
def benchmark():
    """OCR benchmark evaluation commands"""
    pass


@benchmark.command(name="run")
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(["arxivqa", "slidevqa"]),
    help="Benchmark dataset to evaluate on",
)
@click.option(
    "--ocr-models",
    default="marker,deepseek-ocr,deepseek-ocr-v2,markitdown",
    help="Comma-separated list of OCR models to evaluate",
)
@click.option(
    "--include-baseline/--no-baseline",
    default=True,
    help="Include vision-only baseline (Path 0)",
)
@click.option(
    "--max-samples",
    default=0,
    help="Maximum samples to evaluate (0 = all)",
)
@click.option(
    "--qa-model",
    default="Qwen3-VL-8B",
    help="QA model name for API calls",
)
@click.option(
    "--qa-api-base",
    default="http://localhost:8000/v1",
    help="Base URL for the QA model API",
)
@click.option(
    "--qa-api-key",
    default="EMPTY",
    help="API key for the QA model (EMPTY for local vLLM)",
)
@click.option(
    "--results-dir",
    default="data/benchmark_results",
    help="Directory for caching results",
)
def benchmark_run(
    dataset,
    ocr_models,
    include_baseline,
    max_samples,
    qa_model,
    qa_api_base,
    qa_api_key,
    results_dir,
):
    """Run OCR benchmark evaluation on a dataset"""
    from src.benchmark.datasets.arxivqa import ArxivQADataset
    from src.benchmark.datasets.slidevqa import SlideVQADataset
    from src.benchmark.evaluator import BenchmarkEvaluator
    from src.benchmark.qa_client import QwenQAClient
    from src.benchmark.report import print_report, save_report

    # Select dataset
    dataset_map = {
        "arxivqa": ArxivQADataset,
        "slidevqa": SlideVQADataset,
    }
    dataset_cls = dataset_map[dataset]
    ds = dataset_cls(data_dir=results_dir)

    # Initialize QA client
    qa_client = QwenQAClient(
        model=qa_model,
        api_base=qa_api_base,
        api_key=qa_api_key,
    )

    # Initialize evaluator
    evaluator = BenchmarkEvaluator(
        qa_client=qa_client,
        results_dir=results_dir,
    )

    # Parse OCR model list
    model_list = [m.strip() for m in ocr_models.split(",") if m.strip()]

    click.echo(f"Running benchmark on {dataset} with models: {model_list}")
    if include_baseline:
        click.echo("Including vision baseline (Path 0)")
    if max_samples > 0:
        click.echo(f"Max samples: {max_samples}")

    # Run evaluation
    report = evaluator.run(
        dataset=ds,
        ocr_models=model_list,
        max_samples=max_samples,
        include_baseline=include_baseline,
    )

    # Display and save results
    print_report(report)
    saved = save_report(report, results_dir)
    for path in saved:
        click.echo(f"Report saved to: {path}")


@benchmark.command(name="report")
@click.option(
    "--results-dir",
    default="data/benchmark_results",
    help="Directory containing benchmark results",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "csv", "markdown"]),
    default="table",
    help="Output format",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Choice(["arxivqa", "slidevqa"]),
    help="Dataset to show report for",
)
def benchmark_report(results_dir, output_format, dataset):
    """View benchmark results in various formats"""
    from pathlib import Path as P

    results_path = P(results_dir)

    if output_format == "csv":
        csv_file = results_path / f"{dataset}_results.csv"
        if csv_file.exists():
            click.echo(csv_file.read_text())
        else:
            click.echo(f"No CSV results found at {csv_file}", err=True)

    elif output_format == "markdown":
        md_file = results_path / f"{dataset}_results.md"
        if md_file.exists():
            click.echo(md_file.read_text())
        else:
            click.echo(f"No Markdown results found at {md_file}", err=True)

    else:  # table
        click.echo(
            "Run 'benchmark run' first to generate results, "
            "then use --format csv or --format markdown to view saved reports."
        )


def main():
    """Main entry point for the CLI"""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
