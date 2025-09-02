#!/usr/bin/env python3
"""
Ryze-Data CLI: Comprehensive data processing framework for scientific papers
"""

import click
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config_manager import config
from src.cli.data_inspector import inspect_command


@click.group()
@click.option('--config', '-c', default='config.json', help='Path to configuration file')
@click.option('--env', '-e', default='.env', help='Path to environment file')
@click.pass_context
def cli(ctx, config_file, env):
    """Ryze-Data: Comprehensive data processing framework for scientific papers"""
    ctx.ensure_object(dict)
    
    # Load configuration
    config.load(config_file)
    
    # Validate configuration
    if not config.validate():
        click.echo("Configuration validation failed. Please check your settings.", err=True)
        sys.exit(1)
    
    ctx.obj['config'] = config
    click.echo(f"Loaded configuration from {config_file}")


@cli.command()
@click.pass_context
def scrape(ctx):
    """Scrape Nature articles metadata"""
    from src.scrapers.nature_scraper import NatureScraper
    
    cfg = ctx.obj['config']
    click.echo("Starting Nature article scraping...")
    
    scraper = NatureScraper(output_dir=cfg.paths.nature_data)
    scraper.run()
    
    click.echo("Scraping completed!")


@cli.command()
@click.option('--workers', '-w', default=4, help='Number of parallel workers')
@click.option('--servers', help='Comma-separated list of download servers')
@click.pass_context
def download(ctx, workers, servers):
    """Download PDFs from scraped metadata"""
    from src.downloaders.pdf_downloader import PDFDownloader
    
    cfg = ctx.obj['config']
    click.echo(f"Starting PDF downloads with {workers} workers...")
    
    downloader = PDFDownloader(
        metadata_dir=cfg.paths.nature_data,
        output_dir=cfg.paths.pdf_dir,
        workers=workers,
        servers=servers.split(',') if servers else cfg.servers.get('download_servers', [])
    )
    downloader.run()
    
    click.echo("Download completed!")


@cli.command()
@click.option('--input-dir', help='Directory containing PDFs')
@click.option('--output-dir', help='Directory for OCR output')
@click.option('--batch-size', default=10, help='Batch size for processing')
@click.pass_context
def ocr(ctx, input_dir, output_dir, batch_size):
    """Run OCR on PDF files"""
    cfg = ctx.obj['config']
    
    input_path = input_dir or cfg.paths.pdf_dir
    output_path = output_dir or cfg.paths.ocr_output
    
    click.echo(f"Starting OCR processing...")
    click.echo(f"Input: {input_path}")
    click.echo(f"Output: {output_path}")
    
    # Run the chunked OCR script
    import subprocess
    result = subprocess.run([
        sys.executable,
        'src/chunked-ocr.py',
        '--input', input_path,
        '--output', output_path,
        '--batch-size', str(batch_size)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        click.echo("OCR processing completed!")
    else:
        click.echo(f"OCR processing failed: {result.stderr}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--threads', '-t', default=8, help='Number of threads for processing')
@click.pass_context
def extract(ctx, threads):
    """Extract figures and text from OCR results"""
    from src.processors.figure_extractor import FigureExtractor
    
    cfg = ctx.obj['config']
    click.echo(f"Starting figure extraction with {threads} threads...")
    
    extractor = FigureExtractor(
        input_dir=cfg.paths.ocr_output,
        output_dir=cfg.paths.vlm_preprocessing,
        threads=threads
    )
    extractor.run()
    
    click.echo("Figure extraction completed!")


@cli.command(name='generate-qa')
@click.option('--mode', type=click.Choice(['text', 'vision', 'both']), default='both', help='QA generation mode')
@click.option('--model', help='Model to use for generation')
@click.option('--workers', '-w', default=4, help='Number of parallel workers (vision mode)')
@click.option('--qa-ratio', default=8, help='Number of QA pairs per figure/section')
@click.pass_context
def generate_qa(ctx, mode, model, workers, qa_ratio):
    """Generate QA pairs from processed data"""
    cfg = ctx.obj['config']
    
    if mode in ['text', 'both']:
        click.echo("Generating text QA pairs...")
        from src.generators.text_qa_generator import TextQAGenerator
        
        text_gen = TextQAGenerator(
            ocr_dir=cfg.paths.ocr_output,
            abstract_dir=cfg.paths.abstract_dir,
            output_dir=cfg.paths.sft_data,
            model=model or cfg.qa_generation_model.model,
            qa_ratio=qa_ratio
        )
        text_gen.run()
        click.echo("Text QA generation completed!")
    
    if mode in ['vision', 'both']:
        click.echo(f"Generating vision QA pairs with {workers} workers...")
        from src.generators.vision_qa_generator import VisionQAGenerator
        
        vision_gen = VisionQAGenerator(
            vlm_dir=cfg.paths.vlm_preprocessing,
            abstract_dir=cfg.paths.abstract_dir,
            output_dir=cfg.paths.vlm_sft_data,
            model=model or cfg.vision_model.model,
            workers=workers,
            qa_ratio=qa_ratio
        )
        vision_gen.run()
        click.echo("Vision QA generation completed!")


@cli.command()
@click.option('--stages', '-s', multiple=True, 
              type=click.Choice(['scrape', 'download', 'ocr', 'extract', 'qa']),
              help='Stages to run (default: all)')
@click.option('--workers', '-w', default=4, help='Number of parallel workers')
@click.pass_context
def pipeline(ctx, stages, workers):
    """Run complete processing pipeline"""
    # If no stages specified, run all
    if not stages:
        stages = ['scrape', 'download', 'ocr', 'extract', 'qa']
    
    click.echo(f"Running pipeline stages: {', '.join(stages)}")
    
    if 'scrape' in stages:
        ctx.invoke(scrape)
    
    if 'download' in stages:
        ctx.invoke(download, workers=workers)
    
    if 'ocr' in stages:
        ctx.invoke(ocr)
    
    if 'extract' in stages:
        ctx.invoke(extract)
    
    if 'qa' in stages:
        ctx.invoke(generate_qa, mode='both', workers=workers)
    
    click.echo("Pipeline completed successfully!")


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration"""
    cfg = ctx.obj['config']
    
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


def main():
    """Main entry point for the CLI"""
    try:
        cli(obj={})
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()