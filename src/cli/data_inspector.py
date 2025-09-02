#!/usr/bin/env python3
"""
Data Inspector CLI: Inspect and sample data at various pipeline stages
"""

import click
import json
import random
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint


console = Console()


class DataInspector:
    """Inspect data at various stages of the pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.stages = {
            'scraping': {
                'path': config.paths.nature_data,
                'pattern': '*.csv',
                'description': 'Scraped Nature articles metadata'
            },
            'pdf': {
                'path': config.paths.pdf_dir,
                'pattern': '*.pdf',
                'description': 'Downloaded PDF files'
            },
            'ocr': {
                'path': config.paths.ocr_output,
                'pattern': '*/*.md',
                'description': 'OCR extracted text and images'
            },
            'figures': {
                'path': config.paths.vlm_preprocessing,
                'pattern': '*.json',
                'description': 'Extracted figures with context'
            },
            'abstracts': {
                'path': config.paths.abstract_dir,
                'pattern': '*.txt',
                'description': 'Paper abstracts'
            },
            'qa-text': {
                'path': config.paths.sft_data,
                'pattern': '*_qa.jsonl',
                'description': 'Text-based QA pairs'
            },
            'qa-vision': {
                'path': config.paths.vlm_sft_data,
                'pattern': '*_vision_qa.jsonl',
                'description': 'Vision-based QA pairs'
            }
        }
    
    def get_stage_info(self, stage: str) -> Dict[str, Any]:
        """Get information about a specific stage"""
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}. Available stages: {', '.join(self.stages.keys())}")
        
        stage_config = self.stages[stage]
        path = Path(stage_config['path'])
        
        info = {
            'stage': stage,
            'description': stage_config['description'],
            'path': str(path),
            'exists': path.exists(),
            'files': [],
            'count': 0,
            'total_size': 0
        }
        
        if path.exists():
            if path.is_dir():
                files = list(path.glob(stage_config['pattern']))
                info['files'] = [str(f) for f in files]
                info['count'] = len(files)
                info['total_size'] = sum(f.stat().st_size for f in files if f.is_file())
            elif path.is_file():
                info['files'] = [str(path)]
                info['count'] = 1
                info['total_size'] = path.stat().st_size
        
        return info
    
    def sample_file(self, file_path: str, sample_size: int = 5) -> Dict[str, Any]:
        """Sample content from a file"""
        path = Path(file_path)
        if not path.exists():
            return {'error': f"File not found: {file_path}"}
        
        sample = {
            'file': str(path),
            'size': path.stat().st_size,
            'type': path.suffix,
            'content': None
        }
        
        try:
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        sample['content'] = data[:sample_size] if len(data) > sample_size else data
                        sample['total_items'] = len(data)
                    else:
                        sample['content'] = data
            
            elif path.suffix == '.jsonl':
                lines = []
                with open(path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= sample_size:
                            break
                        lines.append(json.loads(line.strip()))
                sample['content'] = lines
                # Count total lines
                with open(path, 'r', encoding='utf-8') as f:
                    sample['total_items'] = sum(1 for _ in f)
            
            elif path.suffix == '.csv':
                rows = []
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= sample_size:
                            break
                        rows.append(row)
                sample['content'] = rows
                # Count total rows
                with open(path, 'r', encoding='utf-8') as f:
                    sample['total_items'] = sum(1 for _ in f) - 1  # Subtract header
            
            elif path.suffix in ['.md', '.txt']:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    sample['content'] = ''.join(lines[:100])  # First 100 lines
                    sample['total_lines'] = len(lines)
            
            elif path.suffix == '.pdf':
                sample['content'] = "PDF file (binary content)"
            
            elif path.suffix in ['.png', '.jpg', '.jpeg']:
                sample['content'] = "Image file (binary content)"
            
            else:
                # Try to read as text
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # First 1000 chars
                    sample['content'] = content
        
        except Exception as e:
            sample['error'] = str(e)
        
        return sample
    
    def get_random_sample(self, stage: str, count: int = 1) -> List[Dict[str, Any]]:
        """Get random samples from a stage"""
        info = self.get_stage_info(stage)
        
        if not info['files']:
            return [{'error': f"No files found for stage: {stage}"}]
        
        # Select random files
        sample_files = random.sample(info['files'], min(count, len(info['files'])))
        
        samples = []
        for file_path in sample_files:
            samples.append(self.sample_file(file_path))
        
        return samples


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def display_stage_info(info: Dict[str, Any]):
    """Display stage information in a formatted table"""
    table = Table(title=f"Stage: {info['stage']}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Description", info['description'])
    table.add_row("Path", info['path'])
    table.add_row("Exists", "✓" if info['exists'] else "✗")
    table.add_row("File Count", str(info['count']))
    table.add_row("Total Size", format_size(info['total_size']))
    
    console.print(table)
    
    if info['count'] > 0 and info['count'] <= 10:
        console.print("\n[bold]Files:[/bold]")
        for f in info['files']:
            console.print(f"  • {f}")


def display_sample(sample: Dict[str, Any]):
    """Display a file sample"""
    console.print(f"\n[bold cyan]File:[/bold cyan] {sample['file']}")
    console.print(f"[bold cyan]Size:[/bold cyan] {format_size(sample['size'])}")
    
    if 'error' in sample:
        console.print(f"[red]Error:[/red] {sample['error']}")
        return
    
    if 'total_items' in sample:
        console.print(f"[bold cyan]Total Items:[/bold cyan] {sample['total_items']}")
    elif 'total_lines' in sample:
        console.print(f"[bold cyan]Total Lines:[/bold cyan] {sample['total_lines']}")
    
    if sample['content']:
        console.print("\n[bold]Sample Content:[/bold]")
        
        if isinstance(sample['content'], str):
            # Display as syntax-highlighted text
            syntax = Syntax(sample['content'][:1000], "markdown" if sample['type'] == '.md' else "text", 
                          theme="monokai", line_numbers=True)
            console.print(syntax)
        
        elif isinstance(sample['content'], list):
            # Display list items
            for i, item in enumerate(sample['content'][:3]):  # Show first 3 items
                console.print(f"\n[bold]Item {i+1}:[/bold]")
                if isinstance(item, dict):
                    # Pretty print JSON
                    json_str = json.dumps(item, indent=2, ensure_ascii=False)[:500]
                    syntax = Syntax(json_str, "json", theme="monokai")
                    console.print(syntax)
                else:
                    console.print(str(item)[:500])
        
        elif isinstance(sample['content'], dict):
            # Pretty print JSON
            json_str = json.dumps(sample['content'], indent=2, ensure_ascii=False)[:1000]
            syntax = Syntax(json_str, "json", theme="monokai")
            console.print(syntax)


@click.group(name='inspect')
@click.pass_context
def inspect_command(ctx):
    """Inspect data at various pipeline stages"""
    pass


@inspect_command.command(name='list')
@click.pass_context
def list_stages(ctx):
    """List all available inspection stages"""
    inspector = DataInspector(ctx.obj['config'])
    
    table = Table(title="Available Pipeline Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Path", style="yellow")
    table.add_column("Status", style="magenta")
    
    for stage_name, stage_config in inspector.stages.items():
        path = Path(stage_config['path'])
        status = "✓ Exists" if path.exists() else "✗ Not found"
        table.add_row(stage_name, stage_config['description'], str(path), status)
    
    console.print(table)


@inspect_command.command(name='stage')
@click.argument('stage_name')
@click.option('--sample', '-s', default=0, help='Number of random files to sample')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed sample content')
@click.pass_context
def inspect_stage(ctx, stage_name, sample, detailed):
    """Inspect a specific pipeline stage"""
    inspector = DataInspector(ctx.obj['config'])
    
    try:
        # Get stage information
        info = inspector.get_stage_info(stage_name)
        display_stage_info(info)
        
        # Sample files if requested
        if sample > 0 and info['count'] > 0:
            console.print(f"\n[bold]Random Samples ({sample}):[/bold]")
            samples = inspector.get_random_sample(stage_name, sample)
            
            for sample_data in samples:
                if detailed:
                    display_sample(sample_data)
                else:
                    # Just show file names
                    console.print(f"  • {sample_data['file']} ({format_size(sample_data['size'])})")
    
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        ctx.exit(1)


@inspect_command.command(name='file')
@click.argument('file_path')
@click.option('--lines', '-l', default=50, help='Number of lines to display (text files)')
@click.pass_context
def inspect_file(ctx, file_path, lines):
    """Inspect a specific file"""
    inspector = DataInspector(ctx.obj['config'])
    
    sample = inspector.sample_file(file_path, sample_size=lines)
    display_sample(sample)


@inspect_command.command(name='all')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
@click.pass_context
def inspect_all(ctx, verbose):
    """Show overview of all pipeline stages"""
    inspector = DataInspector(ctx.obj['config'])
    
    console.print("\n[bold cyan]Pipeline Data Overview[/bold cyan]\n")
    
    total_files = 0
    total_size = 0
    
    for stage_name in inspector.stages:
        info = inspector.get_stage_info(stage_name)
        total_files += info['count']
        total_size += info['total_size']
        
        if verbose:
            display_stage_info(info)
            console.print()
        else:
            status = "✓" if info['exists'] and info['count'] > 0 else "✗"
            console.print(f"{status} [bold]{stage_name:12}[/bold] {info['count']:6} files, {format_size(info['total_size']):>10}")
    
    console.print(f"\n[bold]Total:[/bold] {total_files} files, {format_size(total_size)}")


@inspect_command.command(name='stats')
@click.pass_context
def show_stats(ctx):
    """Show statistics for all stages"""
    inspector = DataInspector(ctx.obj['config'])
    
    stats = {}
    for stage_name in inspector.stages:
        info = inspector.get_stage_info(stage_name)
        stats[stage_name] = {
            'exists': info['exists'],
            'count': info['count'],
            'size': info['total_size']
        }
    
    # Create summary table
    table = Table(title="Pipeline Statistics")
    table.add_column("Stage", style="cyan")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("Avg Size", justify="right", style="magenta")
    
    for stage_name, stage_stats in stats.items():
        if stage_stats['count'] > 0:
            avg_size = stage_stats['size'] / stage_stats['count']
        else:
            avg_size = 0
        
        table.add_row(
            stage_name,
            str(stage_stats['count']),
            format_size(stage_stats['size']),
            format_size(int(avg_size))
        )
    
    console.print(table)
    
    # Show processing status if log files exist
    log_files = [
        (Path(ctx.obj['config'].paths.logs_dir) / 'processing_log.csv', 'Text QA Generation'),
        (Path(ctx.obj['config'].paths.logs_dir) / 'vision_processing_log.csv', 'Vision QA Generation'),
    ]
    
    for log_path, description in log_files:
        if log_path.exists():
            console.print(f"\n[bold]{description} Log:[/bold]")
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    console.print(f"  Last processed: {last_row.get('paper_id', 'N/A')}")
                    console.print(f"  Total processed: {len(rows)}")


if __name__ == '__main__':
    # For testing
    from src.config_manager import config
    config.load()
    inspector = DataInspector(config)
    
    # Test stage inspection
    info = inspector.get_stage_info('ocr')
    display_stage_info(info)