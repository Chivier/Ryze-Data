# Development Guide

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Code Style and Standards](#code-style-and-standards)
- [Adding New Features](#adding-new-features)
- [Testing Guidelines](#testing-guidelines)
- [Debugging Techniques](#debugging-techniques)
- [Performance Profiling](#performance-profiling)
- [Contributing Workflow](#contributing-workflow)
- [Release Process](#release-process)

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)
- IDE with Python support (VSCode, PyCharm recommended)

### Initial Setup

1. **Fork and Clone**
```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR_USERNAME/ryze-data.git
cd ryze-data
git remote add upstream https://github.com/original/ryze-data.git
```

2. **Create Development Environment**
```bash
# Install dependencies with uv
uv sync --all-extras

# Or create a manual venv
python -m venv venv-dev
source venv-dev/bin/activate  # On Windows: venv-dev\Scripts\activate
pip install -e ".[benchmark]"
```

3. **Configure Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

4. **Set Up IDE**

**VSCode settings.json:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestArgs": ["tests"],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true
}
```

**PyCharm Configuration:**
- Set Python interpreter to virtual environment
- Enable code inspections
- Configure pytest as test runner
- Set Black as code formatter

## Project Architecture

### Directory Structure

```
src/
├── cli/                     # Command-line interface
│   ├── main.py             # ✅ CLI entry point
│   └── data_inspector.py   # ✅ Data inspection utilities
├── ocr/                     # ✅ Extensible OCR module (6 models)
│   ├── base_ocr.py          # Abstract base + OCRResult
│   ├── registry.py          # OCRRegistry (decorator-based discovery)
│   ├── marker_ocr.py        # Marker (CLI wrapper)
│   ├── deepseek_ocr.py      # DeepSeek-OCR v1
│   ├── deepseek_ocr_v2.py   # DeepSeek-OCR v2
│   ├── markitdown_ocr.py    # MarkItDown
│   ├── paddle_ocr.py        # PaddleOCR (PP-OCRv5)
│   └── glm_ocr.py           # GLM-OCR (vLLM / Z.AI)
├── benchmark/               # ✅ OCR benchmark evaluation
│   ├── datasets/            # ArxivQA, SlideVQA loaders
│   ├── evaluator.py         # Main evaluator orchestrator
│   ├── qa_client.py         # QwenQAClient (vision + text)
│   └── metrics.py           # Accuracy, EM, BLEU, ROUGE
├── generators/              # ✅ QA generators (text + vision)
├── scrapers/                # ✅ Web scraping (Nature)
├── config_manager.py        # ✅ Configuration management
├── pipeline_manager.py      # ⚠️ Pipeline framework (partial)
├── api_key_balancer.py      # ✅ API key load balancing
└── chunked-ocr.py           # ✅ Legacy chunked OCR
```

**Implementation Status:**
- ✅ Fully implemented
- ⚠️ Framework implemented, functionality partial

### Design Patterns

#### 1. **Abstract Factory Pattern**

Used for creating different types of processors and generators:

```python
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """Abstract base class for all processors"""
    
    @abstractmethod
    def process(self, input_data):
        """Process input data"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data):
        """Validate input before processing"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data):
        """Validate output after processing"""
        pass

class OCRProcessor(BaseProcessor):
    """Concrete implementation for OCR processing"""
    
    def process(self, input_data):
        # Implementation
        pass
```

#### 2. **Singleton Pattern**

Used for configuration management:

```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
```

#### 3. **Strategy Pattern**

Used for different QA generation strategies:

```python
class QAGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, context, config):
        pass

class FactualQAStrategy(QAGenerationStrategy):
    def generate(self, context, config):
        # Generate factual QA pairs
        pass

class ConceptualQAStrategy(QAGenerationStrategy):
    def generate(self, context, config):
        # Generate conceptual QA pairs
        pass
```

## Code Style and Standards

### Python Style Guide

We follow PEP 8 with the following additions:

1. **Line Length**: Maximum 88 characters (Black default)
2. **Import Order**:
   ```python
   # Standard library
   import os
   import sys
   
   # Third-party
   import numpy as np
   import pandas as pd
   
   # Local imports
   from src.config_manager import ConfigManager
   from src.utils import logger
   ```

3. **Docstrings**: Google style
   ```python
   def process_data(input_file: str, output_dir: str, batch_size: int = 10) -> dict:
       """Process data from input file and save to output directory.
       
       Args:
           input_file: Path to input file
           output_dir: Directory for output files
           batch_size: Number of items to process at once
       
       Returns:
           Dictionary containing processing results
       
       Raises:
           FileNotFoundError: If input file doesn't exist
           ValueError: If batch_size is less than 1
       """
       pass
   ```

4. **Type Hints**: Required for all public functions
   ```python
   from typing import List, Dict, Optional, Union
   
   def parse_config(config_path: str) -> Dict[str, Any]:
       pass
   ```

### Code Quality Tools

```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Static type checking
mypy src/

# Security checks
bandit -r src/

# Complexity analysis
radon cc src/ -s
```

## Adding New Features

### 1. Adding a New Scraper

Create a new scraper by extending `BaseScraper`:

```python
# src/scrapers/arxiv_scraper.py
from typing import List, Dict, Any
from src.scrapers.base_scraper import BaseScraper

class ArxivScraper(BaseScraper):
    """Scraper for arXiv papers"""
    
    def __init__(self, output_dir: str, config: dict = None):
        super().__init__(output_dir, config)
        self.base_url = "https://arxiv.org"
    
    def scrape(self, query: str = None) -> List[Dict[str, Any]]:
        """Scrape arXiv papers based on query"""
        # Implementation
        pass
    
    def parse_paper(self, paper_html: str) -> Dict[str, Any]:
        """Parse individual paper metadata"""
        # Implementation
        pass
```

Register the scraper in the pipeline:

```python
# src/pipeline_manager.py
from src.scrapers.arxiv_scraper import ArxivScraper

# In PipelineManager.__init__
self.scrapers['arxiv'] = ArxivScraper
```

### 2. Adding a New Processor

```python
# src/processors/table_extractor.py
from src.processors.base_processor import BaseProcessor

class TableExtractor(BaseProcessor):
    """Extract tables from documents"""
    
    def process(self, input_data):
        """Extract tables from input data"""
        tables = []
        # Extraction logic
        return tables
    
    def validate_input(self, input_data):
        """Ensure input is valid document data"""
        # Validation logic
        return True
    
    def validate_output(self, output_data):
        """Ensure extracted tables are valid"""
        # Validation logic
        return True
```

### 3. Adding a New CLI Command

```python
# src/cli/main.py
@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
@click.pass_context
def export(ctx, format):
    """Export processed data in specified format"""
    cfg = ctx.obj['config']
    
    if format == 'json':
        export_json(cfg)
    elif format == 'csv':
        export_csv(cfg)
    
    click.echo(f"Export completed in {format} format")
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_config_manager.py
│   ├── test_scrapers.py
│   └── test_processors.py
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   └── test_end_to_end.py
├── fixtures/                # Test fixtures
│   ├── sample_data.py
│   └── mock_responses.py
└── conftest.py             # Pytest configuration
```

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_processors.py
import pytest
from unittest.mock import Mock, patch
from src.processors.ocr_processor import OCRProcessor

class TestOCRProcessor:
    """Test suite for OCR processor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return OCRProcessor(config={'batch_size': 10})
    
    def test_process_valid_pdf(self, processor, tmp_path):
        """Test processing valid PDF file"""
        # Arrange
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"PDF content")
        
        # Act
        result = processor.process(str(pdf_file))
        
        # Assert
        assert result is not None
        assert 'text' in result
        assert 'images' in result
    
    def test_process_invalid_file(self, processor):
        """Test handling of invalid file"""
        with pytest.raises(FileNotFoundError):
            processor.process("nonexistent.pdf")
    
    @patch('src.processors.ocr_processor.marker')
    def test_ocr_error_handling(self, mock_marker, processor):
        """Test OCR error handling"""
        mock_marker.convert.side_effect = Exception("OCR failed")
        
        with pytest.raises(ProcessingError):
            processor.process("test.pdf")
```

#### Integration Test Example

```python
# tests/integration/test_pipeline.py
import pytest
from src.pipeline_manager import PipelineManager
from src.config_manager import ConfigManager

@pytest.mark.integration
class TestPipeline:
    """Integration tests for pipeline"""
    
    def test_full_pipeline(self, sample_data):
        """Test complete pipeline execution"""
        # Setup
        config = ConfigManager()
        config.load("tests/config.test.json")
        pipeline = PipelineManager(config)
        
        # Run pipeline (currently implemented stages: scrape, ocr)
        result = pipeline.run(stages=['scrape', 'ocr'])

        # Verify
        assert result.completed_stages == 2
        assert result.failed_stages == 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_config_manager.py

# Run tests matching pattern
pytest -k "test_ocr"

# Run with verbose output
pytest -v

# Run integration tests only
pytest -m integration

# Run with parallel execution
pytest -n auto
```

## Debugging Techniques

### 1. Using Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint() in Python 3.7+
breakpoint()

# IPython debugger (better interface)
import ipdb; ipdb.set_trace()
```

### 2. Logging for Debugging

```python
import logging

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"Processing data: {data[:100]}...")  # Log first 100 chars
    
    try:
        result = transform(data)
        logger.debug(f"Transform successful: {len(result)} items")
    except Exception as e:
        logger.error(f"Transform failed: {e}", exc_info=True)
        raise
    
    return result
```

### 3. Remote Debugging with VSCode

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Pipeline",
            "type": "python",
            "request": "launch",
            "module": "src.cli.main",
            "args": ["pipeline", "--stages", "ocr"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

## Performance Profiling

### 1. CPU Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile code block
profiler = cProfile.Profile()
profiler.enable()

# Your code here
process_large_dataset()

profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(10)  # Top 10 functions

# Save profile data
stats.dump_stats('profile_data.prof')
```

### 2. Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function that uses a lot of memory
    large_list = [i for i in range(1000000)]
    return large_list

# Run with: python -m memory_profiler script.py
```

### 3. Line Profiling

```python
# Install: pip install line_profiler

from line_profiler import LineProfiler

def slow_function():
    result = []
    for i in range(1000):
        result.append(i ** 2)
    return result

lp = LineProfiler()
lp_wrapper = lp(slow_function)
lp_wrapper()
lp.print_stats()
```

## Contributing Workflow

### 1. Branch Strategy

```bash
# Create feature branch
git checkout -b feature/new-scraper

# Create bugfix branch
git checkout -b bugfix/ocr-timeout

# Create hotfix branch
git checkout -b hotfix/critical-issue
```

### 2. Commit Guidelines

Follow conventional commits:

```bash
# Feature
git commit -m "feat: add arXiv scraper support"

# Bug fix
git commit -m "fix: resolve OCR timeout on large PDFs"

# Documentation
git commit -m "docs: update API reference for new endpoints"

# Performance
git commit -m "perf: optimize batch processing in QA generator"

# Refactoring
git commit -m "refactor: simplify config manager implementation"

# Tests
git commit -m "test: add integration tests for pipeline"

# Chore
git commit -m "chore: update dependencies"
```

### 3. Pull Request Process

1. **Update your fork**
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

2. **Create feature branch**
```bash
git checkout -b feature/your-feature
```

3. **Make changes and test**
```bash
# Make your changes
# Run tests
pytest
# Run linting
flake8 src/
black src/
```

4. **Commit and push**
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

5. **Create Pull Request**
- Go to GitHub and create PR
- Fill out PR template
- Link related issues
- Request reviews

### 4. Code Review Guidelines

**For Reviewers:**
- Check code follows style guide
- Verify tests are included
- Ensure documentation is updated
- Test locally if possible
- Provide constructive feedback

**For Authors:**
- Respond to all comments
- Make requested changes
- Update PR description if scope changes
- Rebase if needed

## Release Process

### 1. Version Numbering

Follow Semantic Versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### 2. Release Checklist

```markdown
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in setup.py
- [ ] Release notes prepared
- [ ] Tagged in git
- [ ] Package built and tested
```

### 3. Release Commands

```bash
# Update version
bump2version patch  # or minor, major

# Create release tag
git tag -a v1.2.3 -m "Release version 1.2.3"

# Push tag
git push origin v1.2.3

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (if applicable)
twine upload dist/*
```

## Best Practices

### 1. Error Handling

```python
class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

def robust_process(data):
    """Process data with proper error handling"""
    try:
        # Validate input
        if not validate_data(data):
            raise ValueError("Invalid data format")
        
        # Process
        result = process_internal(data)
        
        # Validate output
        if not validate_result(result):
            raise ProcessingError("Processing produced invalid result")
        
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        # Attempt recovery or graceful degradation
        return process_fallback(data)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise
```

### 2. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessorConfig:
    """Configuration for processor"""
    batch_size: int = 10
    timeout: int = 300
    retry_count: int = 3
    gpu_enabled: bool = False
    
    def validate(self) -> bool:
        """Validate configuration"""
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.timeout < 0:
            raise ValueError("Timeout must be non-negative")
        return True
```

### 3. Resource Management

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(resource_path):
    """Context manager for resource handling"""
    resource = None
    try:
        resource = acquire_resource(resource_path)
        yield resource
    finally:
        if resource:
            release_resource(resource)

# Usage
with managed_resource("/path/to/resource") as resource:
    process_with_resource(resource)
```

## Troubleshooting Development Issues

### Common Issues

1. **Import Errors**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

2. **Dependency Conflicts**
```bash
# Use pip-tools for dependency management
pip install pip-tools
pip-compile requirements.in
pip-sync
```

3. **Test Failures**
```bash
# Clear test cache
pytest --cache-clear

# Run with more verbose output
pytest -vvs
```

4. **Memory Issues During Development**
```python
# Use memory profiling
import tracemalloc
tracemalloc.start()

# Your code
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Resources

### Documentation
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)

### Tools
- [Black - Code Formatter](https://black.readthedocs.io/)
- [Flake8 - Style Checker](https://flake8.pycqa.org/)
- [mypy - Type Checker](http://mypy-lang.org/)
- [pre-commit - Git Hooks](https://pre-commit.com/)

### Learning Resources
- [Real Python Tutorials](https://realpython.com/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Effective Python](https://effectivepython.com/)