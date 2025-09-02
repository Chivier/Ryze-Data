# Ryze-Data Test Suite

## Overview

This test suite provides comprehensive testing for the Ryze-Data pipeline, including unit tests, integration tests, and end-to-end tests with sample data.

## Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_config_manager.py
│   ├── test_data_inspector.py
│   └── test_pipeline_manager.py
├── integration/          # Integration tests
│   ├── test_full_pipeline.py
│   └── test_scraper.py
├── fixtures/            # Test fixtures and mock data
├── data/               # Test data (created during tests)
├── conftest.py         # Pytest configuration and shared fixtures
├── .env.test           # Test environment configuration
└── config.test.json    # Test configuration file
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py all

# Run quick tests (no slow/API tests)
python run_tests.py quick

# Run unit tests only
python run_tests.py unit

# Run integration tests
python run_tests.py integration

# Run tests with sample data
python run_tests.py sample

# Run pipeline test with single article
python run_tests.py pipeline

# Clean test artifacts
python run_tests.py clean
```

### Using Pytest Directly

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_config_manager.py

# Run tests with specific marker
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m sample        # Tests using sample data
pytest -m "not slow"    # Skip slow tests

# Run with verbose output
pytest -vv

# Run specific test function
pytest tests/unit/test_config_manager.py::TestConfigManager::test_singleton_pattern
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- Located in `tests/unit/`

### Integration Tests
- Test component interactions
- Test data flow between stages
- May use real sample data
- Located in `tests/integration/`

### Sample Data Tests
- Use the biocot sample data (nature04244 article)
- Test real processing scenarios
- Marked with `@pytest.mark.sample`

## Test Configuration

### Environment Variables (.env.test)
```bash
# Small scale configuration for tests
RYZE_MAX_PAPERS=1
RYZE_QA_RATIO=2
RYZE_NUM_WORKERS=1
RYZE_BATCH_SIZE=2
RYZE_GPU_ENABLED=false
```

### Test Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests taking > 5 seconds
- `@pytest.mark.sample` - Tests using sample data
- `@pytest.mark.requires_api` - Tests requiring API access
- `@pytest.mark.mock` - Tests using mocked services

## Testing Individual Components

### Testing Config Manager
```python
# Test environment variable expansion
def test_environment_variable_expansion():
    os.environ['TEST_VAR'] = 'test_value'
    config = ConfigManager()
    config.load()
    assert config.get('test.var') == 'test_value'
```

### Testing Data Inspector
```python
# Test inspecting a specific stage
def test_inspect_stage():
    inspector = DataInspector(config)
    info = inspector.get_stage_info('ocr')
    assert info['exists'] == True
    assert info['count'] > 0
```

### Testing Pipeline Manager
```python
# Test running specific stages
def test_run_pipeline_stages():
    pipeline = PipelineManager(config)
    result = pipeline.run(stages=['scrape', 'download'])
    assert result.completed_stages == 2
```

## Testing with Sample Data

The test suite includes sample data from a real Nature article (nature04244):

1. **OCR Result**: Markdown text and extracted figures
2. **Abstract**: Paper abstract text
3. **QA Pairs**: Sample question-answer pairs
4. **Figures**: Extracted figure images with metadata

### Using Sample Data in Tests
```python
@pytest.mark.sample
def test_with_sample_data(sample_ocr_result):
    # sample_ocr_result fixture provides path to OCR data
    assert sample_ocr_result.exists()
    
    md_file = sample_ocr_result / 'nature04244.md'
    assert 'Neural Networks' in md_file.read_text()
```

## Mock Services

### Mocking OpenAI API
```python
@pytest.fixture
def mock_openai_client():
    with patch('openai.OpenAI') as mock:
        client = MagicMock()
        completion = MagicMock()
        completion.choices = [MagicMock(
            message=MagicMock(content='Test response')
        )]
        client.chat.completions.create.return_value = completion
        yield client
```

### Mocking Web Requests
```python
@pytest.fixture
def mock_requests():
    with patch('requests.get') as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.text = '<html>Mock content</html>'
        mock_get.return_value = response
        yield mock_get
```

## Coverage Reports

After running tests with coverage:

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Continuous Integration

The test suite is designed to work in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    python run_tests.py quick
    
- name: Run full test suite
  run: |
    python run_tests.py all --verbose
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure project root is in PYTHONPATH
   ```bash
   export PYTHONPATH=/path/to/Ryze-Data:$PYTHONPATH
   ```

2. **Missing dependencies**: Install test requirements
   ```bash
   pip install pytest pytest-cov pytest-mock
   ```

3. **Test data not found**: Run setup
   ```bash
   python run_tests.py sample  # Sets up sample data automatically
   ```

4. **Clean test artifacts**: 
   ```bash
   python run_tests.py clean
   ```

## Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.unit`, etc.)
2. Add fixtures to `conftest.py` for reusable test data
3. Follow naming convention: `test_*.py` for test files
4. Document complex test scenarios with docstrings
5. Keep tests fast and independent

## Performance

Target test execution times:
- Unit tests: < 1 second each
- Integration tests: < 5 seconds each
- Full test suite: < 1 minute

Use `@pytest.mark.slow` for tests exceeding 5 seconds.