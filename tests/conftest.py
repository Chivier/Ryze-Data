"""
Pytest configuration and shared fixtures for all tests
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config_manager import ConfigManager
from src.pipeline_manager import PipelineManager
from src.cli.data_inspector import DataInspector


# ============== Configuration Fixtures ==============

@pytest.fixture
def test_env_vars():
    """Set up test environment variables"""
    original_env = os.environ.copy()
    
    # Load test environment
    test_env_file = Path(__file__).parent / '.env.test'
    if test_env_file.exists():
        with open(test_env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_config(test_env_vars) -> ConfigManager:
    """Create a test configuration"""
    config = ConfigManager()
    config_path = Path(__file__).parent / 'config.test.json'
    config.load(str(config_path))
    return config


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


# ============== Sample Data Fixtures ==============

@pytest.fixture
def sample_article_metadata() -> Dict[str, Any]:
    """Sample Nature article metadata"""
    return {
        'title': 'Test Article: Understanding Neural Networks',
        'url': 'https://www.nature.com/articles/nature04244',
        'abstract': 'This is a test abstract describing neural network research.',
        'open_access': 'Y',
        'date': '2024-01-15',
        'author': 'John Doe, Jane Smith'
    }


@pytest.fixture
def sample_ocr_result(temp_dir) -> Path:
    """Create sample OCR result files"""
    ocr_dir = temp_dir / 'ocr_results' / 'nature04244'
    ocr_dir.mkdir(parents=True)
    
    # Create sample markdown file
    md_file = ocr_dir / 'nature04244.md'
    md_content = """# Test Article: Understanding Neural Networks

## Abstract
This is a test abstract describing neural network research.

## Introduction
Neural networks have revolutionized machine learning...

## Methods
We used a convolutional neural network architecture...

## Results
Our model achieved 95% accuracy on the test set...

<span id="fig1"></span>
![](page_1_Figure_1.jpeg)
**Figure 1**: Architecture of the proposed neural network.

## Discussion
The results demonstrate the effectiveness of our approach...

## References
1. Smith, J. et al. (2023). Previous work on neural networks.
2. Doe, J. (2022). Foundation of deep learning.
"""
    md_file.write_text(md_content)
    
    # Create sample metadata file
    meta_file = ocr_dir / 'nature04244_meta.json'
    meta_data = {
        'paper_id': 'nature04244',
        'title': 'Test Article: Understanding Neural Networks',
        'pages': 10,
        'figures': 2,
        'tables': 1,
        'processing_time': 45.2
    }
    meta_file.write_text(json.dumps(meta_data, indent=2))
    
    # Create a dummy figure file
    figure_file = ocr_dir / 'page_1_Figure_1.jpeg'
    figure_file.write_bytes(b'\x00' * 100)  # Dummy image data
    
    return ocr_dir


@pytest.fixture
def sample_figure_data() -> Dict[str, Any]:
    """Sample figure extraction data"""
    return {
        'figure_path': '/test/data/page_1_Figure_1.jpeg',
        'related_info': [
            {
                'position': 'before_figure',
                'info': 'Figure 1 shows the architecture of our neural network model.'
            },
            {
                'position': 'after_figure',
                'info': 'The network consists of multiple convolutional layers.'
            }
        ]
    }


@pytest.fixture
def sample_qa_pair() -> Dict[str, Any]:
    """Sample QA pair"""
    return {
        'question': 'What is the main contribution of this paper?',
        'answer': 'The paper presents a novel neural network architecture that achieves state-of-the-art results.',
        'paper_id': 'nature04244',
        'section': 'introduction',
        'difficulty': 'medium',
        'question_type': 'factual',
        'quality_score': 4.2
    }


@pytest.fixture
def sample_vision_qa_pair() -> Dict[str, Any]:
    """Sample vision QA pair"""
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'What does Figure 1 show? <image>'
            },
            {
                'role': 'assistant',
                'content': 'Figure 1 shows the architecture of a convolutional neural network with multiple layers.'
            }
        ],
        'images': ['/test/data/page_1_Figure_1.jpeg']
    }


# ============== Mock Fixtures ==============

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for API tests"""
    with patch('openai.OpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        
        # Mock completion response
        completion = MagicMock()
        completion.choices = [MagicMock(message=MagicMock(content='Test response'))]
        client.chat.completions.create.return_value = completion
        
        yield client


@pytest.fixture
def mock_requests():
    """Mock requests for web scraping tests"""
    with patch('requests.get') as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.text = '''
        <html>
        <div class="c-card__body u-display-flex u-flex-direction-column">
            <h3 class="c-card__title">
                <a href="/articles/nature04244">Test Article</a>
            </h3>
            <div data-test="article-description">
                <p>Test abstract</p>
            </div>
        </div>
        </html>
        '''
        mock_get.return_value = response
        yield mock_get


# ============== Test Data Setup Fixtures ==============

@pytest.fixture
def setup_test_data(temp_dir, sample_ocr_result):
    """Set up complete test data structure"""
    # Create directory structure
    dirs = [
        'nature_metadata',
        'pdfs',
        'abstracts',
        'figures',
        'vlm_preprocessing',
        'sft_data',
        'vlm_sft_data',
        'logs'
    ]
    
    for dir_name in dirs:
        (temp_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create sample metadata CSV
    csv_file = temp_dir / 'nature_metadata' / 'all_articles.csv'
    csv_content = """title,url,abstract,open_access,date,author
"Test Article","https://nature.com/articles/nature04244","Test abstract","Y","2024-01-15","John Doe"
"""
    csv_file.write_text(csv_content)
    
    # Create sample PDF (dummy file)
    pdf_file = temp_dir / 'pdfs' / 'nature04244.pdf'
    pdf_file.write_bytes(b'%PDF-1.4\n%Test PDF content')
    
    # Create sample abstract
    abstract_file = temp_dir / 'abstracts' / 'nature04244_abstract.txt'
    abstract_file.write_text('This is a test abstract for the neural network paper.')
    
    # Create sample figure JSON
    figure_json = temp_dir / 'vlm_preprocessing' / 'nature04244.json'
    figure_data = [
        {
            'figure_path': str(sample_ocr_result / 'page_1_Figure_1.jpeg'),
            'related_info': [
                {'position': 'before_figure', 'info': 'Figure 1 description'}
            ]
        }
    ]
    figure_json.write_text(json.dumps(figure_data, indent=2))
    
    # Create sample QA data
    qa_file = temp_dir / 'sft_data' / 'nature04244_qa.jsonl'
    qa_data = {
        'question': 'What is the paper about?',
        'answer': 'Neural networks',
        'paper_id': 'nature04244'
    }
    qa_file.write_text(json.dumps(qa_data) + '\n')
    
    return temp_dir


# ============== Pipeline Fixtures ==============

@pytest.fixture
def test_pipeline(test_config):
    """Create a test pipeline manager"""
    return PipelineManager(test_config)


@pytest.fixture
def test_inspector(test_config):
    """Create a test data inspector"""
    return DataInspector(test_config)


# ============== Utility Fixtures ==============

@pytest.fixture
def capture_logs():
    """Capture log messages during tests"""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    logger = logging.getLogger()
    logger.addHandler(handler)
    
    yield log_capture
    
    # Clean up
    logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests"""
    from src.config_manager import ConfigManager
    ConfigManager._instance = None
    yield


# ============== Markers ==============

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_api: Tests requiring API access")
    config.addinivalue_line("markers", "sample: Tests using sample data")