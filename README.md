# Ryze-Data

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-green.svg)](docs/)

A comprehensive data processing pipeline framework for scientific literature, providing end-to-end workflow from web scraping to training data generation.

[中文文档](docs/zh-CN/README.md) | [English Documentation](docs/)

## 🌟 Overview

Ryze-Data is an enterprise-grade, modular framework designed to automate the complex process of extracting, processing, and transforming scientific literature into high-quality training datasets for machine learning models. Built with scalability, reliability, and extensibility at its core, it streamlines the entire data pipeline from source to structured output.

### Key Features

- **📚 Intelligent Web Scraping**: Automated collection of scientific articles from Nature and other sources
- **📄 Advanced PDF Processing**: Parallel downloading with fault tolerance and retry mechanisms
- **🔍 State-of-the-art OCR**: High-accuracy text and figure extraction using marker engine
- **🖼️ Context-aware Figure Analysis**: Intelligent extraction of figures with surrounding context
- **🤖 Multi-modal QA Generation**: Automated generation of both text and vision question-answer pairs
- **🔧 Flexible Configuration**: Environment-based configuration with hot-reload support
- **📊 Real-time Monitoring**: Built-in metrics and logging for pipeline observability
- **🚀 Production Ready**: Distributed processing support with checkpoint recovery

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for accelerated OCR)
- 16GB+ RAM recommended
- 100GB+ free disk space for data storage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ryze-data.git
cd ryze-data

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to set your API keys and paths
nano .env
```

### Basic Usage

```bash
# Run the complete pipeline
python -m src.cli.main pipeline

# Or run individual stages
python -m src.cli.main scrape      # Scrape article metadata
python -m src.cli.main download    # Download PDF files
python -m src.cli.main ocr         # Extract text and images
python -m src.cli.main extract     # Extract figures with context
python -m src.cli.main generate-qa # Generate QA pairs
```

### Data Inspection

```bash
# View pipeline status
python -m src.cli.main inspect all

# Inspect specific stage with samples
python -m src.cli.main inspect stage ocr --sample 5 --detailed

# Check configuration
python -m src.cli.main config-show

# View processing statistics
python -m src.cli.main inspect stats
```

## 📂 Project Structure

```
Ryze-Data/
├── src/                    # Source code
│   ├── cli/               # Command-line interface
│   ├── scrapers/          # Web scraping modules
│   ├── downloaders/       # PDF download managers
│   ├── processors/        # Data processing engines
│   ├── generators/        # QA generation modules
│   ├── config_manager.py  # Configuration management
│   └── pipeline_manager.py # Pipeline orchestration
├── prompts/               # LLM prompt templates
│   ├── text/             # Text QA prompts
│   └── vision/           # Vision QA prompts
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                  # Documentation
│   ├── architecture.md   # System architecture
│   ├── configuration.md  # Configuration guide
│   ├── api-reference.md  # API documentation
│   └── zh-CN/           # Chinese documentation
├── scripts/              # Utility scripts
├── data-sample/          # Sample data for testing
├── .env.example          # Environment template
├── config.example.json   # Configuration template
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

Ryze-Data uses a multi-layer configuration system:

1. **Default values** (in code)
2. **Configuration file** (config.json)
3. **Environment variables** (.env)
4. **Command-line arguments**

### Quick Configuration

```bash
# Essential environment variables
OPENAI_API_KEY=sk-...           # For QA generation
RYZE_DATA_ROOT=./data           # Data storage location
RYZE_NUM_WORKERS=4              # Parallel processing threads
RYZE_GPU_ENABLED=true           # Enable GPU acceleration
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## 📊 Pipeline Architecture

The system follows a modular, stage-based architecture:

```mermaid
graph LR
    A[Web Sources] -->|Scraping| B[Metadata]
    B -->|Download| C[PDF Files]
    C -->|OCR| D[Text + Images]
    D -->|Process| E[Structured Data]
    E -->|Generate| F[QA Datasets]
```

Each stage is:
- **Independent**: Can be run separately
- **Resumable**: Supports checkpoint recovery
- **Scalable**: Supports distributed processing
- **Observable**: Provides metrics and logging

See [Architecture Documentation](docs/architecture.md) for details.

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Quick smoke test
python run_tests.py quick

# Test with sample data
python run_tests.py sample

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/
```

## 📚 Documentation

### English Documentation
- [Architecture Design](docs/architecture.md) - System architecture and design decisions
- [Configuration Guide](docs/configuration.md) - Detailed configuration options
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Data Formats](docs/data-formats.md) - Data structure specifications
- [Development Guide](docs/development.md) - Contributing and extending
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### 中文文档
- [架构设计](docs/zh-CN/architecture.md) - 系统架构与设计决策
- [配置指南](docs/zh-CN/configuration.md) - 详细配置选项
- [API参考](docs/zh-CN/api-reference.md) - 完整API文档
- [数据格式](docs/zh-CN/data-formats.md) - 数据结构规范
- [开发指南](docs/zh-CN/development.md) - 贡献与扩展
- [故障排查](docs/zh-CN/troubleshooting.md) - 常见问题与解决方案

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/
pylint src/
```

## 📈 Performance

Typical processing metrics on standard hardware:

| Stage | Documents/Hour | GPU Speedup |
|-------|---------------|-------------|
| Scraping | 1000+ | N/A |
| Download | 200-500 | N/A |
| OCR | 50-100 | 2-3x |
| QA Generation | 100-200 | N/A |

## 🔒 Security

- API keys are stored in environment variables
- Supports credential rotation
- No sensitive data in logs
- Configurable data retention policies

## 📝 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Marker](https://github.com/VikParuchuri/marker) - OCR engine
- [OpenAI](https://openai.com) - LLM APIs
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - Web scraping
- All contributors and users of this project

## 📧 Support

For issues, questions, or contributions:
- Open an [Issue](https://github.com/your-username/ryze-data/issues)
- Check [Troubleshooting Guide](docs/troubleshooting.md)
- Review [FAQ](docs/faq.md)

---

<p align="center">
  Made with ❤️ by the Ryze-Data Team
</p>