# Ryze-Data

A comprehensive data processing framework for extracting and packaging structured content from PDF papers using OCR and advanced text processing techniques.

![Framework Architecture](./imgs/framework.png)

## Overview

Ryze-Data is designed to transform PDF documents into multiple structured formats, making academic and technical content more accessible and processable. The framework leverages OCR technology to extract text, images, and references from PDF papers, then packages this content into various formats suitable for different use cases.

## 🚀 Key Features

- **Multi-format PDF Processing**: Handles various PDF formats and layouts with high accuracy
- **Intelligent Content Segmentation**: Automatically identifies and separates abstracts, text chunks, images, and references
- **Flexible Output Formats**: Multiple output formats including clean text, structured JSON, and QA templates
- **Batch Processing**: Efficient processing of multiple documents with parallel execution
- **Quality Control**: Built-in quality metrics and error handling mechanisms
- **Scalable Architecture**: Supports distributed processing and horizontal scaling

## 🏗️ Architecture

The system consists of four main modules:

### Module 1: Data Processing (OCR Model)
- **Input**: PDF files and optional metadata
- **Process**: OCR extraction, content classification, and structured output generation
- **Output**: Markdown files, extracted images, metadata, and processing logs

### Module 2: Content Packers
- **Text Packer**: Cleans and formats text content
- **Image/Legend/Caption Packer**: Processes visual content with captions
- **Reference Packer**: Structures bibliographic references

### Module 3: QA Template Generation
- Generates multiple types of question-answer pairs
- Creates training data for downstream ML applications
- Supports factual, conceptual, visual, and reference-based questions

### Module 4: Integration and Quality Control
- Monitors processing quality and success rates
- Provides feedback loops for continuous improvement
- Handles errors gracefully with retry mechanisms

## 📋 Requirements

- Python 3.8+
- OCR engine (Tesseract or PaddleOCR)
- GPU support recommended for faster processing
- Minimum 4GB RAM, 8GB+ recommended for batch processing

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/ryze-data.git
cd ryze-data

# Install dependencies
chmod +x scripts/install.sh
./scripts/install.sh

# Or install manually
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t ryze-data .

# Run container
docker run -v /path/to/pdfs:/input -v /path/to/output:/output ryze-data
```

## 🎯 Usage

### Basic Usage

```python
from ryze_data import RyzeProcessor

# Initialize processor
processor = RyzeProcessor(
    ocr_model="paddleocr",  # or "tesseract"
    output_format="json",
    quality_threshold=0.85
)

# Process single PDF
result = processor.process_pdf("path/to/paper.pdf")

# Process batch of PDFs
results = processor.process_batch("path/to/pdf_folder/")
```

### Command Line Interface

```bash
# Process single PDF
python src/chunked-ocr.py --input paper.pdf --output ./output/

# Process multiple PDFs
python src/chunked-ocr.py --input ./papers/ --output ./output/ --batch

# With custom configuration
python src/chunked-ocr.py --input ./papers/ --output ./output/ --config config.json
```

### Configuration Example

```json
{
    "ocr_settings": {
        "model": "paddleocr",
        "language": "en",
        "confidence_threshold": 0.8
    },
    "output_settings": {
        "format": "json",
        "include_images": true,
        "generate_qa": true
    },
    "processing": {
        "batch_size": 10,
        "parallel_workers": 4,
        "max_retries": 3
    }
}
```

## 📊 Input/Output Examples

### Input Structure
```
input_folder/
├── paper1.pdf
├── paper2.pdf
├── paper3.pdf
└── metadata.json (optional)
```

### Output Structure
```
output_folder/
├── ocr_results/
│   ├── paper1/
│   │   ├── paper1.md
│   │   ├── paper1_meta.json
│   │   ├── paper1_abstract.txt
│   │   ├── paper1_references.json
│   │   └── images/
│   └── paper2/
├── text_packed/
│   ├── paper1/
│   │   ├── paper1_clean.txt
│   │   └── paper1_sections.json
│   └── paper2/
├── image_packed/
│   ├── paper1/
│   │   ├── images/
│   │   └── paper1_image_metadata.json
│   └── paper2/
├── reference_packed/
│   ├── paper1/
│   │   ├── paper1_references.json
│   │   └── paper1_bibliography.bib
│   └── paper2/
├── qa_templates/
│   ├── paper1/
│   │   ├── paper1_factual_qa.json
│   │   ├── paper1_conceptual_qa.json
│   │   └── paper1_visual_qa.json
│   └── paper2/
└── processing_log.csv
```

## 🔧 API Reference

### RyzeProcessor Class

```python
class RyzeProcessor:
    def __init__(self, ocr_model="paddleocr", output_format="json", quality_threshold=0.85):
        """Initialize the Ryze processor"""
        
    def process_pdf(self, pdf_path: str, output_path: str = None) -> dict:
        """Process a single PDF file"""
        
    def process_batch(self, input_folder: str, output_folder: str = None) -> list:
        """Process multiple PDF files in batch"""
        
    def get_processing_stats(self) -> dict:
        """Get processing statistics and quality metrics"""
```

### Text Packer

```python
from ryze_data.packers import TextPacker

packer = TextPacker()
result = packer.pack_text("ocr_results/paper1/")
```

### Image Packer

```python
from ryze_data.packers import ImagePacker

packer = ImagePacker()
result = packer.pack_images("ocr_results/paper1/")
```

### QA Generator

```python
from ryze_data.qa import QAGenerator

generator = QAGenerator()
qa_pairs = generator.generate_qa("text_packed/paper1/")
```

## 📈 Performance Optimization

### Recommended Settings

```python
# For high-quality papers
processor = RyzeProcessor(
    ocr_model="paddleocr",
    quality_threshold=0.9,
    batch_size=5,
    parallel_workers=2
)

# For batch processing
processor = RyzeProcessor(
    ocr_model="tesseract",
    quality_threshold=0.75,
    batch_size=20,
    parallel_workers=8
)
```

### GPU Acceleration

```python
# Enable GPU support
processor = RyzeProcessor(
    ocr_model="paddleocr",
    use_gpu=True,
    gpu_mem_limit=0.5
)
```

## 🧪 Use Cases

- **Academic Research**: Digitize and structure research papers for literature reviews
- **Knowledge Extraction**: Build knowledge bases from technical documentation
- **ML Dataset Creation**: Generate training data for NLP and computer vision models
- **Document Analysis**: Extract structured information from legal and technical documents
- **Content Management**: Automate document processing for digital libraries

## 🔍 Quality Metrics

The system provides comprehensive quality metrics:

- **OCR Accuracy**: Character and word-level recognition accuracy
- **Content Extraction**: Success rate for abstracts, references, and images
- **Processing Speed**: Average processing time per page/document
- **Error Rate**: Failed processing attempts with detailed error logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- [Design Document](src/README.md) - Detailed technical specifications
- [API Reference](docs/api.md) - Complete API documentation
- [Configuration Guide](docs/config.md) - Configuration options and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PaddleOCR team for the excellent OCR engine
- Tesseract OCR community for the robust text recognition
- Contributors and users who provide feedback and improvements

## 📞 Support

- Create an issue for bug reports or feature requests
- Join our [Discord community](https://discord.gg/ryze-data) for discussions
- Check the [FAQ](docs/faq.md) for common questions

---

**Made with ❤️ by the Ryze-Data team**