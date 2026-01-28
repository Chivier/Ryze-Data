# RYZE-DATA Design Document

## Implementation Status Overview

| Module | Status | Description |
|--------|--------|-------------|
| OCR Model | ✅ Implemented | `chunked-ocr.py` - PDF to Markdown conversion |
| API Balancer | ✅ Implemented | `api_key_balancer.py` - Multi-key load balancing |
| Config Manager | ✅ Implemented | `config_manager.py` - Configuration management |
| NatureScraper | ✅ Implemented | `scrapers/nature_scraper.py` - Nature article scraping |
| DataInspector | ✅ Implemented | `cli/data_inspector.py` - Data inspection tools |
| Text QA Generator | ✅ Implemented | `generators/text_qa_generator.py` - Text QA generation |
| Vision QA Generator | ✅ Implemented | `generators/vision_qa_generator.py` - Vision QA generation |
| Content Parser | 📋 Planned | Structured content extraction |

## Module 1: OCR Model ✅

OCR Model is provided by [surya](https://github.com/datalab-to/surya) and [marker](https://github.com/datalab-to/marker).

**Implementation:** `src/chunked-ocr.py`

### Input:
- `Input PDF Path`

Folder Structure:

```
PDF_Path
├── paper1.pdf
├── paper2.pdf
├── paper3.pdf
├── ...
├── paperN.pdf
└── metadata.json(optional)
```

### Output:
- OCR Result Folder
```
OCR_Result_Folder
├── paper1
|   ├── figure1.png
|   ├── figure2.png
|   ├── ...
|   ├── figureN.png
|   ├── paper1.md (exact same name with this folder)
|   └── paper1_meta.json (ocr metadata for this paper)
├── paper2
├── ...
└── paperN
```

## Module 2: API Key Balancer ✅

**Implementation:** `src/api_key_balancer.py`

Multi-API key load balancer with automatic retry and fallback.

### Features:
- Multiple API key rotation
- Automatic failure retry with backoff
- Request queue management
- Statistics and monitoring

### Usage:

```python
from src.api_key_balancer import OpenAIAPIBalancer

balancer = OpenAIAPIBalancer(api_keys=["key1", "key2", "key3"])

request_id = balancer.submit_chat_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

result = balancer.get_result(timeout=60)
stats = balancer.get_statistics()
```

## Module 3: QA Generators ✅

**Implementation:** `src/generators/`

### Components:

| File | Description |
|------|-------------|
| `base_generator.py` | Abstract base class with `QAPair` dataclass |
| `prompt_manager.py` | Loads and formats prompt templates |
| `text_qa_generator.py` | Generates text QA from OCR markdown |
| `vision_qa_generator.py` | Generates vision QA from figures |

### Text QA Generator

Generates question-answer pairs from OCR-processed markdown files.

**Input:** OCR markdown files (`data/ocr_results/*.md`)

**Output:** JSONL files (`data/sft_data/{paper_id}_qa.jsonl`)

**Prompt Types:**
- `factual` - 3 questions about specific findings
- `mechanism` - 2 questions about how things work
- `application` - 2 questions about applications

```python
from src.generators import TextQAGenerator

generator = TextQAGenerator(
    ocr_dir="./data/ocr_results",
    abstract_dir="./data/abstracts",
    output_dir="./data/sft_data",
    model="gpt-4o-mini",
    qa_ratio=8,
    quality_filter=False
)
generator.run()
```

### Vision QA Generator

Generates visual QA pairs from scientific figures in LlamaFactory-compatible format.

**Input:**
- Figure context JSON (`data/vlm_preprocessing/{paper_id}.json`)
- Figure images (`*.jpeg`)

**Output:** JSONL files (`data/vlm_sft_data/{paper_id}_vision_qa.jsonl`)

**Prompt Types:**
- `visual-factual` - 3 questions about visible elements
- `visual-mechanism` - 2 questions about processes
- `visual-data-extraction` - 1 question about specific values
- `visual-analysis` - 3 questions requiring interpretation
- `visual-comparison` - 2 questions comparing elements

```python
from src.generators import VisionQAGenerator

generator = VisionQAGenerator(
    vlm_dir="./data/vlm_preprocessing",
    abstract_dir="./data/abstracts",
    output_dir="./data/vlm_sft_data",
    model="gpt-4o-mini",
    workers=4,
    qa_ratio=8
)
generator.run()
```

### Output Formats

**Text QA:**
```json
{"question": "...", "answer": "...", "difficulty": "easy", "question_type": "factual", "paper_id": "...", "section": "section_0", "context": "...", "quality_score": 0.0, "metadata": {}}
```

**Vision QA (LlamaFactory):**
```json
{"messages": [{"role": "user", "content": "Question <image>"}, {"role": "assistant", "content": "Answer"}], "images": ["path/to/figure.jpeg"], "metadata": {"paper_id": "...", "figure_id": "...", "question_type": "...", "difficulty": "..."}}
```

## CLI Usage

```bash
# Generate text QA
uv run python -m src.cli.main generate-qa --mode text --qa-ratio 8

# Generate vision QA with 4 workers
uv run python -m src.cli.main generate-qa --mode vision --workers 4

# Generate both with quality filtering
uv run python -m src.cli.main generate-qa --mode both --quality-filter

# Run full pipeline including QA
uv run python -m src.cli.main pipeline --stages qa
```

## Module 4: Content Parser (📋 Planned)

Parse the markdown content into structured content, including:
- Abstract
- Text Chunks
- Images
- References

Then store the structured content to the output folder.
