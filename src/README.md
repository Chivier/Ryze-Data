# RYZE-DATA Design Document

## Implementation Status Overview

| Module | Status | Description |
|--------|--------|-------------|
| OCR Module | ✅ Implemented | `ocr/` - Extensible OCR with 6 models (Marker, DeepSeek v1/v2, MarkItDown, PaddleOCR, GLM-OCR) |
| Benchmark System | ✅ Implemented | `benchmark/` - OCR quality evaluation on ArxivQA & SlideVQA |
| API Balancer | ✅ Implemented | `api_key_balancer.py` - Multi-key load balancing |
| Config Manager | ✅ Implemented | `config_manager.py` - Configuration management |
| NatureScraper | ✅ Implemented | `scrapers/nature_scraper.py` - Nature article scraping |
| DataInspector | ✅ Implemented | `cli/data_inspector.py` - Data inspection tools |
| Text QA Generator | ✅ Implemented | `generators/text_qa_generator.py` - Text QA generation |
| Vision QA Generator | ✅ Implemented | `generators/vision_qa_generator.py` - Vision QA generation |
| Legacy Chunked OCR | ✅ Implemented | `chunked-ocr.py` - Legacy batch OCR (use `ocr/` module instead) |
| Content Parser | 📋 Planned | Structured content extraction |

## Module 1: OCR Module ✅

Extensible, registry-based OCR system supporting 6 models. See `src/ocr/README.md` for the extension guide.

**Implementation:** `src/ocr/`

### Registered Models

| `--ocr-model` | Class | Backend | GPU | Notes |
|----------------|-------|---------|-----|-------|
| `marker` | `MarkerOCR` | CLI (`marker_single`) | Optional | Default. Supports multi-GPU batch via `marker_chunk_convert` |
| `deepseek-ocr` | `DeepSeekOCRv1` | vLLM / Transformers | Required | `deepseek-ai/DeepSeek-OCR`, bfloat16, anti-repetition logits |
| `deepseek-ocr-v2` | `DeepSeekOCRv2` | vLLM / Transformers | Required | `deepseek-ai/DeepSeek-OCR-2` |
| `markitdown` | `MarkItDownOCR` | Python API | No | Microsoft MarkItDown, text extraction (not OCR) |
| `paddleocr` | `PaddleOCRModel` | PaddleOCR | Optional | PP-OCRv5 + PP-StructureV3, OCR or structure mode |
| `glm-ocr` | `GLMOCRModel` | vLLM / Z.AI API | Required | 0.9B params, dual backend: local vLLM or remote API |

### Architecture

```
BaseOCRModel (ABC)
├── MarkerOCR          (marker, CLI wrapper)
├── BaseDeepSeekOCR    (shared base class)
│   ├── DeepSeekOCRv1  (deepseek-ocr)
│   └── DeepSeekOCRv2  (deepseek-ocr-v2)
├── MarkItDownOCR      (markitdown)
├── PaddleOCRModel     (paddleocr)
└── GLMOCRModel        (glm-ocr)
```

### Output Format

```
{output_dir}/{paper_name}/
├── {paper_name}.md    # Markdown output
└── *.png              # Extracted images (optional)
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
