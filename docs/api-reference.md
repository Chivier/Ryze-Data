# API Reference

## Table of Contents

- [ConfigManager](#configmanager)
- [QA Generators](#qa-generators)
  - [QAPair](#qapair)
  - [BaseQAGenerator](#baseqagenerator)
  - [TextQAGenerator](#textqagenerator)
  - [VisionQAGenerator](#visionqagenerator)
  - [PromptManager](#promptmanager)
- [OpenAIAPIBalancer](#openaiapibalancer)
- [CLI Commands](#cli-commands)

## ConfigManager

Configuration manager for loading, validating, and managing system settings.

### Class Definition

```python
from src.config_manager import ConfigManager

config = ConfigManager()
```

### Methods

#### `load(config_path: str = "config.json") -> None`

Load configuration from file and environment variables.

```python
config = ConfigManager()
config.load("custom_config.json")
```

#### `validate() -> bool`

Validate configuration settings.

```python
if config.validate():
    print("Configuration is valid")
```

#### `get(key_path: str, default: Any = None) -> Any`

Get configuration value by dot-separated path.

```python
batch_size = config.get("ocr.batch_size", 10)
model = config.get("qa_generation_model.model", "gpt-4")
```

### Configuration Dataclasses

```python
@dataclass
class PathConfig:
    data_root: str = "./data"
    ocr_output: str = "./data/ocr_results"
    vlm_preprocessing: str = "./data/vlm_preprocessing"
    sft_data: str = "./data/sft_data"
    vlm_sft_data: str = "./data/vlm_sft_data"
    prompts_dir: str = "./prompts"

@dataclass
class ProcessingConfig:
    parallel_workers: int = 4
    quality_threshold: float = 2.5
    qa_ratio: int = 8

@dataclass
class ModelConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    api_endpoint: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 2048
    temperature: float = 0.7
```

---

## QA Generators

The generators module provides classes for generating question-answer pairs from scientific papers.

### QAPair

Dataclass representing a single question-answer pair.

```python
from src.generators import QAPair

qa = QAPair(
    question="What is CRISPR-Cas9?",
    answer="CRISPR-Cas9 is a gene editing technology...",
    difficulty="medium",
    question_type="factual",
    paper_id="nature04244",
    section="section_0",
    context="The research text...",
    quality_score=4.5,
    metadata={"prompt_template": "factual.txt"}
)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `question` | str | The question text |
| `answer` | str | The answer text |
| `difficulty` | str | Difficulty level (easy/medium/hard) |
| `question_type` | str | Type of question (factual/mechanism/application) |
| `paper_id` | str | Source paper identifier |
| `section` | str | Source section identifier |
| `context` | str | Contextual text (truncated) |
| `quality_score` | float | Quality score (0-5) |
| `metadata` | dict | Additional metadata |

#### Methods

```python
# Convert to dictionary
qa_dict = qa.to_dict()

# Convert to JSONL line
jsonl_line = qa.to_jsonl_line()
```

---

### BaseQAGenerator

Abstract base class for QA generators with shared functionality.

```python
from src.generators import BaseQAGenerator

class CustomGenerator(BaseQAGenerator):
    def run(self):
        # Implementation
        pass

    def process_paper(self, paper_path):
        # Implementation
        pass
```

#### Constructor

```python
def __init__(
    self,
    output_dir: str,
    model: str,
    qa_ratio: int = 8,
    config: Optional[ConfigManager] = None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_dir` | str | Directory to save generated QA pairs |
| `model` | str | Model name for API calls |
| `qa_ratio` | int | Target QA pairs per unit (default: 8) |
| `config` | ConfigManager | Optional config manager instance |

#### Key Methods

```python
# Initialize API balancer with keys
generator._init_balancer(api_keys=["sk-key1", "sk-key2"])

# Filter QA pairs by quality score
filtered = generator._filter_qa_by_quality(qa_pairs, threshold=2.5)

# Parse JSON from LLM response (handles markdown code blocks)
parsed = generator._parse_json_response(response_text)

# Save QA pairs to JSONL file
path = generator._save_qa_pairs(qa_pairs, "paper_qa")

# Create rich progress bar
with generator.create_progress() as progress:
    task = progress.add_task("Processing...", total=100)
```

---

### TextQAGenerator

Generate QA pairs from OCR-processed markdown files.

```python
from src.generators import TextQAGenerator

generator = TextQAGenerator(
    ocr_dir="./data/ocr_results",
    abstract_dir="./data/abstracts",
    output_dir="./data/sft_data",
    model="gpt-4o-mini",
    qa_ratio=8,
    quality_filter=False,
    quality_threshold=2.5,
    max_section_chars=3000
)

generator.run()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ocr_dir` | str | - | Directory containing OCR markdown files |
| `abstract_dir` | str | - | Directory containing paper abstracts |
| `output_dir` | str | - | Output directory for QA JSONL files |
| `model` | str | - | Model name for API calls |
| `qa_ratio` | int | 8 | Target QA pairs per section |
| `quality_filter` | bool | False | Enable quality filtering |
| `quality_threshold` | float | 2.5 | Minimum quality score |
| `max_section_chars` | int | 3000 | Maximum characters per section chunk |

#### Prompt Configuration

| Prompt Type | Questions | Description |
|-------------|-----------|-------------|
| `factual` | 3 | Fact-based questions about findings |
| `mechanism` | 2 | Questions about how things work |
| `application` | 2 | Questions about practical applications |

#### Output Format

```jsonl
{"question": "What was the editing efficiency achieved?", "answer": "The study achieved 95% editing efficiency.", "difficulty": "easy", "question_type": "factual", "paper_id": "sample_paper", "section": "section_0", "context": "...", "quality_score": 0.0, "metadata": {}}
```

---

### VisionQAGenerator

Generate QA pairs from scientific figures using vision models.

```python
from src.generators import VisionQAGenerator

generator = VisionQAGenerator(
    vlm_dir="./data/vlm_preprocessing",
    abstract_dir="./data/abstracts",
    output_dir="./data/vlm_sft_data",
    model="gpt-4o-mini",
    workers=4,
    qa_ratio=8,
    quality_filter=False,
    quality_threshold=2.5
)

generator.run()
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vlm_dir` | str | - | Directory with figure context JSON and images |
| `abstract_dir` | str | - | Directory containing paper abstracts |
| `output_dir` | str | - | Output directory for vision QA JSONL files |
| `model` | str | - | Vision model name for API calls |
| `workers` | int | 4 | Number of parallel workers |
| `qa_ratio` | int | 8 | Target QA pairs per figure |
| `quality_filter` | bool | False | Enable quality filtering |
| `quality_threshold` | float | 2.5 | Minimum quality score |

#### Prompt Configuration

| Prompt Type | Questions | Description |
|-------------|-----------|-------------|
| `visual-factual` | 3 | Questions about visible elements |
| `visual-mechanism` | 2 | Questions about depicted processes |
| `visual-data-extraction` | 1 | Questions about specific data values |
| `visual-analysis` | 3 | Questions requiring figure interpretation |
| `visual-comparison` | 2 | Questions comparing elements |

#### Input Format (Figure Context JSON)

```json
{
  "paper_id": "sample_paper",
  "figures": [
    {
      "id": "Figure_1",
      "label": "Figure 1",
      "image_path": "sample_paper_Figure_1.jpeg",
      "caption": "Comparison of grain width...",
      "related_info": {
        "before": "Text before figure...",
        "after": "Text after figure..."
      }
    }
  ]
}
```

#### Output Format (LlamaFactory Compatible)

```jsonl
{"messages": [{"role": "user", "content": "What does the figure show? <image>"}, {"role": "assistant", "content": "The figure shows a bar chart comparing..."}], "images": ["data/vlm_preprocessing/sample_paper_Figure_1.jpeg"], "metadata": {"paper_id": "sample_paper", "figure_id": "Figure_1", "question_type": "factual", "difficulty": "easy"}}
```

---

### PromptManager

Manages loading and formatting of prompt templates.

```python
from src.generators import PromptManager

pm = PromptManager(prompts_dir="./prompts")
```

#### Methods

```python
# Get formatted text prompt
prompt = pm.get_text_prompt(
    prompt_type="factual",
    context="The research findings show..."
)

# Get formatted vision prompt
prompt = pm.get_vision_prompt(
    prompt_type="visual-factual",
    context="Figure showing bar chart..."
)

# Get quality evaluation prompt
prompt = pm.get_quality_prompt(
    question="What is X?",
    answer="X is Y.",
    context="Original text...",
    is_vision=False
)

# List available prompts
text_prompts = pm.list_text_prompts()    # ['factual', 'mechanism', 'application', 'quality']
vision_prompts = pm.list_vision_prompts()  # ['visual-factual', ...]

# Clear template cache
pm.clear_cache()
```

---

## OpenAIAPIBalancer

API key load balancer for OpenAI requests with automatic retry and failover.

```python
from src.api_key_balancer import OpenAIAPIBalancer

balancer = OpenAIAPIBalancer(api_keys=["sk-key1", "sk-key2", "sk-key3"])
```

### Constructor

```python
def __init__(self, api_keys: List[str], max_queue_size: int = 1000)
```

### Methods

#### `submit_chat_completion(model, messages, callback, **kwargs) -> str`

Submit a chat completion request.

```python
request_id = balancer.submit_chat_completion(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_retries=3
)
```

#### `get_result(timeout) -> Optional[APIRequest]`

Get a processed result from the queue.

```python
result = balancer.get_result(timeout=30.0)
if result and result.status == RequestStatus.SUCCESS:
    content = result.result.choices[0].message.content
```

#### `get_statistics() -> Dict[str, Any]`

Get balancer statistics.

```python
stats = balancer.get_statistics()
# {
#     "total_requests": 100,
#     "pending_requests": 5,
#     "retry_requests": 2,
#     "workers": [{"thread_id": 0, "processed": 35, "failed": 2}, ...]
# }
```

#### `shutdown(wait: bool = True)`

Shutdown the balancer.

```python
balancer.shutdown(wait=True)  # Wait for pending requests
```

### RequestStatus Enum

```python
class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
```

---

## CLI Commands

### Main Command

```bash
uv run python -m src.cli.main [OPTIONS] COMMAND [ARGS]
```

**Global Options**:
- `--config, -c`: Configuration file path
- `--env, -e`: Environment file path

### generate-qa

Generate QA pairs from processed data.

```bash
uv run python -m src.cli.main generate-qa [OPTIONS]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | choice | both | Generation mode: text, vision, or both |
| `--model` | text | config | Model to use for generation |
| `--workers, -w` | int | 4 | Parallel workers (vision mode) |
| `--qa-ratio` | int | 8 | QA pairs per section/figure |
| `--quality-filter` | flag | off | Enable quality filtering |

**Examples**:
```bash
# Text QA only
uv run python -m src.cli.main generate-qa --mode text

# Vision QA with 8 workers
uv run python -m src.cli.main generate-qa --mode vision --workers 8

# Both modes with quality filtering
uv run python -m src.cli.main generate-qa --mode both --quality-filter
```

### pipeline

Run complete processing pipeline.

```bash
uv run python -m src.cli.main pipeline [OPTIONS]
```

**Options**:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stages, -s` | multiple | all | Stages to run: scrape, download, ocr, extract, qa |
| `--workers, -w` | int | 4 | Number of parallel workers |
| `--quality-filter` | flag | off | Enable QA quality filtering |

**Examples**:
```bash
# Run all stages
uv run python -m src.cli.main pipeline

# Run specific stages
uv run python -m src.cli.main pipeline --stages ocr --stages qa

# With quality filtering
uv run python -m src.cli.main pipeline --stages qa --quality-filter
```

### Other Commands

```bash
# Scrape Nature metadata
uv run python -m src.cli.main scrape

# Run OCR processing
uv run python -m src.cli.main ocr --input-dir <pdf_dir> --output-dir <output_dir>

# Show configuration
uv run python -m src.cli.main config-show

# Inspect data
uv run python -m src.cli.main inspect all
uv run python -m src.cli.main inspect stage ocr --sample 5
```
