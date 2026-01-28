# RYZE-DATA Design Document

## Implementation Status Overview

| Module | Status | Description |
|--------|--------|-------------|
| OCR Model | ✅ Implemented | `chunked-ocr.py` - PDF to Markdown conversion |
| API Balancer | ✅ Implemented | `api_key_balancer.py` - Multi-key load balancing |
| Config Manager | ✅ Implemented | `config_manager.py` - Configuration management |
| NatureScraper | ✅ Implemented | `scrapers/nature_scraper.py` - Nature article scraping |
| DataInspector | ✅ Implemented | `cli/data_inspector.py` - Data inspection tools |
| Content Parser | 📋 Planned | Structured content extraction |
| QA Generator | 📋 Planned | Question-answer pair generation |

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

metadata.json(optional): If some metadata is provided, it will be used to filter the papers or provide some extra information.

```json
{
    "paper1": {
        "title": "Paper 1",
        "author": "Author 1",
        "year": 2021,
        "some_other_metadata":
        ...
    },
    "paper2": {
        "title": "Paper 2",
        "author": "Author 2",
        "year": 2022,
        "some_other_metadata":
        ...
    },
    ...
}
```

### Process:
- Use OCR model to convert PDF to markdown format
- Save markdown to output path, create a new folder for each paper, if the paper contains images, save the images to the folder/images

### Output:

- OCR Status Csv Record

```csv
paper_name, original_pdf_path, ocr_status, ocr_time, ocr_result_path
paper1, /path/to/paper1.pdf, success, 2021-01-01 12:00:00, /path/to/paper1_ocr
paper2, /path/to/paper2.pdf, failed, 2021-01-01 12:00:00, /path/to/paper2_ocr
...
```

- OCR Result Folder
```
OCR_Result_Folder
├── paper1
|   ├── figure1.png
|   ├── figure2.png
|   ├── ...
|   ├── figureN.png
|   ├── paper1.md (exact same name with this folder)
|   └── paper1_meta.json (ocr metadata for this paper, use to check markdown content)
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

# Initialize with multiple API keys
balancer = OpenAIAPIBalancer(
    api_keys=["key1", "key2", "key3"],
    num_workers=4
)

# Submit chat completion request
request_id = balancer.submit_chat_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Get result
result = balancer.get_result(request_id, timeout=60)

# Get statistics
stats = balancer.get_statistics()
```

## Module 3: Content Parser (📋 Planned)

Parse the markdown content into structured content, including:
- Abstract
- Text Chunks
- Images
- References
Then store the structured content to the output folder.

### Input:
- OCR Result Folder Path

### Process:
- Parse the markdown content

### Output:
- Parsed Content Status Csv Record

```csv
paper_name, parsed_content_status, parsed_content_time, parsed_content_result_path, total_pages, total_images, total_references, total_text_chunks, total_images, total_references, total_text_chunks
paper1, success, 2021-01-01 12:00:00, /path/to/paper1_parsed, 15, 8, 3, 10, 2, 1, 5
paper2, failed, 2021-01-01 12:00:00, /path/to/paper2_parsed, 0, 0, 0, 0, 0, 0, 0
...
```

- Output Folder Structure
```
OCR_Result_Folder
├── paper1
|   ├── figure1.png
|   ├── figure2.png
|   ├── ...
|   ├── figureN.png
|   ├── paper1.md (exact same name with this folder)
|   ├── paper1_abstract.md (abstract content)
|   ├── paper1_text_chunks.md (text chunks content)
|   ├── paper1_images.json (images content, including figure name, figure path, legend, caption, related text)
|   ├── paper1_references.md (references content)
|   └── paper1_meta.json (ocr metadata for this paper, use to check markdown content)
├── paper2
├── ...
└── paperN
```

## Module 4: QA Template Manager (📋 Planned)

A group of QA templates are provided by [Ryze-Data](https://github.com/Chivier/Ryze-Data), and the user can add more templates to the template manager.

## Module 5: Data Packer and Dataset Generator (📋 Planned)

### Input:
- Parsed Content Folder Path
- QA Template Manager

### Process:
- Pack the parsed content into a dataset, including:
    - Abstract
    - Text Chunks
    - Images
    - References
    - QA Templates
Then call LLM batch inference to generate QA pairs, and store the QA pairs to the output folder.

### Output:
- A group of QA pairs with metadata
