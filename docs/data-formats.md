# 数据格式规范

## 目录

- [数据流概览](#数据流概览)
- [元数据格式](#元数据格式)
- [OCR 输出格式](#ocr-输出格式)
- [图片提取格式](#图片提取格式)
- [QA 数据格式](#qa-数据格式)
- [日志格式](#日志格式)
- [数据验证](#数据验证)

## 数据流概览

```
爬取阶段 → 元数据 (CSV)
    ↓
下载阶段 → PDF 文件
    ↓
OCR 阶段 → Markdown + 图片
    ↓
处理阶段 → 结构化 JSON
    ↓
生成阶段 → QA 对 (JSONL)
```

## 元数据格式

### 文章列表 (all_articles.csv)

**位置**: `{RYZE_NATURE_DATA}/all_articles.csv`

**格式**: CSV

**字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| title | string | ✓ | 文章标题 |
| url | string | ✓ | 文章 URL |
| abstract | string | ✓ | 摘要文本 |
| open_access | string | ✓ | 开放获取状态 (Y/N) |
| date | string | ✓ | 发布日期 (YYYY-MM-DD) |
| author | string | ✓ | 作者列表（逗号分隔） |

**示例**:
```csv
title,url,abstract,open_access,date,author
"Deep learning for molecular design","https://nature.com/articles/s41586-021-03819-2","This paper presents...","Y","2021-08-12","John Doe, Jane Smith"
```

### 下载状态 (download_status.json)

**位置**: `{RYZE_PDF_DIR}/download_status.json`

**格式**: JSON

**结构**:
```json
{
  "total": 1000,
  "completed": 850,
  "failed": 10,
  "skipped": 140,
  "papers": {
    "nature04244": {
      "status": "completed",
      "url": "https://nature.com/articles/nature04244.pdf",
      "file_path": "/data/pdfs/nature04244.pdf",
      "file_size": 2456789,
      "download_time": "2024-01-15T10:30:00Z",
      "retry_count": 0
    }
  }
}
```

## OCR 输出格式

### 文章 Markdown ({paper_id}.md)

**位置**: `{RYZE_OCR_OUTPUT}/{paper_id}/{paper_id}.md`

**格式**: Markdown

**结构**:
```markdown
# [文章标题]

## Abstract
[摘要内容]

## Introduction
[引言内容]

## Methods
[方法部分]

### [子标题]
[内容]

## Results
[结果部分]

<span id="fig1"></span>
![](page_1_Figure_1.jpeg)
**Figure 1**: [图片说明]

## Discussion
[讨论部分]

## References
1. [参考文献1]
2. [参考文献2]
```

### OCR 元数据 ({paper_id}_meta.json)

**位置**: `{RYZE_OCR_OUTPUT}/{paper_id}/{paper_id}_meta.json`

**格式**: JSON

**结构**:
```json
{
  "paper_id": "nature04244",
  "title": "Article Title",
  "authors": ["Author 1", "Author 2"],
  "journal": "Nature",
  "year": 2024,
  "doi": "10.1038/nature04244",
  "pages": {
    "total": 12,
    "start": 100,
    "end": 111
  },
  "figures": {
    "count": 5,
    "list": [
      {
        "id": "fig1",
        "page": 3,
        "caption": "Figure 1 caption",
        "file": "page_3_Figure_1.jpeg"
      }
    ]
  },
  "tables": {
    "count": 2,
    "list": []
  },
  "sections": [
    {
      "title": "Introduction",
      "start_page": 1,
      "word_count": 500
    }
  ],
  "processing": {
    "ocr_engine": "marker",
    "processing_time": 45.2,
    "timestamp": "2024-01-15T10:30:00Z",
    "quality_score": 0.95
  }
}
```

## 图片提取格式

### 图片数据 ({paper_id}.json)

**位置**: `{RYZE_VLM_PREPROCESSING}/{paper_id}.json`

**格式**: JSON

**结构**:
```json
[
  {
    "figure_id": "fig1",
    "figure_path": "/data/ocr_results/nature04244/page_3_Figure_1.jpeg",
    "figure_caption": "Architecture of the neural network",
    "figure_legend": "The network consists of multiple layers...",
    "related_info": [
      {
        "position": "before_figure",
        "type": "text",
        "info": "As shown in Figure 1, the architecture..."
      },
      {
        "position": "after_figure",
        "type": "text",
        "info": "The results demonstrate that..."
      }
    ],
    "metadata": {
      "page_number": 3,
      "figure_number": 1,
      "width": 800,
      "height": 600,
      "format": "jpeg",
      "size_bytes": 156789
    },
    "references_in_text": [
      {
        "section": "Results",
        "text": "Figure 1 shows the overall architecture",
        "page": 5
      }
    ]
  }
]
```

### 摘要文件 ({paper_id}_abstract.txt)

**位置**: `{RYZE_ABSTRACT_DIR}/{paper_id}_abstract.txt`

**格式**: 纯文本

**示例**:
```text
This paper presents a novel approach to deep learning for molecular design. We demonstrate that neural networks can effectively learn molecular representations and generate novel compounds with desired properties. Our method achieves state-of-the-art results on benchmark datasets.
```

## QA 数据格式

### 文本 QA 格式 ({paper_id}_qa.jsonl)

**位置**: `{RYZE_SFT_DATA}/{paper_id}_qa.jsonl`

**格式**: JSONL (每行一个 JSON 对象)

**结构**:
```jsonl
{
  "question": "What is the main contribution of this paper?",
  "answer": "The main contribution is a novel neural network architecture for molecular design that achieves state-of-the-art performance.",
  "paper_id": "nature04244",
  "section": "abstract",
  "difficulty": "medium",
  "question_type": "factual",
  "quality_score": 4.5,
  "context": "This paper presents a novel approach to deep learning...",
  "metadata": {
    "generated_by": "gpt-4",
    "timestamp": "2024-01-15T10:30:00Z",
    "prompt_template": "factual.txt"
  }
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| question | string | 问题文本 |
| answer | string | 答案文本 |
| paper_id | string | 论文 ID |
| section | string | 来源章节 |
| difficulty | string | 难度等级 (easy/medium/hard) |
| question_type | string | 问题类型 |
| quality_score | float | 质量分数 (0-5) |
| context | string | 上下文信息 |

### 视觉 QA 格式 ({paper_id}_vision_qa.jsonl)

**位置**: `{RYZE_VLM_SFT_DATA}/{paper_id}_vision_qa.jsonl`

**格式**: JSONL (LlamaFactory 兼容格式)

**结构**:
```jsonl
{
  "messages": [
    {
      "role": "user",
      "content": "What does this figure show? <image>"
    },
    {
      "role": "assistant",
      "content": "This figure shows a convolutional neural network architecture with multiple layers for feature extraction."
    },
    {
      "role": "user",
      "content": "What is the significance of the highlighted region? <image>"
    },
    {
      "role": "assistant",
      "content": "The highlighted region indicates the attention mechanism that focuses on relevant molecular structures."
    }
  ],
  "images": [
    "/data/ocr_results/nature04244/page_3_Figure_1.jpeg",
    "/data/ocr_results/nature04244/page_3_Figure_1.jpeg"
  ],
  "metadata": {
    "paper_id": "nature04244",
    "figure_id": "fig1",
    "qa_pairs": 2,
    "quality_scores": [4.2, 4.5],
    "generated_by": "gpt-4-vision",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 问题类型分类

| 类型 | 说明 | 示例 |
|------|------|------|
| factual | 事实性问题 | "What is the accuracy of the model?" |
| mechanism | 机制理解 | "How does the attention mechanism work?" |
| application | 应用相关 | "What are the potential applications?" |
| comparison | 比较分析 | "How does this compare to previous methods?" |
| visual-analysis | 视觉分析 | "What trend is shown in the graph?" |
| visual-data | 数据提取 | "What is the value at x=10?" |

## 日志格式

### 处理日志 (processing_log.csv)

**位置**: `{RYZE_LOGS_DIR}/processing_log.csv`

**格式**: CSV

**结构**:
```csv
timestamp,paper_id,stage,status,duration,error_message,details
2024-01-15T10:30:00Z,nature04244,ocr,completed,45.2,,{"pages":12,"figures":5}
2024-01-15T10:31:00Z,nature04245,ocr,failed,120.0,Timeout error,{"retry_count":3}
```

### 视觉处理日志 (vision_processing_log.csv)

**位置**: `{RYZE_LOGS_DIR}/vision_processing_log.csv`

**格式**: CSV

**结构**:
```csv
timestamp,paper_id,figure_count,qa_generated,avg_quality,model,processing_time
2024-01-15T10:30:00Z,nature04244,5,40,4.2,gpt-4-vision,120.5
```

### 系统日志格式

**JSON 格式**:
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "PipelineManager",
  "message": "Processing stage completed",
  "context": {
    "stage": "ocr",
    "paper_id": "nature04244",
    "duration": 45.2
  },
  "traceback": null
}
```

**文本格式**:
```
2024-01-15 10:30:00,123 - PipelineManager - INFO - Processing stage completed [stage=ocr, paper_id=nature04244]
```

## 数据验证

### 模式验证

使用 JSON Schema 验证数据格式：

```python
import jsonschema

# QA 数据模式
qa_schema = {
    "type": "object",
    "required": ["question", "answer", "paper_id"],
    "properties": {
        "question": {"type": "string", "minLength": 10},
        "answer": {"type": "string", "minLength": 20},
        "paper_id": {"type": "string", "pattern": "^[a-z]+[0-9]+$"},
        "quality_score": {"type": "number", "minimum": 0, "maximum": 5}
    }
}

# 验证数据
jsonschema.validate(qa_data, qa_schema)
```

### 数据完整性检查

```python
def validate_paper_data(paper_id: str) -> dict:
    """验证论文数据完整性"""
    checks = {
        "pdf_exists": False,
        "ocr_completed": False,
        "figures_extracted": False,
        "qa_generated": False,
        "consistency": True
    }
    
    # 检查 PDF 文件
    pdf_path = Path(f"{PDF_DIR}/{paper_id}.pdf")
    checks["pdf_exists"] = pdf_path.exists()
    
    # 检查 OCR 输出
    ocr_path = Path(f"{OCR_DIR}/{paper_id}/{paper_id}.md")
    checks["ocr_completed"] = ocr_path.exists()
    
    # 检查图片提取
    figures_path = Path(f"{VLM_DIR}/{paper_id}.json")
    checks["figures_extracted"] = figures_path.exists()
    
    # 检查 QA 生成
    qa_path = Path(f"{SFT_DIR}/{paper_id}_qa.jsonl")
    checks["qa_generated"] = qa_path.exists()
    
    return checks
```

### 质量指标

| 指标 | 计算方法 | 阈值 |
|------|---------|------|
| OCR 质量 | 文本置信度平均值 | > 0.8 |
| QA 质量 | 模型评分平均值 | > 2.5 |
| 完整性 | 有效字段比例 | > 0.9 |
| 一致性 | 交叉验证通过率 | = 1.0 |

## 数据转换工具

### CSV 转 JSON

```python
import csv
import json

def csv_to_json(csv_file: str, json_file: str):
    """将 CSV 转换为 JSON"""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

### JSONL 合并

```python
def merge_jsonl_files(files: list, output_file: str):
    """合并多个 JSONL 文件"""
    with open(output_file, 'w', encoding='utf-8') as out:
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    out.write(line)
```

### 数据采样

```python
import random

def sample_qa_data(input_file: str, output_file: str, sample_size: int):
    """从 QA 数据中随机采样"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    sampled = random.sample(lines, min(sample_size, len(lines)))
    
    with open(output_file, 'w') as f:
        for line in sampled:
            f.write(line)
```