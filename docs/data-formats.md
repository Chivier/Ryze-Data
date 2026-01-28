# 数据格式规范

## 目录

- [数据流概览](#数据流概览)
- [元数据格式](#元数据格式)
- [OCR 输出格式](#ocr-输出格式)
- [图片提取格式](#图片提取格式-计划中)
- [QA 数据格式](#qa-数据格式-计划中)
- [日志格式](#日志格式)
- [数据验证](#数据验证)

## 数据流概览

```
爬取阶段 → 元数据 (CSV)           ✅ 已实现
    ↓
OCR 阶段 → Markdown + 图片        ✅ 已实现
    ↓
处理阶段 → 结构化 JSON            📋 计划中
    ↓
生成阶段 → QA 对 (JSONL)          📋 计划中
```

## 元数据格式

### 文章列表 (CSV) ✅ 已实现

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

## OCR 输出格式

### OCR 状态记录 (CSV) ✅ 已实现

**位置**: `{RYZE_OCR_OUTPUT}/ocr_status.csv`

**格式**: CSV

**字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| paper_name | string | ✓ | 论文名称 |
| original_pdf_path | string | ✓ | 原始 PDF 路径 |
| ocr_status | string | ✓ | OCR 状态 (success/failed) |
| ocr_time | string | ✓ | OCR 处理时间 |
| ocr_result_path | string | ✓ | OCR 结果路径 |

**示例**:
```csv
paper_name,original_pdf_path,ocr_status,ocr_time,ocr_result_path
paper1,/path/to/paper1.pdf,success,2024-01-15 12:00:00,/path/to/paper1_ocr
paper2,/path/to/paper2.pdf,failed,2024-01-15 12:05:00,/path/to/paper2_ocr
```

### 文章 Markdown ({paper_id}.md) ✅ 已实现

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

### OCR 元数据 ({paper_id}_meta.json) ✅ 已实现

**位置**: `{RYZE_OCR_OUTPUT}/{paper_id}/{paper_id}_meta.json`

**格式**: JSON

**结构**:
```json
{
  "paper_id": "nature04244",
  "title": "Article Title",
  "pages": {
    "total": 12
  },
  "figures": {
    "count": 5,
    "list": [
      {
        "id": "fig1",
        "page": 3,
        "file": "page_3_Figure_1.jpeg"
      }
    ]
  },
  "processing": {
    "ocr_engine": "marker",
    "processing_time": 45.2,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### OCR 结果目录结构 ✅ 已实现

```
OCR_Result_Folder/
├── paper1/
│   ├── figure1.png
│   ├── figure2.png
│   ├── ...
│   ├── figureN.png
│   ├── paper1.md              # Markdown 内容
│   └── paper1_meta.json       # OCR 元数据
├── paper2/
├── ...
└── paperN/
```

## 图片提取格式 (📋 计划中)

> 以下格式为计划中的功能，尚未实现。

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
      }
    ],
    "metadata": {
      "page_number": 3,
      "figure_number": 1,
      "width": 800,
      "height": 600,
      "format": "jpeg"
    }
  }
]
```

## QA 数据格式 (📋 计划中)

> 以下格式为计划中的功能，尚未实现。

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
    }
  ],
  "images": [
    "/data/ocr_results/nature04244/page_3_Figure_1.jpeg"
  ],
  "metadata": {
    "paper_id": "nature04244",
    "figure_id": "fig1",
    "qa_pairs": 1,
    "quality_scores": [4.2],
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
  }
}
```

**文本格式**:
```
2024-01-15 10:30:00,123 - PipelineManager - INFO - Processing stage completed [stage=ocr, paper_id=nature04244]
```

## 数据验证

### 数据完整性检查

```python
def validate_paper_data(paper_id: str) -> dict:
    """验证论文数据完整性"""
    checks = {
        "ocr_completed": False,
        "markdown_exists": False,
        "meta_exists": False
    }

    # 检查 OCR 输出
    ocr_dir = Path(f"{OCR_DIR}/{paper_id}")
    checks["ocr_completed"] = ocr_dir.exists()

    # 检查 Markdown 文件
    md_path = ocr_dir / f"{paper_id}.md"
    checks["markdown_exists"] = md_path.exists()

    # 检查元数据
    meta_path = ocr_dir / f"{paper_id}_meta.json"
    checks["meta_exists"] = meta_path.exists()

    return checks
```

### 质量指标

| 指标 | 计算方法 | 阈值 |
|------|---------|------|
| OCR 质量 | 文本置信度平均值 | > 0.8 |
| 完整性 | 有效字段比例 | > 0.9 |

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
