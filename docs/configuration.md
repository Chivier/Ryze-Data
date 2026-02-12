# 配置指南

## 目录

- [配置概述](#配置概述)
- [环境变量配置](#环境变量配置)
- [配置文件详解](#配置文件详解)
- [配置优先级](#配置优先级)
- [常用配置场景](#常用配置场景)
- [高级配置](#高级配置)
- [配置验证](#配置验证)

## 配置概述

Ryze-Data 支持多层次的配置管理，提供灵活的配置方式：

1. **环境变量** (.env 文件)
2. **配置文件** (config.json)
3. **命令行参数**
4. **默认值**

### 快速配置

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑环境变量
vim .env

# 3. 复制配置文件模板
cp config.example.json config.json

# 4. 验证配置
python -m src.cli.main config-show
```

## 环境变量配置

### 基础配置 (.env)

```bash
# ========== 基础路径配置 ==========
# 数据根目录
RYZE_DATA_ROOT=./data

# 日志目录
RYZE_LOGS_DIR=./logs

# ========== 数据路径配置 ==========
# Nature 元数据存储路径
RYZE_NATURE_DATA=${RYZE_DATA_ROOT}/nature_metadata

# PDF 文件存储路径
RYZE_PDF_DIR=${RYZE_DATA_ROOT}/pdfs

# OCR 输出路径
RYZE_OCR_OUTPUT=${RYZE_DATA_ROOT}/ocr_results

# 摘要存储路径
RYZE_ABSTRACT_DIR=${RYZE_DATA_ROOT}/abstracts

# 图片存储路径
RYZE_FIGURES_DIR=${RYZE_DATA_ROOT}/figures

# 视觉预处理路径
RYZE_VLM_PREPROCESSING=${RYZE_DATA_ROOT}/vlm_preprocessing

# 文本 QA 数据路径
RYZE_SFT_DATA=${RYZE_DATA_ROOT}/sft_data

# 视觉 QA 数据路径
RYZE_VLM_SFT_DATA=${RYZE_DATA_ROOT}/vlm_sft_data

# 提示词模板路径
RYZE_PROMPTS_DIR=./prompts
```

### API 配置

```bash
# ========== API 密钥配置 ==========
# OpenAI API 密钥
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# OpenAI API 端点（可选，默认为官方端点）
OPENAI_BASE_URL=https://api.openai.com/v1

# 解析模型 API 密钥
RYZE_PARSING_API_KEY=your-parsing-api-key

# QA 生成 API 密钥
RYZE_QA_API_KEY=your-qa-api-key
```

### 模型配置

```bash
# ========== LLM 模型配置 ==========
# 文本处理模型
RYZE_LLM_MODEL=gpt-4

# 视觉处理模型
RYZE_VISION_MODEL=gpt-4-vision

# QA 生成比例（每个单元生成的 QA 对数）
RYZE_QA_RATIO=8

# 质量阈值（0-5 分）
RYZE_QUALITY_THRESHOLD=2.5
```

### 处理配置

```bash
# ========== 处理配置 ==========
# 批处理大小
RYZE_BATCH_SIZE=10

# 并行工作线程数
RYZE_NUM_WORKERS=4

# 是否启用 GPU
RYZE_GPU_ENABLED=true

# 最大处理文章数
RYZE_MAX_PAPERS=1000
```

### 服务器配置

```bash
# ========== 分布式下载配置 ==========
# 下载服务器列表（逗号分隔）
RYZE_DOWNLOAD_SERVERS=server1,server2,server3

# 工作时间开始（24小时制）
RYZE_WORK_HOURS_START=0

# 工作时间结束（24小时制）
RYZE_WORK_HOURS_END=6

# ========== 状态服务器配置 ==========
# 状态监控端口
RYZE_STATUS_PORT=9090
```

## 配置文件详解

### config.json 结构

```json
{
  "project": {
    "name": "Ryze-Data",
    "version": "2.0.0",
    "environment": "${RYZE_ENVIRONMENT:development}"
  },
  
  "ocr": {
    "model": "marker",
    "batch_size": "${RYZE_BATCH_SIZE:10}",
    "confidence_threshold": 0.8,
    "language": "en",
    "gpu_enabled": "${RYZE_GPU_ENABLED:true}",
    "gpu_memory_limit": 0.5,
    "timeout_seconds": 300
  },
  // 注意：ocr.model 可选值见下方 "OCR 模型选择" 小节
  
  "paths": {
    "data_root": "${RYZE_DATA_ROOT:./data}",
    "logs_dir": "${RYZE_LOGS_DIR:./logs}",
    // ... 更多路径配置
  },
  
  "processing": {
    "parallel_workers": "${RYZE_NUM_WORKERS:4}",
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "quality_threshold": "${RYZE_QUALITY_THRESHOLD:2.5}",
    "qa_ratio": "${RYZE_QA_RATIO:8}",
    "max_papers": "${RYZE_MAX_PAPERS:1000}"
  }
}
```

### 环境变量引用语法

配置文件支持环境变量引用：

```json
"field": "${ENV_VAR:default_value}"
```

- `ENV_VAR`: 环境变量名
- `default_value`: 默认值（可选）

示例：
```json
"batch_size": "${RYZE_BATCH_SIZE:10}"  // 使用 RYZE_BATCH_SIZE，默认 10
"model": "${RYZE_MODEL:gpt-4}"         // 使用 RYZE_MODEL，默认 gpt-4
```

## 配置优先级

配置按以下优先级加载（高到低）：

1. **命令行参数**
   ```bash
   python -m src.cli.main --config custom.json ocr --batch-size 20
   ```

2. **环境变量**
   ```bash
   export RYZE_BATCH_SIZE=15
   ```

3. **配置文件**
   ```json
   "batch_size": 10
   ```

4. **默认值**
   ```python
   batch_size: int = 5  # 代码中的默认值
   ```

## 常用配置场景

### 1. 开发环境配置

```bash
# .env.dev
RYZE_ENVIRONMENT=development
RYZE_DATA_ROOT=./test_data
RYZE_NUM_WORKERS=1
RYZE_BATCH_SIZE=2
RYZE_MAX_PAPERS=10
RYZE_GPU_ENABLED=false
RYZE_LOG_LEVEL=DEBUG
```

### 2. 生产环境配置

```bash
# .env.prod
RYZE_ENVIRONMENT=production
RYZE_DATA_ROOT=/data/ryze
RYZE_NUM_WORKERS=16
RYZE_BATCH_SIZE=50
RYZE_MAX_PAPERS=10000
RYZE_GPU_ENABLED=true
RYZE_LOG_LEVEL=INFO
```

### 3. 测试环境配置

```bash
# .env.test
RYZE_ENVIRONMENT=test
RYZE_DATA_ROOT=./tests/data
RYZE_NUM_WORKERS=1
RYZE_BATCH_SIZE=1
RYZE_MAX_PAPERS=1
RYZE_QA_RATIO=2
RYZE_GPU_ENABLED=false
OPENAI_BASE_URL=http://localhost:8080/v1  # Mock 服务器
```

### 4. GPU 加速配置

```bash
# GPU 相关配置
RYZE_GPU_ENABLED=true
RYZE_GPU_MEMORY_LIMIT=0.8  # 使用 80% GPU 内存
RYZE_BATCH_SIZE=20         # 增大批处理
RYZE_NUM_WORKERS=8         # 增加并行度
```

### 5. OCR 模型选择

Ryze-Data 内置多种 OCR 模型，通过 `--ocr-model` 命令行参数或配置文件中的 `ocr.model` 字段选择：

```bash
# 查看所有可用模型及安装状态
uv run python -m src.cli.main list-ocr-models

# 使用指定模型运行 OCR
uv run python -m src.cli.main ocr \
    --input-dir data/pdfs \
    --output-dir data/ocr_results \
    --ocr-model deepseek-ocr
```

#### 已注册模型

| 模型名称 | `--ocr-model` 值 | 依赖 | 说明 |
|----------|------------------|------|------|
| Marker | `marker` (默认) | `marker` CLI | 基于 CLI 的 PDF 转换，支持多 GPU 批处理 |
| DeepSeek-OCR v1 | `deepseek-ocr` | `torch`, `transformers` | 本地 HuggingFace 推理，640px 输入 |
| DeepSeek-OCR v2 | `deepseek-ocr-v2` | `torch`, `transformers` | 本地 HuggingFace 推理，768px 输入 |
| MarkItDown | `markitdown` | `markitdown` | Microsoft MarkItDown PDF 转 Markdown |

#### 安装 DeepSeek-OCR 依赖

DeepSeek-OCR v1/v2 需要额外的 GPU 依赖：

```bash
# 安装可选依赖组
uv sync --extra deepseek-ocr

# 或手动安装
uv add torch transformers flash-attn einops
```

> **硬件要求**：DeepSeek-OCR 模型约需 6GB 显存 (bfloat16)，建议使用支持 flash_attention_2 的 GPU 以获得最佳性能。

### 6. 独立 OCR 预处理脚本

除了通过 CLI 运行 OCR，还可以使用 `scripts/utils/` 下的独立脚本对 HuggingFace 数据集进行批量 OCR 预处理。每个模型拥有独立的虚拟环境：

```bash
# 设置模型环境
cd scripts/utils/markitdown && bash setup_env.sh

# 运行 OCR（ArxivQA，5 个样本）
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5

# Marker 支持流水线并行和 GPU 选择
cd ../marker && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --workers 4 --gpu cpu

# DeepSeek 模型需要指定 GPU
cd ../deepseek_ocr_v1 && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 0
```

**通用 CLI 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | `arxivqa` 或 `slidevqa` |
| `--output-dir` | `data/ocr_precompute/{model}/{dataset}` | 输出目录 |
| `--cache-dir` | `data/benchmark_data` | 共享图像缓存目录 |
| `--max-samples` | `0`（全部） | 最大样本数 |
| `--hf-endpoint` | 未设置 | HuggingFace 镜像地址（如 `https://hf-mirror.com`） |

**模型专有参数：**

| 模型 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Marker | `--workers` | `0` | 流水线并行 worker 数量（`0`=自动） |
| Marker | `--gpu` | 未设置 | 设置 `CUDA_VISIBLE_DEVICES`（如 `0`、`0,1` 或 `cpu`） |
| DeepSeek v1/v2 | `--gpu` | `0` | GPU 设备 ID |
| DeepSeek v1/v2 | `--backend` | `transformers` | 推理后端：`auto`/`vllm`/`transformers` |

### 7. 分布式处理配置

```bash
# 多服务器下载
RYZE_DOWNLOAD_SERVERS=server1.example.com,server2.example.com,server3.example.com
RYZE_WORK_HOURS_START=22  # 晚上 10 点开始
RYZE_WORK_HOURS_END=6     # 早上 6 点结束

# 分布式处理
RYZE_NUM_WORKERS=32
RYZE_BATCH_SIZE=100
```

## 高级配置

### 1. 自定义 API 端点

使用自己的 API 服务器或代理：

```bash
# 使用 Azure OpenAI
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-azure-api-key

# 使用本地 LLM 服务
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
RYZE_LLM_MODEL=llama2:70b
```

### 2. 代理配置

```bash
# HTTP 代理
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# SOCKS 代理
ALL_PROXY=socks5://127.0.0.1:1080
```

### 3. 资源限制

```bash
# 内存限制
RYZE_MAX_MEMORY_GB=16

# CPU 限制
RYZE_CPU_CORES=8

# 磁盘缓存
RYZE_CACHE_SIZE_GB=100
```

### 4. 日志配置

```bash
# 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
RYZE_LOG_LEVEL=INFO

# 日志格式：json, text
RYZE_LOG_FORMAT=json

# 日志文件轮转
RYZE_LOG_MAX_SIZE_MB=100
RYZE_LOG_BACKUP_COUNT=10
```

### 5. 重试策略

```json
{
  "processing": {
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "retry_backoff_factor": 2,
    "retry_max_delay_seconds": 300
  }
}
```

## 配置验证

### 验证配置有效性

```python
from src.config_manager import ConfigManager

config = ConfigManager()
config.load()

if config.validate():
    print("✅ 配置有效")
else:
    print("❌ 配置无效")
```

### 常见验证项

1. **路径存在性**
   - 数据目录是否可访问
   - 日志目录是否可写

2. **API 密钥**
   - 必需的 API 密钥是否设置
   - API 端点是否可达

3. **数值范围**
   - 置信度阈值：0-1
   - 质量阈值：0-5
   - 工作线程数：>=1

4. **依赖检查**
   - GPU 是否可用（如果启用）
   - 必需的 Python 包是否安装

### 配置检查命令

```bash
# 显示当前配置
python -m src.cli.main config-show

# 验证配置
python -c "from src.config_manager import config; config.load(); print(config.validate())"

# 检查环境变量
env | grep RYZE_
```

## 配置最佳实践

### 1. 使用环境文件

不同环境使用不同的配置文件：

```bash
# 开发（运行爬虫）
python -m src.cli.main --env .env.dev scrape --max-pages 10

# 生产（运行 OCR）
python -m src.cli.main --env .env.prod ocr --pdf-dir /data/pdfs
```

### 2. 敏感信息管理

- **永远不要**将 API 密钥提交到版本控制
- 使用环境变量存储敏感信息
- 使用密钥管理服务（如 AWS Secrets Manager）

### 3. 配置文档化

在 `.env.example` 中记录所有配置项：

```bash
# API 密钥（必需）
# 获取方式：https://platform.openai.com/api-keys
OPENAI_API_KEY=your-api-key-here

# 批处理大小（可选，默认 10）
# 增大可提高效率，但需要更多内存
RYZE_BATCH_SIZE=10
```

### 4. 配置版本控制

```json
{
  "project": {
    "config_version": "2.0.0",
    "min_config_version": "1.5.0"
  }
}
```

### 5. 配置备份

定期备份重要配置：

```bash
# 备份配置
cp config.json config.json.bak
cp .env .env.bak

# 恢复配置
cp config.json.bak config.json
cp .env.bak .env
```

## 故障排除

### 配置不生效

1. 检查环境变量是否正确设置：
   ```bash
   echo $RYZE_BATCH_SIZE
   ```

2. 检查配置文件路径：
   ```bash
   python -m src.cli.main --config ./config.json config-show
   ```

3. 检查配置优先级冲突

### API 密钥错误

1. 验证密钥格式：
   ```bash
   echo $OPENAI_API_KEY | wc -c  # 应该是 51 个字符
   ```

2. 测试 API 连接：
   ```python
   import openai
   client = openai.OpenAI()
   client.models.list()
   ```

### 路径权限问题

1. 检查目录权限：
   ```bash
   ls -la ./data
   ```

2. 创建必需的目录：
   ```bash
   mkdir -p ./data ./logs
   chmod 755 ./data ./logs
   ```
