# 配置指南

## 目录

- [配置概述](#配置概述)
- [快速配置](#快速配置)
- [环境变量配置](#环境变量配置)
- [配置文件详解](#配置文件详解)
- [配置优先级](#配置优先级)
- [常用配置场景](#常用配置场景)
- [高级配置](#高级配置)
- [配置验证](#配置验证)
- [配置最佳实践](#配置最佳实践)
- [故障排除](#故障排除)

## 配置概述

Ryze-Data 采用灵活的多层次配置管理系统，支持多种配置方式以适应不同的部署环境和使用场景。

### 配置层次

1. **默认值**：代码中的硬编码默认值
2. **配置文件**：JSON格式的配置文件（config.json）
3. **环境变量**：系统环境变量或.env文件
4. **命令行参数**：运行时传入的参数

### 配置特性

- **热重载**：支持配置文件的热更新
- **环境隔离**：不同环境使用不同配置
- **类型安全**：自动类型转换和验证
- **密钥管理**：敏感信息加密存储
- **配置继承**：支持配置文件继承和覆盖

## 快速配置

### 最小配置示例

```bash
# 1. 复制配置模板
cp .env.example .env
cp config.example.json config.json

# 2. 编辑必要的环境变量
cat > .env << EOF
# API密钥（必需）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# 数据路径（必需）
RYZE_DATA_ROOT=./data

# 基本设置（可选）
RYZE_NUM_WORKERS=4
RYZE_BATCH_SIZE=10
EOF

# 3. 验证配置
python -m src.cli.main config-show

# 4. 测试运行（运行爬虫）
python -m src.cli.main scrape --max-pages 1
```

### 常用配置项

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|---------|--------|------|
| API密钥 | `OPENAI_API_KEY` | 无 | OpenAI API密钥 |
| 数据根目录 | `RYZE_DATA_ROOT` | ./data | 数据存储根目录 |
| 工作线程数 | `RYZE_NUM_WORKERS` | 4 | 并行处理线程数 |
| 批处理大小 | `RYZE_BATCH_SIZE` | 10 | 批量处理大小 |
| GPU启用 | `RYZE_GPU_ENABLED` | true | 是否使用GPU |
| 日志级别 | `RYZE_LOG_LEVEL` | INFO | 日志详细程度 |

## 环境变量配置

### 基础配置 (.env)

```bash
# ========================================
# 基础路径配置
# ========================================

# 数据根目录 - 所有数据文件的基础路径
RYZE_DATA_ROOT=./data

# 日志目录 - 存储所有日志文件
RYZE_LOGS_DIR=./logs

# 临时文件目录 - 处理过程中的临时文件
RYZE_TEMP_DIR=/tmp/ryze

# ========================================
# 数据路径配置
# ========================================

# Nature元数据存储路径
RYZE_NATURE_DATA=${RYZE_DATA_ROOT}/nature_metadata

# PDF文件存储路径
RYZE_PDF_DIR=${RYZE_DATA_ROOT}/pdfs

# OCR输出路径
RYZE_OCR_OUTPUT=${RYZE_DATA_ROOT}/ocr_results

# 摘要存储路径
RYZE_ABSTRACT_DIR=${RYZE_DATA_ROOT}/abstracts

# 图片存储路径
RYZE_FIGURES_DIR=${RYZE_DATA_ROOT}/figures

# 视觉预处理路径
RYZE_VLM_PREPROCESSING=${RYZE_DATA_ROOT}/vlm_preprocessing

# 文本QA数据路径
RYZE_SFT_DATA=${RYZE_DATA_ROOT}/sft_data

# 视觉QA数据路径
RYZE_VLM_SFT_DATA=${RYZE_DATA_ROOT}/vlm_sft_data

# 提示词模板路径
RYZE_PROMPTS_DIR=./prompts
```

### API配置

```bash
# ========================================
# API密钥配置
# ========================================

# OpenAI API配置
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ORG_ID=org-xxxxxxxxxxxxxxxxxxxxxxxx

# 其他API服务（如果使用）
ANTHROPIC_API_KEY=your-anthropic-api-key
COHERE_API_KEY=your-cohere-api-key
HUGGINGFACE_TOKEN=your-huggingface-token

# 自定义API端点
RYZE_PARSING_API_KEY=your-parsing-api-key
RYZE_QA_API_KEY=your-qa-api-key
RYZE_VISION_API_KEY=your-vision-api-key
```

### 模型配置

```bash
# ========================================
# LLM模型配置
# ========================================

# 文本处理模型
RYZE_LLM_MODEL=gpt-4
RYZE_LLM_TEMPERATURE=0.7
RYZE_LLM_MAX_TOKENS=2048

# 视觉处理模型
RYZE_VISION_MODEL=gpt-4-vision-preview
RYZE_VISION_MAX_TOKENS=4096

# 嵌入模型
RYZE_EMBEDDING_MODEL=text-embedding-ada-002

# QA生成参数
RYZE_QA_RATIO=8              # 每个单元生成的QA对数
RYZE_QUALITY_THRESHOLD=2.5   # 质量阈值（0-5分）
RYZE_MIN_ANSWER_LENGTH=20    # 最小答案长度
RYZE_MAX_ANSWER_LENGTH=500   # 最大答案长度
```

### 处理配置

```bash
# ========================================
# 处理配置
# ========================================

# 并行处理
RYZE_NUM_WORKERS=4            # 并行工作线程数
RYZE_BATCH_SIZE=10           # 批处理大小
RYZE_QUEUE_SIZE=100          # 任务队列大小

# GPU配置
RYZE_GPU_ENABLED=true        # 是否启用GPU
RYZE_GPU_MEMORY_LIMIT=0.8    # GPU内存使用限制（比例）
RYZE_GPU_DEVICE_ID=0         # 使用的GPU设备ID

# 超时和重试
RYZE_TIMEOUT_SECONDS=300     # 单个任务超时时间
RYZE_MAX_RETRIES=3          # 最大重试次数
RYZE_RETRY_DELAY=5          # 重试延迟（秒）
RYZE_RETRY_BACKOFF=2        # 重试退避因子

# 资源限制
RYZE_MAX_PAPERS=1000        # 最大处理文章数
RYZE_MAX_MEMORY_GB=16       # 最大内存使用（GB）
RYZE_MAX_DISK_GB=100        # 最大磁盘使用（GB）
```

### 网络配置

```bash
# ========================================
# 网络配置
# ========================================

# HTTP代理
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1,*.local

# 请求配置
RYZE_REQUEST_TIMEOUT=30      # 请求超时（秒）
RYZE_CONNECTION_POOLSIZE=10  # 连接池大小
RYZE_MAX_REDIRECTS=5        # 最大重定向次数

# 速率限制
RYZE_RATE_LIMIT=1.0         # 每秒请求数
RYZE_BURST_SIZE=10          # 突发请求数
```

### 日志配置

```bash
# ========================================
# 日志配置
# ========================================

# 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
RYZE_LOG_LEVEL=INFO

# 日志格式：json, text, structured
RYZE_LOG_FORMAT=json

# 日志输出：file, console, both
RYZE_LOG_OUTPUT=both

# 日志文件设置
RYZE_LOG_MAX_SIZE_MB=100    # 单个日志文件最大大小
RYZE_LOG_BACKUP_COUNT=10    # 保留的日志文件数量
RYZE_LOG_ROTATION=daily     # 日志轮转策略：daily, size, time

# 性能日志
RYZE_ENABLE_PROFILING=false # 是否启用性能分析
RYZE_PROFILE_OUTPUT=./profiles # 性能分析输出目录
```

## 配置文件详解

### config.json 结构

```json
{
  "project": {
    "name": "Ryze-Data",
    "version": "2.0.0",
    "environment": "${RYZE_ENVIRONMENT:development}",
    "description": "Scientific Literature Processing Pipeline"
  },
  
  "ocr": {
    "model": "marker",
    "batch_size": "${RYZE_BATCH_SIZE:10}",
    "confidence_threshold": 0.8,
    "language": "en",
    "gpu_enabled": "${RYZE_GPU_ENABLED:true}",
    "gpu_memory_limit": 0.5,
    "timeout_seconds": 300,
    "output_format": ["markdown", "json"],
    "preserve_layout": true,
    "extract_tables": true,
    "extract_figures": true
  },
  
  "paths": {
    "data_root": "${RYZE_DATA_ROOT:./data}",
    "logs_dir": "${RYZE_LOGS_DIR:./logs}",
    "nature_data": "${RYZE_NATURE_DATA:./data/nature_metadata}",
    "pdf_dir": "${RYZE_PDF_DIR:./data/pdfs}",
    "ocr_output": "${RYZE_OCR_OUTPUT:./data/ocr_results}",
    "abstract_dir": "${RYZE_ABSTRACT_DIR:./data/abstracts}",
    "figures_dir": "${RYZE_FIGURES_DIR:./data/figures}",
    "vlm_preprocessing": "${RYZE_VLM_PREPROCESSING:./data/vlm_preprocessing}",
    "sft_data": "${RYZE_SFT_DATA:./data/sft_data}",
    "vlm_sft_data": "${RYZE_VLM_SFT_DATA:./data/vlm_sft_data}",
    "prompts_dir": "${RYZE_PROMPTS_DIR:./prompts}",
    "temp_dir": "${RYZE_TEMP_DIR:/tmp/ryze}"
  },
  
  "processing": {
    "parallel_workers": "${RYZE_NUM_WORKERS:4}",
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "retry_backoff_factor": 2,
    "quality_threshold": "${RYZE_QUALITY_THRESHOLD:2.5}",
    "qa_ratio": "${RYZE_QA_RATIO:8}",
    "max_papers": "${RYZE_MAX_PAPERS:1000}",
    "checkpoint_enabled": true,
    "checkpoint_interval": 100,
    "memory_limit_gb": 16,
    "disk_limit_gb": 100
  },
  
  "scrapers": {
    "nature": {
      "enabled": true,
      "base_url": "https://www.nature.com",
      "rate_limit": 1.0,
      "timeout": 30,
      "max_pages": 100,
      "filters": {
        "start_date": "2020-01-01",
        "end_date": null,
        "subjects": ["physics", "chemistry", "biology"],
        "open_access_only": false
      }
    },
    "arxiv": {
      "enabled": false,
      "base_url": "https://arxiv.org",
      "rate_limit": 0.5,
      "categories": ["cs.AI", "cs.LG", "physics.comp-ph"]
    }
  },
  
  "models": {
    "parsing": {
      "provider": "openai",
      "model": "${RYZE_PARSING_MODEL:gpt-4}",
      "api_endpoint": "${OPENAI_BASE_URL:https://api.openai.com/v1}",
      "api_key_env": "OPENAI_API_KEY",
      "max_tokens": 2048,
      "temperature": 0.3,
      "timeout": 60
    },
    "qa_generation": {
      "provider": "openai",
      "model": "${RYZE_QA_MODEL:gpt-4}",
      "api_endpoint": "${OPENAI_BASE_URL:https://api.openai.com/v1}",
      "api_key_env": "OPENAI_API_KEY",
      "max_tokens": 2048,
      "temperature": 0.7,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.0,
      "batch_inference": true,
      "batch_size": 20
    },
    "vision": {
      "provider": "openai",
      "model": "${RYZE_VISION_MODEL:gpt-4-vision-preview}",
      "api_endpoint": "${OPENAI_BASE_URL:https://api.openai.com/v1}",
      "api_key_env": "OPENAI_API_KEY",
      "max_tokens": 4096,
      "temperature": 0.5,
      "image_detail": "high"
    }
  },
  
  "qa_templates": {
    "version": "v2.0",
    "templates_dir": "${RYZE_PROMPTS_DIR:./prompts}",
    "enabled_types": [
      "factual",
      "conceptual",
      "mechanism",
      "application",
      "comparison",
      "visual-analysis",
      "visual-data"
    ],
    "difficulty_distribution": {
      "easy": 0.3,
      "medium": 0.5,
      "hard": 0.2
    }
  },
  
  "output_formats": {
    "markdown": true,
    "json": true,
    "bibtex": true,
    "include_images": true,
    "compress_images": false,
    "image_format": "jpeg",
    "image_quality": 85
  },
  
  "monitoring": {
    "enable_metrics": true,
    "metrics_port": 9090,
    "metrics_endpoint": "/metrics",
    "log_level": "${RYZE_LOG_LEVEL:INFO}",
    "log_format": "${RYZE_LOG_FORMAT:json}",
    "enable_tracing": false,
    "tracing_endpoint": "http://localhost:14268/api/traces"
  },
  
  "cache": {
    "enabled": true,
    "type": "disk",
    "path": "${RYZE_TEMP_DIR}/cache",
    "max_size_gb": 10,
    "ttl_hours": 24,
    "compression": true
  },
  
  "servers": {
    "download_servers": [
      "server1.example.com",
      "server2.example.com"
    ],
    "work_hours": {
      "start": 0,
      "end": 6,
      "timezone": "UTC"
    }
  }
}
```

### 环境变量引用语法

配置文件支持环境变量引用，语法如下：

```json
"field": "${ENV_VAR:default_value}"
```

- `ENV_VAR`：环境变量名
- `default_value`：默认值（可选）

**示例**：
```json
{
  "batch_size": "${RYZE_BATCH_SIZE:10}",     // 使用RYZE_BATCH_SIZE，默认10
  "model": "${RYZE_MODEL:gpt-4}",            // 使用RYZE_MODEL，默认gpt-4
  "enabled": "${FEATURE_ENABLED:true}",      // 布尔值
  "timeout": "${TIMEOUT:300}"                // 数字值
}
```

### 配置继承

支持配置文件继承，实现配置复用：

```json
// base.json - 基础配置
{
  "extends": null,
  "project": {
    "name": "Ryze-Data"
  }
}

// dev.json - 开发环境配置
{
  "extends": "base.json",
  "project": {
    "environment": "development"
  }
}

// prod.json - 生产环境配置
{
  "extends": "base.json",
  "project": {
    "environment": "production"
  }
}
```

## 配置优先级

配置按以下优先级加载（高到低）：

1. **命令行参数**（最高优先级）
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

4. **默认值**（最低优先级）
   ```python
   batch_size: int = 5  # 代码中的默认值
   ```

### 优先级示例

```python
# 默认值：5
# config.json: 10
# 环境变量 RYZE_BATCH_SIZE=15
# 命令行 --batch-size 20

# 最终值：20（命令行参数优先级最高）
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
RYZE_LOG_FORMAT=text
RYZE_ENABLE_PROFILING=true

# 快速测试配置
RYZE_SKIP_DOWNLOAD=true     # 跳过下载阶段
RYZE_USE_CACHED_OCR=true    # 使用缓存的OCR结果
RYZE_DRY_RUN=false          # 正常运行
```

使用开发配置：
```bash
export ENV_FILE=.env.dev
python -m src.cli.main pipeline
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
RYZE_LOG_FORMAT=json

# 生产优化
RYZE_ENABLE_METRICS=true
RYZE_CACHE_ENABLED=true
RYZE_CHECKPOINT_ENABLED=true
RYZE_ERROR_RECOVERY=true
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
RYZE_LOG_LEVEL=WARNING

# 使用Mock服务
OPENAI_BASE_URL=http://localhost:8080/v1
RYZE_USE_MOCK_API=true
```

### 4. GPU加速配置

```bash
# GPU相关配置
RYZE_GPU_ENABLED=true
RYZE_GPU_MEMORY_LIMIT=0.8    # 使用80% GPU内存
RYZE_GPU_DEVICE_ID=0         # 使用GPU 0
CUDA_VISIBLE_DEVICES=0,1     # 可见GPU设备

# 优化参数
RYZE_BATCH_SIZE=20           # 增大批处理
RYZE_NUM_WORKERS=8           # 增加并行度
RYZE_USE_FP16=true          # 使用半精度计算
```

### 5. 独立 OCR 预处理脚本

除了通过 CLI 运行 OCR，还可以使用 `scripts/utils/` 下的独立脚本对 HuggingFace 数据集进行批量 OCR 预处理。每个模型拥有独立的虚拟环境：

```bash
# 设置模型环境
cd scripts/utils/markitdown && bash setup_env.sh

# 运行 OCR（ArxivQA，5 个样本）
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5

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
| `--gpu` | `0` | GPU 设备 ID（仅 DeepSeek） |

### 6. 分布式处理配置

```bash
# 主节点配置
RYZE_ROLE=master
RYZE_MASTER_HOST=0.0.0.0
RYZE_MASTER_PORT=5000
RYZE_WORKER_NODES=worker1,worker2,worker3

# 工作节点配置
RYZE_ROLE=worker
RYZE_MASTER_HOST=master.example.com
RYZE_MASTER_PORT=5000
RYZE_WORKER_ID=worker1

# 共享存储
RYZE_SHARED_STORAGE=/nfs/ryze-data
RYZE_USE_S3=true
AWS_S3_BUCKET=ryze-data-bucket
```

### 6. 容器化配置

```dockerfile
# Dockerfile环境变量
ENV RYZE_DATA_ROOT=/app/data
ENV RYZE_LOGS_DIR=/app/logs
ENV RYZE_CONFIG_PATH=/app/config/config.json
ENV PYTHONPATH=/app

# docker-compose.yml
version: '3.8'
services:
  ryze:
    image: ryze-data:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RYZE_NUM_WORKERS=4
      - RYZE_GPU_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./config:/app/config
```

## 高级配置

### 1. 动态配置更新

```python
# 配置热重载
from src.config_manager import config

# 监听配置变化
config.watch_changes(callback=on_config_change)

# 手动重载
config.reload()

# 运行时修改
config.processing.batch_size = 20
config.save()  # 保存到文件
```

### 2. 配置验证规则

```json
{
  "validation": {
    "rules": {
      "batch_size": {
        "type": "integer",
        "min": 1,
        "max": 100
      },
      "quality_threshold": {
        "type": "float",
        "min": 0.0,
        "max": 5.0
      },
      "api_key": {
        "type": "string",
        "pattern": "^sk-[a-zA-Z0-9]{48}$",
        "required": true
      }
    }
  }
}
```

### 3. 条件配置

```json
{
  "processing": {
    "batch_size": {
      "$if": "${RYZE_GPU_ENABLED}",
      "$then": 50,
      "$else": 10
    }
  }
}
```

### 4. 配置模板

```bash
# 使用Jinja2模板
cat > config.j2 << EOF
{
  "workers": {{ NUM_WORKERS | default(4) }},
  "gpu": {% if GPU_ENABLED %}true{% else %}false{% endif %},
  "servers": [
    {% for server in SERVERS.split(',') %}
    "{{ server }}"{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
EOF

# 渲染配置
j2 config.j2 > config.json
```

### 5. 密钥管理

```bash
# 使用密钥管理服务
# AWS Secrets Manager
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id ryze/openai-api-key \
  --query SecretString --output text)

# HashiCorp Vault
export OPENAI_API_KEY=$(vault kv get -field=api_key secret/ryze/openai)

# Azure Key Vault
export OPENAI_API_KEY=$(az keyvault secret show \
  --vault-name ryze-vault \
  --name openai-api-key \
  --query value -o tsv)
```

## 配置验证

### 验证命令

```bash
# 显示当前配置
python -m src.cli.main config-show

# 验证配置有效性
python -m src.cli.main config-validate

# 测试特定配置
python -m src.cli.main config-test --component ocr

# 导出配置
python -m src.cli.main config-export --format json > config-export.json
```

### 编程验证

```python
from src.config_manager import ConfigManager

config = ConfigManager()
config.load()

# 验证配置
if config.validate():
    print("✅ 配置有效")
else:
    print("❌ 配置无效")
    for error in config.get_errors():
        print(f"  - {error}")

# 检查特定配置
if not config.parsing_model.api_key:
    raise ValueError("API密钥未设置")

if config.processing.batch_size < 1:
    raise ValueError("批处理大小必须大于0")
```

### 常见验证项

1. **路径存在性**
   ```python
   # 检查必要目录
   for path in [config.paths.data_root, config.paths.logs_dir]:
       if not Path(path).exists():
           Path(path).mkdir(parents=True)
   ```

2. **API密钥格式**
   ```python
   # 验证OpenAI密钥格式
   import re
   api_key = config.parsing_model.api_key
   if not re.match(r'^sk-[a-zA-Z0-9]{48}$', api_key):
       raise ValueError("无效的OpenAI API密钥格式")
   ```

3. **数值范围**
   ```python
   # 验证数值配置
   assert 0 < config.processing.batch_size <= 100
   assert 0 <= config.ocr.confidence_threshold <= 1
   assert config.processing.parallel_workers >= 1
   ```

4. **依赖检查**
   ```python
   # GPU可用性
   if config.ocr.gpu_enabled:
       import torch
       if not torch.cuda.is_available():
           print("警告：GPU已启用但不可用")
   ```

## 配置最佳实践

### 1. 环境分离

```bash
# 使用不同的配置文件
config/
├── base.json         # 基础配置
├── development.json  # 开发环境
├── testing.json      # 测试环境
├── staging.json      # 预发布环境
└── production.json   # 生产环境

# 加载特定环境
python -m src.cli.main --config config/production.json
```

### 2. 敏感信息管理

```bash
# .env文件（不提交到版本控制）
OPENAI_API_KEY=sk-real-key-here
DATABASE_PASSWORD=actual-password

# .env.example（提交到版本控制）
OPENAI_API_KEY=sk-your-api-key-here
DATABASE_PASSWORD=your-password-here
```

### 3. 配置文档化

```bash
# config.example.json带注释
{
  "batch_size": 10,  // 批处理大小，增大可提高效率但需要更多内存
  "workers": 4,      // 并行工作线程数，建议设置为CPU核心数
  "gpu": true       // 是否启用GPU，需要CUDA支持
}
```

### 4. 配置版本控制

```json
{
  "config_version": "2.0.0",
  "min_version": "1.5.0",
  "migrations": {
    "1.5.0": "migrate_v1_5.py",
    "2.0.0": "migrate_v2_0.py"
  }
}
```

### 5. 配置备份

```bash
# 自动备份脚本
#!/bin/bash
BACKUP_DIR="./config-backups"
mkdir -p $BACKUP_DIR

# 备份当前配置
cp config.json "$BACKUP_DIR/config-$(date +%Y%m%d-%H%M%S).json"
cp .env "$BACKUP_DIR/env-$(date +%Y%m%d-%H%M%S)"

# 保留最近10个备份
ls -t $BACKUP_DIR/config-*.json | tail -n +11 | xargs -r rm
ls -t $BACKUP_DIR/env-* | tail -n +11 | xargs -r rm
```

## 故障排除

### 配置加载失败

**问题**：配置文件无法加载
```
Error: Failed to load config from config.json
```

**解决方案**：
```bash
# 检查文件是否存在
ls -la config.json

# 验证JSON格式
python -m json.tool config.json

# 检查文件权限
chmod 644 config.json

# 使用默认配置
cp config.example.json config.json
```

### 环境变量未生效

**问题**：设置的环境变量没有被识别

**解决方案**：
```bash
# 确认环境变量已设置
echo $RYZE_BATCH_SIZE

# 重新加载.env文件
source .env

# 或使用dotenv加载
python -c "from dotenv import load_dotenv; load_dotenv()"

# 直接在命令中指定
RYZE_BATCH_SIZE=20 python -m src.cli.main pipeline
```

### API密钥错误

**问题**：API密钥无效或未设置
```
openai.AuthenticationError: Invalid API key
```

**解决方案**：
```bash
# 验证密钥格式
echo $OPENAI_API_KEY | wc -c  # 应该是51个字符

# 测试API连接
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 使用配置文件设置
{
  "models": {
    "parsing": {
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

### 路径权限问题

**问题**：无法创建或访问配置的路径
```
PermissionError: [Errno 13] Permission denied: '/data/logs'
```

**解决方案**：
```bash
# 创建目录并设置权限
sudo mkdir -p /data/logs
sudo chown -R $USER:$USER /data
chmod -R 755 /data

# 或使用用户目录
export RYZE_DATA_ROOT=~/ryze-data
export RYZE_LOGS_DIR=~/ryze-data/logs
```

### 配置冲突

**问题**：不同配置源的值冲突

**诊断步骤**：
```python
# 查看配置来源
from src.config_manager import config

config.debug_sources()  # 显示每个配置值的来源

# 输出示例：
# batch_size: 20 (source: CLI)
# workers: 8 (source: ENV)
# gpu_enabled: true (source: CONFIG_FILE)
# timeout: 300 (source: DEFAULT)
```

### 性能问题

**问题**：配置不当导致性能问题

**优化建议**：
```bash
# CPU密集型任务
RYZE_NUM_WORKERS=$(nproc)  # 设置为CPU核心数

# 内存受限环境
RYZE_BATCH_SIZE=5          # 减小批处理大小
RYZE_CACHE_ENABLED=false   # 禁用缓存

# I/O密集型任务
RYZE_NUM_WORKERS=$(($(nproc) * 2))  # 设置为CPU核心数的2倍

# GPU优化
RYZE_GPU_MEMORY_LIMIT=0.9  # 使用90% GPU内存
RYZE_USE_FP16=true         # 使用半精度
```

## 配置示例集合

### 最小化配置

```json
{
  "paths": {
    "data_root": "./data"
  },
  "models": {
    "qa_generation": {
      "api_key_env": "OPENAI_API_KEY"
    }
  }
}
```

### 完整生产配置

```json
{
  "project": {
    "name": "Ryze-Data-Production",
    "version": "2.0.0",
    "environment": "production"
  },
  "processing": {
    "parallel_workers": 16,
    "batch_size": 50,
    "max_papers": 10000,
    "checkpoint_enabled": true,
    "error_recovery": true
  },
  "monitoring": {
    "enable_metrics": true,
    "enable_tracing": true,
    "alert_endpoints": [
      "https://alerts.example.com/webhook"
    ]
  },
  "security": {
    "encrypt_api_keys": true,
    "audit_logging": true,
    "data_retention_days": 90
  }
}
```