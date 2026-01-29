# Ryze-Data 架构设计文档

## 目录

- [系统概述](#系统概述)
- [架构原则](#架构原则)
- [项目结构](#项目结构)
- [系统架构](#系统架构)
- [核心模块](#核心模块)
- [数据流设计](#数据流设计)
- [技术栈](#技术栈)
- [扩展性设计](#扩展性设计)

## 系统概述

Ryze-Data 是一个模块化、可扩展的科学文献处理框架，采用流水线架构设计，将复杂的文献处理任务分解为独立的处理阶段。

### 设计目标

1. **模块化**：各组件独立开发、测试和部署
2. **可扩展**：易于添加新的数据源和处理步骤
3. **高效性**：支持并行处理和分布式执行
4. **可靠性**：错误恢复和断点续传机制
5. **可观测**：完整的日志和监控支持

## 架构原则

### 1. 单一职责原则
每个模块负责一个明确的功能：
- **Scraper**：数据爬取
- **OCR Processor**：文档 OCR 处理
- **API Balancer**：LLM API 负载均衡

### 2. 依赖倒置原则
- 核心业务逻辑不依赖具体实现
- 通过接口和抽象类定义契约
- 配置驱动的依赖注入

### 3. 开闭原则
- 对扩展开放：易于添加新的处理器
- 对修改关闭：核心流程稳定不变

## 项目结构

```
Ryze-Data/
├── pyproject.toml               # 项目配置 (uv)
├── config.json                  # 运行时配置
├── .env                         # 环境变量（不提交）
├── requirements.txt             # Python 依赖（兼容 pip）
├── pytest.ini                   # Pytest 配置
├── README.md                    # 项目文档
├── LICENSE                      # AGPL-3.0 许可证
│
├── src/                         # 源代码目录
│   ├── __init__.py
│   ├── config_manager.py        # 配置管理（支持环境变量）
│   ├── pipeline_manager.py      # 流水线编排和执行
│   ├── api_key_balancer.py      # LLM API 负载均衡器
│   ├── chunked-ocr.py           # 分块 OCR 处理
│   │
│   ├── cli/                     # 命令行界面
│   │   ├── __init__.py
│   │   ├── main.py              # CLI 主入口
│   │   └── data_inspector.py    # 数据检查和采样工具
│   │
│   ├── generators/              # QA 生成器模块
│   │   ├── __init__.py          # 包导出
│   │   ├── base_generator.py    # 抽象基类和 QAPair
│   │   ├── prompt_manager.py    # 提示词模板管理
│   │   ├── text_qa_generator.py # 文本 QA 生成
│   │   └── vision_qa_generator.py # 视觉 QA 生成
│   │
│   └── scrapers/                # 数据源爬虫
│       ├── __init__.py
│       └── nature_scraper.py    # Nature 文章爬虫
│
├── tests/                       # 测试套件
│   ├── conftest.py              # 测试夹具和配置
│   ├── test_generators.py       # QA 生成器测试
│   └── README.md                # 测试文档
│
├── docs/                        # 文档
│   ├── architecture.md          # 架构设计文档
│   ├── api-reference.md         # API 文档
│   ├── configuration.md         # 配置指南
│   ├── data-formats.md          # 数据格式规范
│   ├── development.md           # 开发指南
│   ├── troubleshooting.md       # 故障排除指南
│   └── zh-CN/                   # 中文文档
│
├── prompts/                     # LLM 提示词模板
│   ├── text/                    # 文本 QA 提示词
│   │   ├── factual.txt
│   │   ├── mechanism.txt
│   │   ├── application.txt
│   │   └── quality.txt
│   └── vision/                  # 视觉 QA 提示词
│       ├── visual-factual.txt
│       ├── visual-mechanism.txt
│       ├── visual-analysis.txt
│       ├── visual-comparison.txt
│       ├── visual-data-extraction.txt
│       └── visual-quality.txt
│
├── data/                        # 数据目录（git 忽略）
│   ├── nature_metadata/         # 爬取的元数据
│   ├── pdfs/                    # 下载的 PDF 文件
│   ├── ocr_results/             # OCR 处理结果（Markdown）
│   ├── vlm_preprocessing/       # 图表上下文 JSON
│   ├── sft_data/                # 文本 QA 训练数据
│   └── vlm_sft_data/            # 视觉 QA 训练数据
│
└── data-sample/                 # 测试用样本数据
```

### 文件用途说明

| 文件/目录 | 用途 | 实现状态 |
|-----------|------|----------|
| `src/config_manager.py` | 统一配置管理 | ✅ 已实现 |
| `src/pipeline_manager.py` | 流水线编排逻辑 | ✅ 已实现 |
| `src/api_key_balancer.py` | LLM API 负载均衡 | ✅ 已实现 |
| `src/chunked-ocr.py` | 分块 OCR 处理 | ✅ 已实现 |
| `src/cli/main.py` | CLI 命令实现 | ✅ 已实现 |
| `src/cli/data_inspector.py` | 数据检查工具 | ✅ 已实现 |
| `src/scrapers/nature_scraper.py` | Nature 文章爬虫 | ✅ 已实现 |
| `src/generators/` | QA 生成器模块 | ✅ 已实现 |
| `src/generators/text_qa_generator.py` | 文本 QA 生成 | ✅ 已实现 |
| `src/generators/vision_qa_generator.py` | 视觉 QA 生成 | ✅ 已实现 |
| `tests/` | 完整测试套件 | ✅ 已实现 |
| `docs/` | 技术文档 | ✅ 已实现 |

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     CLI Interface                        │
│                  (src/cli/main.py)                      │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  Pipeline Manager                        │
│              (src/pipeline_manager.py)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • Stage Orchestration                            │   │
│  │ • Dependency Resolution                          │   │
│  │ • Error Handling                                 │   │
│  │ • State Management                               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────┬──────────────────────┬────────────────────────┘
          │                      │
    ┌─────▼────┐           ┌─────▼─────┐
    │ Scrapers │           │ Chunked   │
    │          │           │ OCR       │
    └─────┬────┘           └─────┬─────┘
          │                      │
    ┌─────▼──────────────────────▼────────────────────┐
    │              API Key Balancer                    │
    │           (src/api_key_balancer.py)             │
    │  ┌────────────────────────────────────────┐     │
    │  │ • Multi-key Load Balancing             │     │
    │  │ • Automatic Retry & Fallback           │     │
    │  │ • Rate Limiting                        │     │
    │  │ • Statistics & Monitoring              │     │
    │  └────────────────────────────────────────┘     │
    └─────────────────────┬───────────────────────────┘
                          │
    ┌─────────────────────▼───────────────────────────┐
    │           Configuration Manager                  │
    │         (src/config_manager.py)                 │
    └──────────────┬──────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────┐
    │              Data Storage                        │
    │  ┌────────────────────────────────────────┐     │
    │  │ • File System (Local/NFS)              │     │
    │  │ • CSV Metadata                         │     │
    │  │ • Markdown + Images                    │     │
    │  └────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────┘
```

### 流水线阶段

| 阶段 | 状态 | 描述 |
|------|------|------|
| scrape | ✅ 已实现 | 从 Nature 等来源爬取文章元数据 |
| download | 📋 计划中 | PDF 文件批量下载 |
| ocr | ✅ 已实现 | 使用 marker 引擎进行 OCR 处理 |
| extract | 📋 计划中 | 图表和结构化内容提取 |
| generate-qa (text) | ✅ 已实现 | 从 OCR Markdown 生成文本 QA |
| generate-qa (vision) | ✅ 已实现 | 从图表生成视觉 QA（LlamaFactory 格式） |

## 核心模块

### 1. CLI Interface (`src/cli/`)

**职责**：提供用户交互接口

**主要组件**：
- `main.py`：命令行入口
- `data_inspector.py`：数据检查工具

**设计模式**：
- Command Pattern：命令封装
- Factory Pattern：命令创建

### 2. Pipeline Manager (`src/pipeline_manager.py`)

**职责**：协调和管理处理流水线

**核心功能**：
```python
class PipelineManager:
    def __init__(self, config: ConfigManager):
        self.stages = {}  # 阶段注册表
        self.execution_order = []  # 执行顺序

    def add_stage(self, stage: PipelineStage):
        # 注册新阶段

    def run(self, stages: List[str]):
        # 执行指定阶段

    def _resolve_dependencies(self):
        # 依赖解析（拓扑排序）
```

**设计特点**：
- 声明式阶段定义
- 自动依赖解析
- 支持选择性执行
- 断点续传支持

### 3. Configuration Manager (`src/config_manager.py`)

**职责**：统一配置管理

**分层配置**：
1. 默认配置（代码中）
2. 文件配置（config.json）
3. 环境变量（.env）
4. 运行时参数（CLI args）

**配置热加载**：
```python
config.load()  # 加载配置
config.validate()  # 验证配置
config.save()  # 保存配置
```

### 4. OpenAI API Balancer (`src/api_key_balancer.py`)

**职责**：LLM API 请求负载均衡

**核心功能**：
- 多 API 密钥并发处理
- 自动失败重试和 fallback
- 请求队列管理
- 统计信息收集

**架构设计**：
```python
class OpenAIAPIBalancer:
    def __init__(self, api_keys: List[str]):
        self.workers = []  # 工作线程
        self.request_queue = Queue()  # 请求队列
        self.result_queue = Queue()  # 结果队列
        self.retry_queue = Queue()  # 重试队列

    def submit_chat_completion(self, model, messages, **kwargs):
        # 提交聊天请求

    def get_statistics(self):
        # 获取统计信息
```

**请求状态流转**：
```
PENDING → PROCESSING → SUCCESS
                    ↘ FAILED → RETRYING → SUCCESS/FAILED
```

### 5. Scrapers (`src/scrapers/`)

**职责**：数据源爬取

**扩展接口**：
```python
class BaseScraper(ABC):
    @abstractmethod
    def scrape(self) -> List[ArticleMetadata]:
        pass
```

**已实现**：
- `NatureScraper`：Nature 文章爬取

### 6. OCR 模块 (`src/ocr/`)

**职责**：可扩展的 PDF 到 Markdown 转换

**架构**：
```
BaseOCRModel (ABC)
├── MarkerOCR          (marker, 默认)
├── BaseDeepSeekOCR    (共享基类)
│   ├── DeepSeekOCRv1  (deepseek-ocr,    HF: deepseek-ai/DeepSeek-OCR)
│   └── DeepSeekOCRv2  (deepseek-ocr-v2, HF: deepseek-ai/DeepSeek-OCR-2)
├── MarkItDownOCR      (markitdown, 存根)
└── Pdf2MdOCR          (pdf2md, 存根)
```

**核心机制**：
- `OCRRegistry`：装饰器注册 + 名称查找
- `BaseDeepSeekOCR`：懒加载模型、PDF 转图像、逐页推理、Markdown 组装
- 通过 `--ocr-model` CLI 参数选择模型

**模型选择**：
```bash
uv run python -m src.cli.main list-ocr-models          # 查看可用模型
uv run python -m src.cli.main ocr --ocr-model marker    # Marker (默认)
uv run python -m src.cli.main ocr --ocr-model deepseek-ocr    # DeepSeek v1
uv run python -m src.cli.main ocr --ocr-model deepseek-ocr-v2 # DeepSeek v2
```

### 7. Chunked OCR (`src/chunked-ocr.py`) (Legacy)

**职责**：大规模 PDF 批量 OCR 处理（旧版，建议使用 `src/ocr/` 模块）

**核心功能**：
- 并行处理多个 PDF 文件
- 自动分块和进度跟踪
- 错误恢复和重试

## 数据流设计

### 1. 数据流向

```
Web Sources → Scrapers → Metadata CSV
                ↓
            OCR Engine → Markdown + Images
                ↓
            (Planned) Processors → Structured Data
                ↓
            (Planned) Generators → Training Data
```

### 2. 数据格式转换

| 阶段 | 输入格式 | 输出格式 | 状态 |
|------|---------|----------|------|
| Scraping | HTML | CSV | ✅ 已实现 |
| OCR | PDF | Markdown + Images | ✅ 已实现 |
| Text QA | Markdown | JSONL (QA pairs) | ✅ 已实现 |
| Vision QA | JSON + Images | JSONL (LlamaFactory) | ✅ 已实现 |

### 3. 数据存储策略

**数据组织**：
```
data/
├── nature_metadata/     # 元数据 CSV
│   └── articles.csv
├── pdfs/               # 原始 PDF
│   └── {paper_id}.pdf
├── ocr_results/        # OCR 结果
│   └── {paper_id}/
│       ├── {paper_id}.md
│       ├── {paper_id}_meta.json
│       └── images/
├── sft_data/           # 文本 QA（计划中）
└── vlm_sft_data/       # 视觉 QA（计划中）
```

## 技术栈

### 核心技术

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 语言 | Python 3.8+ | 主开发语言 |
| CLI | Click | 命令行框架 |
| 配置 | python-dotenv | 环境变量管理 |
| OCR | Marker / DeepSeek-OCR | PDF 转换引擎（多模型可选） |
| 爬虫 | BeautifulSoup | HTML 解析 |
| 并行 | threading/multiprocessing | 多线程/多进程处理 |
| 测试 | pytest | 测试框架 |

### 外部依赖

| 服务 | 用途 | 可选性 |
|------|------|--------|
| OpenAI API | QA 生成 | 必需 |
| GPU | OCR 加速 | 可选 |

## 扩展性设计

### 1. 自定义爬虫

添加新的数据源：

```python
from src.scrapers.base import BaseScraper

class ArxivScraper(BaseScraper):
    def scrape(self):
        # Arxiv 爬取逻辑
        return articles

# 配置数据源
config.scrapers.add('arxiv', ArxivScraper)
```

### 2. 自定义 OCR 模型

添加新的 OCR 引擎：

```python
from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry

@OCRRegistry.register
class MyOCR(BaseOCRModel):
    MODEL_NAME = "my-ocr"

    @property
    def name(self) -> str:
        return "My OCR"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_ocr_lib  # noqa: F401
            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        # 实现 PDF → Markdown 转换
        ...
```

在 `src/ocr/__init__.py` 中添加导入即可自动注册：
```python
import src.ocr.my_ocr  # noqa: F401
```

### 3. 自定义处理器

添加新的处理器（需实现）：

```python
class CustomProcessor:
    def process(self, input_data):
        # 自定义处理逻辑
        return processed_data

# 注册到流水线
pipeline.add_stage(
    name='custom',
    processor=CustomProcessor(),
    dependencies=['ocr']
)
```

## 性能优化

### 1. 并行处理

- **进程级并行**：多进程处理不同文件
- **线程级并行**：I/O 密集型操作（API 请求）
- **异步处理**：网络请求和 API 调用

### 2. API 负载均衡

`OpenAIAPIBalancer` 支持：
- 多 API 密钥轮询
- 自动失败重试（最多 3 次）
- 请求限流（每 10 个请求休眠 1 秒）
- 统计信息监控

### 3. 批处理优化

```python
# 批量处理配置
BATCH_SIZES = {
    'ocr': 5,
    'qa_generation': 20  # 计划中
}
```

## 监控和日志

### 1. 日志分级

| 级别 | 用途 | 示例 |
|------|------|------|
| DEBUG | 详细调试信息 | 变量值、函数调用 |
| INFO | 正常流程信息 | 阶段开始/结束 |
| WARNING | 警告信息 | 跳过的文件 |
| ERROR | 错误信息 | 处理失败 |
| CRITICAL | 严重错误 | 系统异常 |

### 2. 监控指标

- **吞吐量**：文件/小时
- **成功率**：成功/总数
- **延迟**：平均处理时间
- **资源使用**：CPU/内存/磁盘

### 3. API 统计

```python
stats = balancer.get_statistics()
# {
#     "total_requests": 100,
#     "pending_requests": 5,
#     "retry_requests": 2,
#     "completed_results": 93,
#     "workers": [
#         {"thread_id": 0, "processed": 35, "failed": 2},
#         {"thread_id": 1, "processed": 33, "failed": 1},
#         {"thread_id": 2, "processed": 32, "failed": 0}
#     ]
# }
```

## 安全性考虑

### 1. API 密钥管理

- 使用环境变量存储
- 不在代码中硬编码
- 支持密钥轮换

### 2. 数据隐私

- 本地处理优先
- 敏感数据脱敏
- 访问控制

### 3. 错误处理

- 优雅降级
- 错误隔离
- 自动重试

## 未来规划

### 短期目标
1. 实现 PDF 下载模块
2. 实现图表提取模块
3. 实现 QA 生成模块

### 中期目标
1. 支持更多数据源（Arxiv、PubMed）
2. 改进 OCR 精度
3. Web UI 界面

### 长期目标
1. 分布式处理支持
2. 机器学习驱动的质量控制
3. 自动化数据标注
