# Ryze-Data 架构设计文档

## 目录

- [系统概述](#系统概述)
- [架构原则](#架构原则)
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
- Scraper：数据爬取
- Downloader：文件下载
- Processor：内容处理
- Generator：数据生成

### 2. 依赖倒置原则
- 核心业务逻辑不依赖具体实现
- 通过接口和抽象类定义契约
- 配置驱动的依赖注入

### 3. 开闭原则
- 对扩展开放：易于添加新的处理器
- 对修改关闭：核心流程稳定不变

## Project Structure / 项目结构

```
Ryze-Data/
├── .env.example                 # Environment variables template
├── .env.test                    # Test environment configuration
├── config.example.json          # Configuration template with env var support
├── config.test.json             # Test configuration
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── run_tests.py                 # Test runner script
├── README.md                    # Project documentation
├── LICENSE                      # AGPL-3.0 license
│
├── src/                         # Source code directory
│   ├── __init__.py
│   ├── config_manager.py        # Configuration management with env expansion
│   ├── pipeline_manager.py      # Pipeline orchestration and execution
│   │
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py             # Main CLI entry point
│   │   └── data_inspector.py   # Data inspection and sampling tool
│   │
│   ├── scrapers/               # Data source scrapers
│   │   ├── __init__.py
│   │   ├── base.py            # Base scraper interface
│   │   └── nature.py          # Nature articles scraper
│   │
│   ├── downloaders/            # File download utilities
│   │   ├── __init__.py
│   │   └── pdf_downloader.py  # PDF and supplement downloader
│   │
│   ├── processors/             # Data processing modules
│   │   ├── __init__.py
│   │   ├── ocr_processor.py   # OCR processing with marker
│   │   └── figure_extractor.py # Figure and image extraction
│   │
│   └── generators/             # Data generation modules
│       ├── __init__.py
│       ├── text_qa.py         # Text-based QA generation
│       └── vision_qa.py       # Vision-based QA generation
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Test fixtures and configuration
│   ├── README.md              # Test documentation
│   │
│   ├── unit/                  # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config_manager.py
│   │   └── test_data_inspector.py
│   │
│   ├── integration/           # Integration tests
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   │
│   └── fixtures/              # Test data
│       ├── sample.pdf
│       ├── sample_metadata.csv
│       └── mock_responses.json
│
├── docs/                      # Documentation
│   ├── architecture.md        # Architecture design document
│   ├── api-reference.md       # API documentation
│   ├── configuration.md       # Configuration guide
│   ├── data-formats.md        # Data format specifications
│   ├── development.md         # Development guide
│   ├── troubleshooting.md     # Troubleshooting guide
│   │
│   └── zh-CN/                # Chinese documentation
│       ├── README.md         # 中文项目说明
│       ├── architecture.md   # 架构设计文档
│       ├── configuration.md  # 配置指南
│       └── development.md    # 开发指南
│
├── prompts/                   # LLM prompt templates
│   ├── text_qa_prompt.txt    # Text QA generation prompt
│   └── vision_qa_prompt.txt  # Vision QA generation prompt
│
├── data/                      # Data directory (git-ignored)
│   ├── nature_metadata/       # Scraped metadata
│   ├── pdfs/                 # Downloaded PDFs
│   ├── ocr_results/          # OCR processing results
│   ├── figures/              # Extracted figures
│   ├── sft_data/             # Text QA training data
│   └── vlm_sft_data/         # Vision QA training data
│
└── data-sample/              # Sample data for testing
    ├── nature_metadata/
    │   └── sample.csv
    ├── pdfs/
    │   └── sample.pdf
    └── ocr_results/
        └── sample/
            ├── sample.md
            └── sample_meta.json
```

### File Purpose Description / 文件用途说明

| File/Directory | Purpose | 用途 |
|----------------|---------|------|
| `.env.example` | Environment configuration template | 环境配置模板 |
| `config.example.json` | Configuration with ${VAR:default} syntax | 支持环境变量的配置文件 |
| `src/config_manager.py` | Unified configuration management | 统一配置管理 |
| `src/pipeline_manager.py` | Pipeline orchestration logic | 流水线编排逻辑 |
| `src/cli/main.py` | CLI commands implementation | CLI命令实现 |
| `src/cli/data_inspector.py` | Data inspection utilities | 数据检查工具 |
| `src/scrapers/` | Web scraping modules | 网页爬取模块 |
| `src/downloaders/` | File download logic | 文件下载逻辑 |
| `src/processors/` | Data processing modules | 数据处理模块 |
| `src/generators/` | QA generation modules | QA生成模块 |
| `tests/` | Comprehensive test suite | 完整测试套件 |
| `docs/` | Technical documentation | 技术文档 |
| `prompts/` | LLM prompt templates | LLM提示词模板 |
| `data/` | Runtime data storage | 运行时数据存储 |

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
└─────────┬──────────┬──────────┬──────────┬────────────┘
          │          │          │          │
    ┌─────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐
    │ Scrapers │ │ Down │ │Process │ │ Gener  │
    │          │ │loader│ │  ors   │ │ ators  │
    └─────┬────┘ └──┬───┘ └───┬────┘ └───┬────┘
          │          │          │          │
    ┌─────▼──────────▼──────────▼──────────▼────┐
    │           Configuration Manager            │
    │         (src/config_manager.py)            │
    └──────────────┬──────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────┐
    │              Data Storage                    │
    │  ┌────────────────────────────────────┐     │
    │  │ • File System (Local/NFS)          │     │
    │  │ • Database (Metadata)              │     │
    │  │ • Object Storage (S3/OSS)          │     │
    │  └────────────────────────────────────┘     │
    └──────────────────────────────────────────────┘
```

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

### 4. Scrapers (`src/scrapers/`)

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

### 5. Processors (`src/processors/`)

**职责**：内容处理和提取

**处理器类型**：
- `FigureExtractor`：图片提取
- `ContentParser`：内容解析
- `AbstractExtractor`：摘要提取

### 6. Generators (`src/generators/`)

**职责**：生成训练数据

**生成器类型**：
- `TextQAGenerator`：文本 QA 生成
- `VisionQAGenerator`：视觉 QA 生成

**并行处理支持**：
```python
class ParallelGenerator:
    def __init__(self, workers: int):
        self.pool = Pool(workers)
    
    def process_batch(self, items: List):
        return self.pool.map(self.process_single, items)
```

## 数据流设计

### 1. 数据流向

```
Web Sources → Scrapers → Metadata DB
                ↓
            Downloader → PDF Storage
                ↓
            OCR Engine → Text/Image Storage
                ↓
            Processors → Structured Data
                ↓
            Generators → Training Data
```

### 2. 数据格式转换

| 阶段 | 输入格式 | 输出格式 |
|------|---------|----------|
| Scraping | HTML | CSV/JSON |
| Download | URLs | PDF files |
| OCR | PDF | Markdown + Images |
| Processing | Markdown | Structured JSON |
| Generation | JSON | JSONL (QA pairs) |

### 3. 数据存储策略

**分层存储**：
- **热数据**：正在处理的数据（本地 SSD）
- **温数据**：近期处理的数据（本地 HDD）
- **冷数据**：归档数据（对象存储）

**数据组织**：
```
data/
├── nature_metadata/     # 元数据
├── pdfs/               # 原始 PDF
├── ocr_results/        # OCR 结果
│   ├── {paper_id}/
│   │   ├── {paper_id}.md
│   │   ├── {paper_id}_meta.json
│   │   └── figures/
├── sft_data/           # 文本 QA
└── vlm_sft_data/       # 视觉 QA
```

## 技术栈

### 核心技术

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 语言 | Python 3.8+ | 主开发语言 |
| CLI | Click | 命令行框架 |
| 配置 | python-dotenv | 环境变量管理 |
| OCR | Marker | PDF 转换引擎 |
| 爬虫 | BeautifulSoup | HTML 解析 |
| 并行 | multiprocessing | 多进程处理 |
| 测试 | pytest | 测试框架 |

### 外部依赖

| 服务 | 用途 | 可选性 |
|------|------|--------|
| OpenAI API | QA 生成 | 必需 |
| GPU | OCR 加速 | 可选 |
| Redis | 任务队列 | 可选 |
| S3/OSS | 数据存储 | 可选 |

## 扩展性设计

### 1. 插件架构

支持通过插件扩展功能：

```python
# 插件接口
class Plugin(ABC):
    @abstractmethod
    def initialize(self, config):
        pass
    
    @abstractmethod
    def process(self, data):
        pass

# 插件注册
plugin_registry.register('custom_processor', CustomPlugin)
```

### 2. 自定义处理器

添加新的处理器：

```python
class CustomProcessor(BaseProcessor):
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

### 3. 数据源扩展

添加新的数据源：

```python
class ArxivScraper(BaseScraper):
    def scrape(self):
        # Arxiv 爬取逻辑
        return articles

# 配置数据源
config.scrapers.add('arxiv', ArxivScraper)
```

### 4. 输出格式扩展

支持自定义输出格式：

```python
class CustomFormatter(BaseFormatter):
    def format(self, data):
        # 自定义格式化
        return formatted_data
```

## 性能优化

### 1. 并行处理

- **进程级并行**：多进程处理不同文件
- **线程级并行**：I/O 密集型操作
- **异步处理**：网络请求和 API 调用

### 2. 缓存策略

- **结果缓存**：避免重复处理
- **配置缓存**：减少解析开销
- **连接池**：复用网络连接

### 3. 批处理优化

```python
# 批量处理配置
BATCH_SIZES = {
    'download': 10,
    'ocr': 5,
    'qa_generation': 20
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

### 3. 状态追踪

```python
# 处理状态
class ProcessingStatus:
    total: int
    completed: int
    failed: int
    skipped: int
    
    @property
    def progress(self):
        return self.completed / self.total * 100
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

1. 支持更多数据源（Arxiv、PubMed）
2. 改进 OCR 精度
3. 优化并行处理性能

### 中期目标

1. 分布式处理支持
2. Web UI 界面
3. 实时处理流水线

### 长期目标

1. 机器学习驱动的质量控制
2. 自动化数据标注
3. 多模态数据处理