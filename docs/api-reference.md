# API 参考文档

## 目录

- [ConfigManager](#configmanager)
- [PipelineManager](#pipelinemanager)
- [DataInspector](#datainspector)
- [Scrapers](#scrapers)
- [Processors](#processors)
- [Generators](#generators)
- [CLI Commands](#cli-commands)

## ConfigManager

配置管理器，负责加载、验证和管理系统配置。

### 类定义

```python
from src.config_manager import ConfigManager

config = ConfigManager()
```

### 方法

#### `load(config_path: str = "config.json") -> None`

加载配置文件和环境变量。

**参数**：
- `config_path`: 配置文件路径，默认为 "config.json"

**示例**：
```python
config = ConfigManager()
config.load("custom_config.json")
```

#### `validate() -> bool`

验证配置的有效性。

**返回**：
- `bool`: 配置是否有效

**示例**：
```python
if config.validate():
    print("配置有效")
else:
    print("配置无效，请检查设置")
```

#### `get(key_path: str, default: Any = None) -> Any`

通过点分路径获取配置值。

**参数**：
- `key_path`: 配置路径，如 "ocr.model"
- `default`: 默认值

**返回**：
- 配置值或默认值

**示例**：
```python
batch_size = config.get("ocr.batch_size", 10)
model = config.get("parsing_model.model", "gpt-4")
```

#### `save(config_path: Optional[str] = None) -> None`

保存当前配置到文件。

**参数**：
- `config_path`: 保存路径，默认使用加载时的路径

**示例**：
```python
config.ocr.batch_size = 20
config.save("updated_config.json")
```

### 配置结构

```python
@dataclass
class OCRConfig:
    model: str = "marker"
    batch_size: int = 10
    confidence_threshold: float = 0.8
    language: str = "en"
    gpu_enabled: bool = True
    gpu_memory_limit: float = 0.5
    timeout_seconds: int = 300

@dataclass
class PathConfig:
    data_root: str = "./data"
    logs_dir: str = "./logs"
    nature_data: str = "./data/nature_metadata"
    pdf_dir: str = "./data/pdfs"
    ocr_output: str = "./data/ocr_results"
    # ... 更多路径配置

@dataclass
class ProcessingConfig:
    parallel_workers: int = 4
    max_retries: int = 3
    retry_delay_seconds: int = 5
    quality_threshold: float = 2.5
    qa_ratio: int = 8
    max_papers: int = 1000
```

## PipelineManager

流水线管理器，协调各处理阶段的执行。

### 类定义

```python
from src.pipeline_manager import PipelineManager

pipeline = PipelineManager(config)
```

### 方法

#### `add_stage(name, description, function, dependencies, output_path)`

添加处理阶段到流水线。

**参数**：
- `name`: 阶段名称
- `description`: 阶段描述
- `function`: 执行函数
- `dependencies`: 依赖的阶段列表
- `output_path`: 输出路径

**示例**：
```python
pipeline.add_stage(
    name="custom_stage",
    description="自定义处理阶段",
    function=custom_processor,
    dependencies=["ocr"],
    output_path="/data/custom_output"
)
```

#### `run(stages: Optional[List[str]], skip_existing: bool, force: bool) -> PipelineResult`

运行流水线。

**参数**：
- `stages`: 要运行的阶段列表，None 表示全部
- `skip_existing`: 是否跳过已存在输出的阶段
- `force`: 强制运行，即使有错误也继续

**返回**：
- `PipelineResult`: 执行结果

**示例**：
```python
# 运行特定阶段
result = pipeline.run(stages=["scrape", "download"])

# 运行全部阶段，跳过已完成的
result = pipeline.run(skip_existing=True)

# 强制运行，忽略错误
result = pipeline.run(force=True)
```

#### `get_status() -> Dict[str, Any]`

获取流水线状态。

**返回**：
- 包含各阶段状态的字典

**示例**：
```python
status = pipeline.get_status()
for stage_name, info in status['stages'].items():
    print(f"{stage_name}: {info['status']}")
```

#### `save_state(filepath: str) -> None`

保存流水线状态到文件。

**参数**：
- `filepath`: 保存路径

**示例**：
```python
pipeline.save_state("pipeline_state.json")
```

### PipelineResult 结构

```python
@dataclass
class PipelineResult:
    total_stages: int
    completed_stages: int
    failed_stages: int
    skipped_stages: int
    total_time: float
    stage_results: Dict[str, Dict[str, Any]]
```

## DataInspector

数据检查器，用于检查和采样各阶段数据。

### 类定义

```python
from src.cli.data_inspector import DataInspector

inspector = DataInspector(config)
```

### 方法

#### `get_stage_info(stage: str) -> Dict[str, Any]`

获取指定阶段的信息。

**参数**：
- `stage`: 阶段名称

**返回**：
- 包含阶段信息的字典

**示例**：
```python
info = inspector.get_stage_info("ocr")
print(f"文件数量: {info['count']}")
print(f"总大小: {info['total_size']}")
```

#### `sample_file(file_path: str, sample_size: int = 5) -> Dict[str, Any]`

采样文件内容。

**参数**：
- `file_path`: 文件路径
- `sample_size`: 采样大小

**返回**：
- 包含文件样本的字典

**示例**：
```python
sample = inspector.sample_file("/data/test.json", sample_size=3)
print(sample['content'])
```

#### `get_random_sample(stage: str, count: int = 1) -> List[Dict[str, Any]]`

从阶段随机采样文件。

**参数**：
- `stage`: 阶段名称
- `count`: 采样数量

**返回**：
- 样本列表

**示例**：
```python
samples = inspector.get_random_sample("qa-text", count=5)
for sample in samples:
    print(sample['file'])
```

### 支持的阶段

| 阶段名称 | 描述 | 文件模式 |
|---------|------|----------|
| scraping | 爬取的元数据 | *.csv |
| pdf | 下载的 PDF | *.pdf |
| ocr | OCR 结果 | */*.md |
| figures | 提取的图片 | *.json |
| abstracts | 摘要 | *.txt |
| qa-text | 文本 QA | *_qa.jsonl |
| qa-vision | 视觉 QA | *_vision_qa.jsonl |

## Scrapers

### NatureScraper

Nature 文章爬虫。

```python
from src.scrapers.nature_scraper import NatureScraper

scraper = NatureScraper(output_dir="./data/nature_metadata")
```

#### 方法

##### `run() -> None`

运行爬虫，爬取所有文章元数据。

**示例**：
```python
scraper = NatureScraper(output_dir="./nature_data")
scraper.run()
```

#### 输出格式

CSV 文件，包含以下字段：
- `title`: 文章标题
- `url`: 文章 URL
- `abstract`: 摘要
- `open_access`: 是否开放获取
- `date`: 发布日期
- `author`: 作者列表

## Processors

### FigureExtractor

图片提取器。

```python
from src.processors.figure_extractor import FigureExtractor

extractor = FigureExtractor(
    input_dir="./ocr_results",
    output_dir="./figures",
    threads=8
)
```

#### 方法

##### `run() -> None`

运行图片提取。

**示例**：
```python
extractor = FigureExtractor(
    input_dir=config.paths.ocr_output,
    output_dir=config.paths.figures_dir,
    threads=4
)
extractor.run()
```

#### 输出格式

JSON 文件，结构如下：
```json
[{
    "figure_path": "/path/to/figure.jpg",
    "related_info": [{
        "position": "before_figure",
        "info": "图片相关文本"
    }]
}]
```

## Generators

### TextQAGenerator

文本 QA 生成器。

```python
from src.generators.text_qa_generator import TextQAGenerator

generator = TextQAGenerator(
    ocr_dir="./ocr_results",
    abstract_dir="./abstracts",
    output_dir="./sft_data",
    model="gpt-4",
    qa_ratio=8
)
```

#### 方法

##### `run() -> None`

生成文本 QA 对。

**示例**：
```python
generator = TextQAGenerator(
    ocr_dir=config.paths.ocr_output,
    abstract_dir=config.paths.abstract_dir,
    output_dir=config.paths.sft_data,
    model="gpt-4",
    qa_ratio=10
)
generator.run()
```

### VisionQAGenerator

视觉 QA 生成器。

```python
from src.generators.vision_qa_generator import VisionQAGenerator

generator = VisionQAGenerator(
    vlm_dir="./vlm_preprocessing",
    abstract_dir="./abstracts", 
    output_dir="./vlm_sft_data",
    model="gpt-4-vision",
    workers=4,
    qa_ratio=8
)
```

#### 方法

##### `run() -> None`

生成视觉 QA 对。

**示例**：
```python
generator = VisionQAGenerator(
    vlm_dir=config.paths.vlm_preprocessing,
    abstract_dir=config.paths.abstract_dir,
    output_dir=config.paths.vlm_sft_data,
    model="gpt-4-vision",
    workers=8,
    qa_ratio=5
)
generator.run()
```

## CLI Commands

### 主命令

```bash
python -m src.cli.main [OPTIONS] COMMAND [ARGS]
```

**全局选项**：
- `--config, -c`: 配置文件路径
- `--env, -e`: 环境文件路径

### 子命令

#### `scrape`

爬取 Nature 文章。

```bash
python -m src.cli.main scrape
```

#### `download`

下载 PDF 文件。

```bash
python -m src.cli.main download [OPTIONS]
```

**选项**：
- `--workers, -w`: 并行工作线程数
- `--servers`: 下载服务器列表

#### `ocr`

运行 OCR 处理。

```bash
python -m src.cli.main ocr [OPTIONS]
```

**选项**：
- `--input-dir`: 输入目录
- `--output-dir`: 输出目录
- `--batch-size`: 批处理大小

#### `extract`

提取图片和文本。

```bash
python -m src.cli.main extract [OPTIONS]
```

**选项**：
- `--threads, -t`: 处理线程数

#### `generate-qa`

生成 QA 对。

```bash
python -m src.cli.main generate-qa [OPTIONS]
```

**选项**：
- `--mode`: 生成模式 (text/vision/both)
- `--model`: 使用的模型
- `--workers, -w`: 并行工作线程数
- `--qa-ratio`: 每个单元的 QA 对数量

#### `pipeline`

运行完整流水线。

```bash
python -m src.cli.main pipeline [OPTIONS]
```

**选项**：
- `--stages, -s`: 要运行的阶段
- `--workers, -w`: 并行工作线程数

#### `inspect`

数据检查命令组。

##### `inspect list`

列出所有可检查的阶段。

```bash
python -m src.cli.main inspect list
```

##### `inspect stage`

检查特定阶段。

```bash
python -m src.cli.main inspect stage STAGE_NAME [OPTIONS]
```

**参数**：
- `STAGE_NAME`: 阶段名称

**选项**：
- `--sample, -s`: 随机采样数量
- `--detailed, -d`: 显示详细内容

##### `inspect file`

检查特定文件。

```bash
python -m src.cli.main inspect file FILE_PATH [OPTIONS]
```

**参数**：
- `FILE_PATH`: 文件路径

**选项**：
- `--lines, -l`: 显示的行数

##### `inspect all`

显示所有阶段概览。

```bash
python -m src.cli.main inspect all [OPTIONS]
```

**选项**：
- `--verbose, -v`: 显示详细信息

##### `inspect stats`

显示统计信息。

```bash
python -m src.cli.main inspect stats
```

#### `config-show`

显示当前配置。

```bash
python -m src.cli.main config-show
```

## 错误处理

### 异常类型

```python
class ConfigurationError(Exception):
    """配置错误"""
    pass

class ProcessingError(Exception):
    """处理错误"""
    pass

class ValidationError(Exception):
    """验证错误"""
    pass
```

### 错误处理模式

```python
try:
    result = pipeline.run(stages=["ocr"])
except ProcessingError as e:
    logger.error(f"处理失败: {e}")
    # 错误恢复逻辑
except Exception as e:
    logger.critical(f"未预期的错误: {e}")
    # 紧急处理
```

## 扩展 API

### 自定义处理器

```python
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理输入数据"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data: Any) -> bool:
        """验证输出"""
        pass
```

### 自定义爬虫

```python
from abc import ABC, abstractmethod

class BaseScraper(ABC):
    @abstractmethod
    def scrape(self) -> List[Dict[str, Any]]:
        """爬取数据"""
        pass
    
    @abstractmethod
    def parse(self, html: str) -> Dict[str, Any]:
        """解析 HTML"""
        pass
```

### 插件接口

```python
from abc import ABC, abstractmethod

class Plugin(ABC):
    @abstractmethod
    def initialize(self, config: ConfigManager) -> None:
        """初始化插件"""
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
```