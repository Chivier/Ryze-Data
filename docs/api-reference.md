# API 参考文档

## 目录

- [ConfigManager](#configmanager)
- [PipelineManager](#pipelinemanager)
- [DataInspector](#datainspector)
- [OpenAIAPIBalancer](#openaiapibalancer)
- [Scrapers](#scrapers)
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

> **注意**：目前 PipelineManager 提供了基础框架，实际实现的阶段仅包括 `scrape` 和 `ocr`。

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
result = pipeline.run(stages=["scrape", "ocr"])

# 运行全部阶段，跳过已完成的
result = pipeline.run(skip_existing=True)
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
samples = inspector.get_random_sample("ocr", count=5)
for sample in samples:
    print(sample['file'])
```

### 支持的阶段

| 阶段名称 | 描述 | 文件模式 |
|---------|------|----------|
| scraping | 爬取的元数据 | *.csv |
| ocr | OCR 结果 | */*.md |

## OpenAIAPIBalancer

OpenAI API 密钥负载均衡器，支持多 API 密钥并发请求和自动重试。

### 类定义

```python
from src.api_key_balancer import OpenAIAPIBalancer

balancer = OpenAIAPIBalancer(api_keys=["sk-key1", "sk-key2", "sk-key3"])
```

### 构造函数

#### `__init__(api_keys: List[str], max_queue_size: int = 1000)`

初始化负载均衡器。

**参数**：
- `api_keys`: API 密钥列表
- `max_queue_size`: 最大队列大小

**示例**：
```python
import os

api_keys = [
    os.environ.get("OPENAI_API_KEY_1"),
    os.environ.get("OPENAI_API_KEY_2"),
]
balancer = OpenAIAPIBalancer(api_keys)
```

### 方法

#### `submit_request(method: str, params: Dict[str, Any], callback: Optional[Callable], max_retries: int = 3) -> str`

提交请求到负载均衡器。

**参数**：
- `method`: API 方法名（如 "chat.completions.create"）
- `params`: API 参数
- `callback`: 可选的回调函数
- `max_retries`: 最大重试次数

**返回**：
- 请求 ID

**支持的方法**：
- `chat.completions.create`
- `completions.create`
- `embeddings.create`
- `images.generate`
- `audio.transcriptions.create`
- `audio.translations.create`

#### `submit_chat_completion(model: str, messages: List[Dict[str, str]], callback: Optional[Callable] = None, **kwargs) -> str`

提交聊天完成请求（便捷方法）。

**参数**：
- `model`: 模型名称
- `messages`: 消息列表
- `callback`: 可选的回调函数
- `**kwargs`: 其他 OpenAI API 参数

**返回**：
- 请求 ID

**示例**：
```python
request_id = balancer.submit_chat_completion(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7
)
```

#### `submit_embedding(input: str, model: str = "text-embedding-ada-002", **kwargs) -> str`

提交嵌入请求。

**参数**：
- `input`: 输入文本
- `model`: 模型名称
- `**kwargs`: 其他参数

**返回**：
- 请求 ID

**示例**：
```python
request_id = balancer.submit_embedding(
    input="Hello, world!",
    model="text-embedding-ada-002"
)
```

#### `get_result(timeout: Optional[float] = None) -> Optional[APIRequest]`

获取处理结果。

**参数**：
- `timeout`: 超时时间（秒）

**返回**：
- 处理完成的请求对象，超时返回 None

#### `get_result_by_id(request_id: str, timeout: Optional[float] = None) -> Optional[APIRequest]`

获取特定 ID 的处理结果。

**参数**：
- `request_id`: 请求 ID
- `timeout`: 超时时间（秒，默认 60）

**返回**：
- 处理完成的请求对象，未找到返回 None

#### `wait_for_result(request_id: str, timeout: float = 60) -> APIRequest`

等待特定请求的结果。

**参数**：
- `request_id`: 请求 ID
- `timeout`: 超时时间

**返回**：
- 处理完成的请求对象

**异常**：
- `TimeoutError`: 如果超时

**示例**：
```python
request_id = balancer.submit_chat_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

try:
    result = balancer.wait_for_result(request_id, timeout=30)
    if result.status == RequestStatus.SUCCESS:
        print(result.result.choices[0].message.content)
except TimeoutError:
    print("请求超时")
```

#### `get_all_results() -> List[APIRequest]`

获取所有当前可用的结果。

**返回**：
- 结果列表

#### `get_statistics() -> Dict[str, Any]`

获取统计信息。

**返回**：
- 统计信息字典

**示例**：
```python
stats = balancer.get_statistics()
print(f"总请求数: {stats['total_requests']}")
print(f"待处理: {stats['pending_requests']}")
print(f"重试中: {stats['retry_requests']}")

for worker in stats['workers']:
    print(f"Worker {worker['thread_id']}: "
          f"processed={worker['processed']}, "
          f"failed={worker['failed']}")
```

#### `shutdown(wait: bool = True)`

关闭负载均衡器。

**参数**：
- `wait`: 是否等待所有请求处理完成

**示例**：
```python
# 优雅关闭，等待所有请求完成
balancer.shutdown(wait=True)

# 立即关闭
balancer.shutdown(wait=False)
```

### APIRequest 结构

```python
@dataclass
class APIRequest:
    id: str                              # 请求 ID
    method: str                          # API 方法
    params: Dict[str, Any]               # 请求参数
    callback: Optional[Callable] = None  # 回调函数
    retry_count: int = 0                 # 当前重试次数
    max_retries: int = 3                 # 最大重试次数
    result: Any = None                   # 成功时的结果
    error: Any = None                    # 失败时的错误
    status: RequestStatus = RequestStatus.PENDING  # 请求状态

class RequestStatus(Enum):
    PENDING = "pending"       # 等待处理
    PROCESSING = "processing" # 处理中
    SUCCESS = "success"       # 成功
    FAILED = "failed"         # 失败
    RETRYING = "retrying"     # 重试中
```

### 使用回调函数

```python
def on_complete(result, error=None):
    if error:
        print(f"请求失败: {error}")
    else:
        print(f"响应: {result.choices[0].message.content}")

balancer.submit_chat_completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    callback=on_complete
)
```

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

## CLI Commands

### 主命令

```bash
python -m src.cli.main [OPTIONS] COMMAND [ARGS]
```

**全局选项**：
- `--config, -c`: 配置文件路径
- `--env, -e`: 环境文件路径

### 已实现的子命令

#### `scrape`

爬取 Nature 文章元数据。

```bash
python -m src.cli.main scrape
```

#### `ocr`

运行 OCR 处理。

```bash
python -m src.cli.main ocr [OPTIONS]
```

**选项**：
- `--input-dir`: 输入目录
- `--output-dir`: 输出目录
- `--batch-size`: 批处理大小

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
