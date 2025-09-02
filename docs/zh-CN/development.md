# 开发指南

## 目录

- [开发环境搭建](#开发环境搭建)
- [项目架构详解](#项目架构详解)
- [代码风格与规范](#代码风格与规范)
- [添加新功能](#添加新功能)
- [测试指南](#测试指南)
- [调试技巧](#调试技巧)
- [性能分析](#性能分析)
- [贡献流程](#贡献流程)
- [发布流程](#发布流程)

## 开发环境搭建

### 先决条件

- Python 3.8 或更高版本
- Git 版本控制工具
- 虚拟环境工具（venv、conda 或 virtualenv）
- 支持 Python 的 IDE（推荐 VSCode、PyCharm）

### 初始设置

1. **Fork 并克隆仓库**
```bash
# 首先在 GitHub 上 Fork 仓库
git clone https://github.com/YOUR_USERNAME/ryze-data.git
cd ryze-data
git remote add upstream https://github.com/original/ryze-data.git
```

2. **创建开发环境**
```bash
# 创建虚拟环境
python -m venv venv-dev
source venv-dev/bin/activate  # Windows 系统: venv-dev\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **配置预提交钩子**
```bash
# 安装 pre-commit
pip install pre-commit

# 设置钩子
pre-commit install

# 手动运行钩子
pre-commit run --all-files
```

4. **IDE 配置**

**VSCode 配置文件 settings.json:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestArgs": ["tests"],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true
}
```

**PyCharm 配置：**
- 设置 Python 解释器为虚拟环境
- 启用代码检查
- 配置 pytest 作为测试运行器
- 设置 Black 作为代码格式化工具

## 项目架构详解

### 目录结构说明

```
src/
├── cli/                 # 命令行接口
│   ├── __init__.py
│   ├── main.py         # CLI 入口点
│   └── data_inspector.py # 数据检查工具
├── scrapers/           # 网页爬虫模块
│   ├── __init__.py
│   ├── base_scraper.py # 抽象基类
│   └── nature_scraper.py # Nature 实现
├── downloaders/        # 下载管理器
│   ├── __init__.py
│   ├── base_downloader.py
│   └── pdf_downloader.py
├── processors/         # 数据处理器
│   ├── __init__.py
│   ├── base_processor.py
│   ├── ocr_processor.py
│   └── figure_extractor.py
├── generators/         # 问答生成器
│   ├── __init__.py
│   ├── base_generator.py
│   ├── text_qa_generator.py
│   └── vision_qa_generator.py
├── config_manager.py   # 配置管理
├── pipeline_manager.py # 流水线编排
└── utils/             # 工具函数
    ├── __init__.py
    ├── logging.py
    └── validation.py
```

### 设计模式

#### 1. **抽象工厂模式**

用于创建不同类型的处理器和生成器：

```python
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """所有处理器的抽象基类"""
    
    @abstractmethod
    def process(self, input_data):
        """处理输入数据"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data):
        """处理前验证输入"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data):
        """处理后验证输出"""
        pass

class OCRProcessor(BaseProcessor):
    """OCR 处理的具体实现"""
    
    def process(self, input_data):
        # 实现细节
        pass
```

#### 2. **单例模式**

用于配置管理：

```python
class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
```

#### 3. **策略模式**

用于不同的问答生成策略：

```python
class QAGenerationStrategy(ABC):
    @abstractmethod
    def generate(self, context, config):
        pass

class FactualQAStrategy(QAGenerationStrategy):
    def generate(self, context, config):
        # 生成事实型问答对
        pass

class ConceptualQAStrategy(QAGenerationStrategy):
    def generate(self, context, config):
        # 生成概念型问答对
        pass
```

## 代码风格与规范

### Python 代码规范

我们遵循 PEP 8 规范，并有以下补充规定：

1. **行长度**：最大 88 字符（Black 默认值）
2. **导入顺序**：
   ```python
   # 标准库
   import os
   import sys
   
   # 第三方库
   import numpy as np
   import pandas as pd
   
   # 本地导入
   from src.config_manager import ConfigManager
   from src.utils import logger
   ```

3. **文档字符串**：使用 Google 风格
   ```python
   def process_data(input_file: str, output_dir: str, batch_size: int = 10) -> dict:
       """处理输入文件中的数据并保存到输出目录。
       
       Args:
           input_file: 输入文件路径
           output_dir: 输出文件目录
           batch_size: 批处理大小
       
       Returns:
           包含处理结果的字典
       
       Raises:
           FileNotFoundError: 输入文件不存在时
           ValueError: batch_size 小于 1 时
       """
       pass
   ```

4. **类型注解**：所有公共函数必须使用
   ```python
   from typing import List, Dict, Optional, Union
   
   def parse_config(config_path: str) -> Dict[str, Any]:
       pass
   ```

### 代码质量工具

```bash
# 格式化代码
black src/ tests/

# 检查代码风格
flake8 src/ tests/

# 静态类型检查
mypy src/

# 安全性检查
bandit -r src/

# 复杂度分析
radon cc src/ -s
```

## 添加新功能

### 1. 添加新的爬虫

通过扩展 `BaseScraper` 创建新爬虫：

```python
# src/scrapers/arxiv_scraper.py
from typing import List, Dict, Any
from src.scrapers.base_scraper import BaseScraper

class ArxivScraper(BaseScraper):
    """arXiv 论文爬虫"""
    
    def __init__(self, output_dir: str, config: dict = None):
        super().__init__(output_dir, config)
        self.base_url = "https://arxiv.org"
    
    def scrape(self, query: str = None) -> List[Dict[str, Any]]:
        """根据查询条件爬取 arXiv 论文"""
        # 实现细节
        pass
    
    def parse_paper(self, paper_html: str) -> Dict[str, Any]:
        """解析单篇论文的元数据"""
        # 实现细节
        pass
```

在流水线中注册爬虫：

```python
# src/pipeline_manager.py
from src.scrapers.arxiv_scraper import ArxivScraper

# 在 PipelineManager.__init__ 中
self.scrapers['arxiv'] = ArxivScraper
```

### 2. 添加新的处理器

```python
# src/processors/table_extractor.py
from src.processors.base_processor import BaseProcessor

class TableExtractor(BaseProcessor):
    """从文档中提取表格"""
    
    def process(self, input_data):
        """提取输入数据中的表格"""
        tables = []
        # 提取逻辑
        return tables
    
    def validate_input(self, input_data):
        """确保输入是有效的文档数据"""
        # 验证逻辑
        return True
    
    def validate_output(self, output_data):
        """确保提取的表格有效"""
        # 验证逻辑
        return True
```

### 3. 添加新的 CLI 命令

```python
# src/cli/main.py
@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
@click.pass_context
def export(ctx, format):
    """以指定格式导出处理后的数据"""
    cfg = ctx.obj['config']
    
    if format == 'json':
        export_json(cfg)
    elif format == 'csv':
        export_csv(cfg)
    
    click.echo(f"已完成 {format} 格式导出")
```

## 测试指南

### 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_config_manager.py
│   ├── test_scrapers.py
│   └── test_processors.py
├── integration/             # 集成测试
│   ├── test_pipeline.py
│   └── test_end_to_end.py
├── fixtures/                # 测试固件
│   ├── sample_data.py
│   └── mock_responses.py
└── conftest.py             # Pytest 配置
```

### 编写测试

#### 单元测试示例

```python
# tests/unit/test_processors.py
import pytest
from unittest.mock import Mock, patch
from src.processors.ocr_processor import OCRProcessor

class TestOCRProcessor:
    """OCR 处理器测试套件"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return OCRProcessor(config={'batch_size': 10})
    
    def test_process_valid_pdf(self, processor, tmp_path):
        """测试处理有效 PDF 文件"""
        # 准备
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"PDF content")
        
        # 执行
        result = processor.process(str(pdf_file))
        
        # 断言
        assert result is not None
        assert 'text' in result
        assert 'images' in result
    
    def test_process_invalid_file(self, processor):
        """测试处理无效文件"""
        with pytest.raises(FileNotFoundError):
            processor.process("nonexistent.pdf")
    
    @patch('src.processors.ocr_processor.marker')
    def test_ocr_error_handling(self, mock_marker, processor):
        """测试 OCR 错误处理"""
        mock_marker.convert.side_effect = Exception("OCR 失败")
        
        with pytest.raises(ProcessingError):
            processor.process("test.pdf")
```

#### 集成测试示例

```python
# tests/integration/test_pipeline.py
import pytest
from src.pipeline_manager import PipelineManager
from src.config_manager import ConfigManager

@pytest.mark.integration
class TestPipeline:
    """流水线集成测试"""
    
    def test_full_pipeline(self, sample_data):
        """测试完整流水线执行"""
        # 设置
        config = ConfigManager()
        config.load("tests/config.test.json")
        pipeline = PipelineManager(config)
        
        # 运行流水线
        result = pipeline.run(stages=['scrape', 'download', 'ocr'])
        
        # 验证
        assert result.completed_stages == 3
        assert result.failed_stages == 0
```

### 运行测试

```bash
# 运行所有测试
pytest

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行特定测试文件
pytest tests/unit/test_config_manager.py

# 运行匹配模式的测试
pytest -k "test_ocr"

# 详细输出
pytest -v

# 仅运行集成测试
pytest -m integration

# 并行执行测试
pytest -n auto
```

## 调试技巧

### 1. 使用 Python 调试器

```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# 或使用 Python 3.7+ 的内置断点
breakpoint()

# IPython 调试器（更好的界面）
import ipdb; ipdb.set_trace()
```

### 2. 调试日志

```python
import logging

# 配置调试日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"处理数据: {data[:100]}...")  # 记录前 100 个字符
    
    try:
        result = transform(data)
        logger.debug(f"转换成功: {len(result)} 项")
    except Exception as e:
        logger.error(f"转换失败: {e}", exc_info=True)
        raise
    
    return result
```

### 3. VSCode 远程调试

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "调试流水线",
            "type": "python",
            "request": "launch",
            "module": "src.cli.main",
            "args": ["pipeline", "--stages", "ocr"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

## 性能分析

### 1. CPU 性能分析

```python
import cProfile
import pstats
from pstats import SortKey

# 分析代码块
profiler = cProfile.Profile()
profiler.enable()

# 你的代码
process_large_dataset()

profiler.disable()

# 打印统计信息
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(10)  # 前 10 个函数

# 保存分析数据
stats.dump_stats('profile_data.prof')
```

### 2. 内存分析

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 使用大量内存的函数
    large_list = [i for i in range(1000000)]
    return large_list

# 运行: python -m memory_profiler script.py
```

### 3. 行级分析

```python
# 安装: pip install line_profiler

from line_profiler import LineProfiler

def slow_function():
    result = []
    for i in range(1000):
        result.append(i ** 2)
    return result

lp = LineProfiler()
lp_wrapper = lp(slow_function)
lp_wrapper()
lp.print_stats()
```

## 贡献流程

### 1. 分支策略

```bash
# 创建功能分支
git checkout -b feature/new-scraper

# 创建修复分支
git checkout -b bugfix/ocr-timeout

# 创建紧急修复分支
git checkout -b hotfix/critical-issue
```

### 2. 提交规范

遵循约定式提交：

```bash
# 功能
git commit -m "feat: 添加 arXiv 爬虫支持"

# 错误修复
git commit -m "fix: 解决大型 PDF 的 OCR 超时问题"

# 文档
git commit -m "docs: 更新新端点的 API 参考"

# 性能
git commit -m "perf: 优化问答生成器的批处理"

# 重构
git commit -m "refactor: 简化配置管理器实现"

# 测试
git commit -m "test: 添加流水线集成测试"

# 杂项
git commit -m "chore: 更新依赖"
```

### 3. 拉取请求流程

1. **更新你的 fork**
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

2. **创建功能分支**
```bash
git checkout -b feature/your-feature
```

3. **修改并测试**
```bash
# 进行修改
# 运行测试
pytest
# 运行代码检查
flake8 src/
black src/
```

4. **提交并推送**
```bash
git add .
git commit -m "feat: 你的功能描述"
git push origin feature/your-feature
```

5. **创建拉取请求**
- 前往 GitHub 创建 PR
- 填写 PR 模板
- 关联相关 issue
- 请求审查

### 4. 代码审查指南

**审查者：**
- 检查代码是否遵循规范
- 验证是否包含测试
- 确保文档已更新
- 如可能，在本地测试
- 提供建设性反馈

**作者：**
- 回应所有评论
- 进行请求的修改
- 如范围变化，更新 PR 描述
- 必要时进行变基

## 发布流程

### 1. 版本号规则

遵循语义化版本（主版本.次版本.修订版本）：
- 主版本：重大变更
- 次版本：新功能（向后兼容）
- 修订版本：错误修复

### 2. 发布清单

```markdown
- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] CHANGELOG.md 已更新
- [ ] setup.py 中版本号已更新
- [ ] 发布说明已准备
- [ ] 已在 git 中打标签
- [ ] 包已构建和测试
```

### 3. 发布命令

```bash
# 更新版本
bump2version patch  # 或 minor、major

# 创建发布标签
git tag -a v1.2.3 -m "发布版本 1.2.3"

# 推送标签
git push origin v1.2.3

# 构建包
python setup.py sdist bdist_wheel

# 上传到 PyPI（如适用）
twine upload dist/*
```

## 最佳实践

### 1. 错误处理

```python
class ProcessingError(Exception):
    """处理错误的自定义异常"""
    pass

def robust_process(data):
    """具有完善错误处理的数据处理"""
    try:
        # 验证输入
        if not validate_data(data):
            raise ValueError("无效的数据格式")
        
        # 处理
        result = process_internal(data)
        
        # 验证输出
        if not validate_result(result):
            raise ProcessingError("处理产生了无效结果")
        
        return result
    
    except ValueError as e:
        logger.error(f"验证错误: {e}")
        raise
    except ProcessingError as e:
        logger.error(f"处理错误: {e}")
        # 尝试恢复或优雅降级
        return process_fallback(data)
    except Exception as e:
        logger.critical(f"意外错误: {e}", exc_info=True)
        raise
```

### 2. 配置管理

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessorConfig:
    """处理器配置"""
    batch_size: int = 10
    timeout: int = 300
    retry_count: int = 3
    gpu_enabled: bool = False
    
    def validate(self) -> bool:
        """验证配置"""
        if self.batch_size < 1:
            raise ValueError("批处理大小必须为正数")
        if self.timeout < 0:
            raise ValueError("超时时间必须非负")
        return True
```

### 3. 资源管理

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(resource_path):
    """资源处理的上下文管理器"""
    resource = None
    try:
        resource = acquire_resource(resource_path)
        yield resource
    finally:
        if resource:
            release_resource(resource)

# 使用方法
with managed_resource("/path/to/resource") as resource:
    process_with_resource(resource)
```

## 开发问题排查

### 常见问题

1. **导入错误**
```bash
# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

2. **依赖冲突**
```bash
# 使用 pip-tools 管理依赖
pip install pip-tools
pip-compile requirements.in
pip-sync
```

3. **测试失败**
```bash
# 清除测试缓存
pytest --cache-clear

# 运行更详细的输出
pytest -vvs
```

4. **开发时的内存问题**
```python
# 使用内存分析
import tracemalloc
tracemalloc.start()

# 你的代码
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## 资源链接

### 文档资料
- [Python 风格指南 (PEP 8)](https://pep8.org/)
- [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)
- [pytest 文档](https://docs.pytest.org/)

### 工具
- [Black - 代码格式化](https://black.readthedocs.io/)
- [Flake8 - 风格检查](https://flake8.pycqa.org/)
- [mypy - 类型检查](http://mypy-lang.org/)
- [pre-commit - Git 钩子](https://pre-commit.com/)

### 学习资源
- [Real Python 教程](https://realpython.com/)
- [Python 测试入门](https://realpython.com/python-testing/)
- [Effective Python](https://effectivepython.com/)