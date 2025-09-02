# Ryze-Data 数据处理框架

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-green.svg)](../zh-CN/)

一个综合性的科学文献数据处理流水线框架，提供从网络爬取到训练数据生成的端到端工作流。

[English Documentation](../../README.md) | [中文文档](../zh-CN/)

## 🌟 项目概述

Ryze-Data 是一个企业级的模块化框架，专门设计用于自动化科学文献的复杂处理流程，将文献提取、处理和转换为高质量的机器学习训练数据集。该框架以可扩展性、可靠性和可扩展性为核心，简化了从源到结构化输出的整个数据流水线。

### 核心特性

- **📚 智能网页爬取**：自动收集来自 Nature 等来源的科学文章
- **📄 高级 PDF 处理**：具有容错和重试机制的并行下载
- **🔍 先进的 OCR 技术**：使用 marker 引擎进行高精度文本和图像提取
- **🖼️ 上下文感知的图像分析**：智能提取图像及其周围上下文
- **🤖 多模态问答生成**：自动生成文本和视觉问答对
- **🔧 灵活配置**：支持热重载的基于环境的配置
- **📊 实时监控**：内置指标和日志记录，实现流水线可观测性
- **🚀 生产就绪**：支持分布式处理和检查点恢复

## 🚀 快速开始

### 系统要求

- Python 3.8 或更高版本
- CUDA 支持的 GPU（可选，用于加速 OCR）
- 建议 16GB+ RAM
- 100GB+ 可用磁盘空间用于数据存储

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-username/ryze-data.git
cd ryze-data

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows 系统: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境
cp .env.example .env
# 编辑 .env 设置 API 密钥和路径
nano .env
```

### 基本用法

```bash
# 运行完整流水线
python -m src.cli.main pipeline

# 或者运行单个阶段
python -m src.cli.main scrape      # 爬取文章元数据
python -m src.cli.main download    # 下载 PDF 文件
python -m src.cli.main ocr         # 提取文本和图像
python -m src.cli.main extract     # 提取带上下文的图像
python -m src.cli.main generate-qa # 生成问答对
```

### 数据检查

```bash
# 查看流水线状态
python -m src.cli.main inspect all

# 检查特定阶段并采样
python -m src.cli.main inspect stage ocr --sample 5 --detailed

# 检查配置
python -m src.cli.main config-show

# 查看处理统计
python -m src.cli.main inspect stats
```

## 📂 项目结构

```
Ryze-Data/
├── src/                    # 源代码
│   ├── cli/               # 命令行接口
│   ├── scrapers/          # 网页爬取模块
│   ├── downloaders/       # PDF 下载管理器
│   ├── processors/        # 数据处理引擎
│   ├── generators/        # 问答生成模块
│   ├── config_manager.py  # 配置管理
│   └── pipeline_manager.py # 流水线编排
├── prompts/               # LLM 提示词模板
│   ├── text/             # 文本问答提示词
│   └── vision/           # 视觉问答提示词
├── tests/                 # 测试套件
│   ├── unit/             # 单元测试
│   └── integration/      # 集成测试
├── docs/                  # 文档
│   ├── architecture.md   # 系统架构
│   ├── configuration.md  # 配置指南
│   ├── api-reference.md  # API 文档
│   └── zh-CN/           # 中文文档
├── scripts/              # 实用脚本
├── data-sample/          # 测试用样本数据
├── .env.example          # 环境变量模板
├── config.example.json   # 配置文件模板
├── requirements.txt      # Python 依赖
└── README.md            # 主文档
```

## 🔧 配置系统

Ryze-Data 采用多层配置系统：

1. **默认值**（代码中定义）
2. **配置文件**（config.json）
3. **环境变量**（.env）
4. **命令行参数**

### 快速配置

```bash
# 必要的环境变量
OPENAI_API_KEY=sk-...           # 用于问答生成
RYZE_DATA_ROOT=./data           # 数据存储位置
RYZE_NUM_WORKERS=4              # 并行处理线程数
RYZE_GPU_ENABLED=true           # 启用 GPU 加速
```

详细配置选项请参见[配置指南](configuration.md)。

## 📊 流水线架构

系统采用模块化、基于阶段的架构：

```
网页源 → 爬取 → 元数据
         ↓
      下载 → PDF文件
         ↓
      OCR → 文本+图像
         ↓
      处理 → 结构化数据
         ↓
      生成 → 问答数据集
```

每个阶段具有以下特点：
- **独立性**：可以单独运行
- **可恢复**：支持检查点恢复
- **可扩展**：支持分布式处理
- **可观测**：提供指标和日志

详细架构说明请参见[架构文档](architecture.md)。

## 🧪 测试

```bash
# 运行所有测试
python -m pytest

# 运行并生成覆盖率报告
python -m pytest --cov=src --cov-report=html

# 快速冒烟测试
python run_tests.py quick

# 使用样本数据测试
python run_tests.py sample

# 运行特定类别的测试
python -m pytest tests/unit/
python -m pytest tests/integration/
```

## 📚 文档导航

### 核心文档
- [架构设计](architecture.md) - 系统架构和设计决策
- [配置指南](configuration.md) - 详细配置选项
- [API 参考](api-reference.md) - 完整的 API 文档
- [数据格式](data-formats.md) - 数据结构规范
- [开发指南](development.md) - 贡献和扩展指南
- [故障排查](troubleshooting.md) - 常见问题和解决方案

### 使用指南
- [快速开始](quickstart.md) - 5分钟上手教程
- [高级用法](advanced-usage.md) - 高级功能和技巧
- [性能优化](performance.md) - 性能调优指南
- [部署指南](deployment.md) - 生产环境部署

## 🤝 贡献指南

我们欢迎各种形式的贡献！请查看我们的[贡献指南](development.md)了解详情。

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 设置预提交钩子
pre-commit install

# 运行代码格式化
black src/ tests/

# 运行代码检查
flake8 src/ tests/
pylint src/
```

## 📈 性能指标

在标准硬件上的典型处理指标：

| 阶段 | 文档/小时 | GPU 加速 |
|------|----------|----------|
| 爬取 | 1000+ | 不适用 |
| 下载 | 200-500 | 不适用 |
| OCR | 50-100 | 2-3倍 |
| 问答生成 | 100-200 | 不适用 |

### 性能优化建议

1. **硬件优化**
   - 使用 SSD 存储提升 I/O 性能
   - 配置足够的 RAM 避免内存交换
   - 使用 GPU 加速 OCR 处理

2. **软件优化**
   - 调整批处理大小
   - 增加并行工作线程
   - 启用结果缓存

3. **网络优化**
   - 使用代理服务器
   - 配置请求重试策略
   - 实施速率限制

## 🔒 安全性

- API 密钥存储在环境变量中
- 支持凭证轮换
- 日志中不包含敏感数据
- 可配置的数据保留策略
- 支持审计日志

### 安全最佳实践

1. 定期更新依赖库
2. 使用密钥管理服务
3. 实施访问控制
4. 加密敏感数据
5. 定期安全审计

## 📝 许可证

本项目采用 GNU Affero General Public License v3.0 许可证 - 详情请见 [LICENSE](../../LICENSE) 文件。

## 🙏 致谢

- [Marker](https://github.com/VikParuchuri/marker) - OCR 引擎
- [OpenAI](https://openai.com) - LLM API
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - 网页爬取
- 所有贡献者和用户

## 📧 支持

如有问题、建议或贡献：
- 提交 [Issue](https://github.com/your-username/ryze-data/issues)
- 查看[故障排查指南](troubleshooting.md)
- 查阅[常见问题](faq.md)

## 🔗 相关链接

- [项目主页](https://github.com/your-username/ryze-data)
- [发布说明](https://github.com/your-username/ryze-data/releases)
- [路线图](https://github.com/your-username/ryze-data/projects)

---

<p align="center">
  由 Ryze-Data 团队用 ❤️ 打造
</p>