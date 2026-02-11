# Benchmark 脚本使用指南

OCR Benchmark 工具链使用说明 — 从环境配置到生成最终报告的完整流程。

## 概览

Benchmark 工具链由 4 个独立脚本组成，各自负责一个阶段：

```
scripts/benchmark/
├── _env.sh               # 共享配置（被其他脚本 source）
├── start_services.sh     # 阶段 1：启动 vLLM QA 服务
├── stop_services.sh      # 停止 vLLM QA 服务
├── run_benchmark.sh      # 阶段 2：运行并行 Benchmark 评估
└── generate_report.sh    # 阶段 3：合并并展示报告

scripts/bench.sh           # 编排器（封装上述脚本）
```

**两种使用方式：**

- **一键运行** — 用 `bench.sh` 编排器自动完成全流程
- **分步执行** — 单独调用每个脚本，灵活控制每个阶段

## 前提条件

| 项目 | 要求 |
|------|------|
| GPU | 4x NVIDIA A6000 (48GB) |
| Python | 3.10+ |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| QA 模型 | Qwen3-VL-8B 权重（本地路径） |
| vLLM | 已安装且可用 |

## 1. 首次配置

### 1.1 自动检测环境

```bash
scripts/bench.sh setup
```

该命令会：
- 检查 Python、uv、GPU、curl、vLLM 是否就绪
- 运行 `uv sync` 安装基础 + benchmark 依赖
- 从 `.bench.env.example` 创建 `.bench.env` 配置文件

### 1.2 编辑配置文件

```bash
vim .bench.env
```

**必须设置 `QA_MODEL_PATH`**，其余项可保留默认值：

```bash
# .bench.env

# ── 必填 ──
QA_MODEL_PATH=/data/models/Qwen3-VL-8B

# ── 可选（以下为默认值） ──
QA_MODEL_NAME=Qwen3-VL-8B
VLLM_GPU_A=2              # vLLM Server A 使用的 GPU
VLLM_GPU_B=3              # vLLM Server B 使用的 GPU
VLLM_PORT_A=8000
VLLM_PORT_B=8001
DATASETS="arxivqa slidevqa"
MAX_SAMPLES=0              # 0 = 全部样本
HEALTH_TIMEOUT=300         # vLLM 启动超时（秒）
RESULTS_BASE=data/benchmark_results
SHARED_DATA_DIR=data/benchmark_data
```

> 所有变量也可以通过环境变量覆盖，例如 `MAX_SAMPLES=5 scripts/bench.sh run`。

### 1.3 安装 OCR 模型依赖

```bash
uv sync --extra deepseek-ocr    # DeepSeek v1/v2 (torch + transformers)
uv sync --extra markitdown       # MarkItDown
```

Marker 需要独立安装，参考 [Marker 官方文档](https://github.com/VikParuchuri/marker)。

## 2. 一键运行（推荐）

### 快速验证（5 个样本）

```bash
scripts/bench.sh quick
```

自动执行：启动 vLLM → 每个数据集跑 5 个样本 → 合并报告 → 停止 vLLM。

### 完整 Benchmark

```bash
scripts/bench.sh full
```

自动执行：启动 vLLM → 跑全部样本 → 合并报告 → 停止 vLLM。

### 指定数据集

```bash
scripts/bench.sh full --datasets arxivqa
scripts/bench.sh quick --datasets "arxivqa slidevqa"
```

### Ctrl+C 安全中断

`quick` 和 `full` 模式下按 Ctrl+C 会自动终止所有子进程并停止 vLLM 服务。

## 3. 分步执行

适合需要精细控制、反复调试、或保持 vLLM 常驻的场景。

### 3.1 启动 vLLM 服务

```bash
scripts/benchmark/start_services.sh
```

该脚本会：
- 检查 `QA_MODEL_PATH` 是否有效
- 检测端口是否被占用
- 在 GPU 2/3 上启动两个 vLLM 实例
- 轮询 `/v1/models` 健康检查（最多 300 秒）
- 将 PID 写入 `logs/benchmark/.pids/` 供后续管理

**查看服务状态：**

```bash
scripts/benchmark/start_services.sh --status
# 或
scripts/bench.sh status
```

输出示例：

```
  vLLM Server Status
  ─────────────────────────────────────────
  Server-a: RUNNING (PID 12345, port 8000) - healthy
  Server-b: RUNNING (PID 12346, port 8001) - healthy

  GPU Usage
  ─────────────────────────────────────────
  0, 0 %, 5 MiB, 49140 MiB
  1, 0 %, 5 MiB, 49140 MiB
  2, 85 %, 38000 MiB, 49140 MiB
  3, 85 %, 38000 MiB, 49140 MiB
```

### 3.2 运行 Benchmark

**前提：vLLM 服务已启动。**

```bash
# 全部模型，全部样本
scripts/benchmark/run_benchmark.sh

# 限制样本数
scripts/benchmark/run_benchmark.sh --max-samples 20

# 只跑一个数据集
scripts/benchmark/run_benchmark.sh --datasets arxivqa

# 只跑部分模型
scripts/benchmark/run_benchmark.sh --models deepseek-ocr,marker

# 组合使用
scripts/benchmark/run_benchmark.sh --datasets arxivqa --max-samples 10 --models baseline,marker
```

**可用选项：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--datasets <list>` | `"arxivqa slidevqa"` | 空格分隔的数据集列表 |
| `--max-samples <n>` | `0`（全部） | 每个数据集最大样本数 |
| `--models <list>` | 全部 5 个 | 逗号分隔的模型子集 |

**可用模型：**

| 模型名 | GPU | vLLM Server |
|--------|-----|-------------|
| `baseline` | CPU | Server A (8000) |
| `deepseek-ocr` | GPU 0 | Server A (8000) |
| `deepseek-ocr-v2` | GPU 1 | Server B (8001) |
| `marker` | CPU | Server B (8001) |
| `markitdown` | CPU | Server A (8000) |

同一数据集内 5 个模型并行执行，数据集间串行执行。

### 3.3 生成报告

```bash
# 所有数据集
scripts/benchmark/generate_report.sh

# 指定数据集
scripts/benchmark/generate_report.sh arxivqa

# 指定结果路径
scripts/benchmark/generate_report.sh --results-base /tmp/my_results arxivqa
```

输出示例：

```
       Merged Benchmark Results: arxivqa
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Model          ┃ Accuracy ┃ Avg OCR Time (s) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Baseline (VLM) │   0.7200 │                - │
│ DeepSeek v1    │   0.8100 │           2.3400 │
│ DeepSeek v2    │   0.8500 │           1.9200 │
│ MarkItDown     │   0.6900 │           0.4200 │
│ Marker (Ours)  │   0.7800 │           3.1500 │
└────────────────┴──────────┴──────────────────┘
```

同时生成文件：
- `data/benchmark_results/arxivqa_results.csv`
- `data/benchmark_results/arxivqa_results.md`

### 3.4 停止 vLLM 服务

```bash
scripts/benchmark/stop_services.sh
```

先发送 SIGTERM，等待 10 秒后 SIGKILL 强制终止。

## 4. 单模型调试

调试某一条 OCR 路径时，不需要运行全部 5 个模型：

```bash
# 启动服务（如果还没启动）
scripts/bench.sh start

# 调试 deepseek-ocr（默认 10 个样本）
scripts/bench.sh single deepseek-ocr

# 调试 marker，指定数据集和样本数
scripts/bench.sh single marker --datasets slidevqa --max-samples 5

# 调试完毕停止服务
scripts/bench.sh stop
```

或直接使用 `run_benchmark.sh`：

```bash
scripts/benchmark/run_benchmark.sh --models deepseek-ocr --max-samples 10 --datasets arxivqa
```

## 5. 典型工作流

### 场景 A：首次完整实验

```bash
# 1. 环境配置
scripts/bench.sh setup
vim .bench.env                    # 设置 QA_MODEL_PATH

# 2. 快速验证（确认流程正常）
scripts/bench.sh quick

# 3. 完整实验
scripts/bench.sh full
```

### 场景 B：反复调试，保持 vLLM 常驻

```bash
# 启动一次
scripts/benchmark/start_services.sh

# 多次迭代
scripts/benchmark/run_benchmark.sh --max-samples 10
scripts/benchmark/run_benchmark.sh --max-samples 50
scripts/benchmark/run_benchmark.sh                     # 全量

# 随时查看报告
scripts/benchmark/generate_report.sh

# 结束时停止
scripts/benchmark/stop_services.sh
```

### 场景 C：只重跑失败的模型

```bash
# 假设 deepseek-ocr-v2 失败了
# 查看日志
cat logs/benchmark/latest/arxivqa_deepseek-ocr-v2.log

# 修复问题后，只重跑该模型（已完成的样本自动跳过）
scripts/benchmark/run_benchmark.sh --models deepseek-ocr-v2

# 重新合并报告
scripts/benchmark/generate_report.sh
```

### 场景 D：只重新合并报告

```bash
scripts/benchmark/generate_report.sh arxivqa
```

## 6. GPU 分配方案

```
GPU 0  ──  DeepSeek-OCR v1      (~6 GB)
GPU 1  ──  DeepSeek-OCR v2      (~6 GB)
GPU 2  ──  vLLM Server A        (port 8000, Qwen3-VL-8B, ~38 GB)
GPU 3  ──  vLLM Server B        (port 8001, Qwen3-VL-8B, ~38 GB)
CPU    ──  Baseline / Marker / MarkItDown
```

如需调整 GPU 分配，修改 `.bench.env` 中的 `VLLM_GPU_A` / `VLLM_GPU_B`。OCR 模型的 GPU 分配由 `run_benchmark.sh` 内部管理。

## 7. 输出结构

```
data/benchmark_results/
├── run_baseline/              # 各模型独立结果目录
│   ├── pdfs/                  # PDF 缓存
│   ├── baseline/
│   │   └── qa_results.jsonl   # QA 缓存（支持断点续跑）
│   └── arxivqa_results.csv
├── run_deepseek-ocr/
├── run_deepseek-ocr-v2/
├── run_marker/
├── run_markitdown/
├── arxivqa_results.csv        # ← 合并报告
├── arxivqa_results.md
├── slidevqa_results.csv
└── slidevqa_results.md

logs/benchmark/
├── latest -> 20260211_143000  # 符号链接到最新日志
└── 20260211_143000/
    ├── vllm_a.log
    ├── vllm_b.log
    ├── arxivqa_baseline.log
    ├── arxivqa_deepseek-ocr.log
    ├── arxivqa_deepseek-ocr-v2.log
    ├── arxivqa_marker.log
    ├── arxivqa_markitdown.log
    └── merge_arxivqa.log
```

## 8. OCR 预处理（独立脚本）

除了 Benchmark 流程中的在线 OCR，还可以使用 `scripts/utils/` 下的独立脚本提前批量预处理 OCR 结果。每个模型有自己的目录和独立虚拟环境。

### 目录结构

```
scripts/utils/
├── _shared/                    # 共享工具
│   ├── dataset_loader.py       # HuggingFace 数据集加载
│   └── image_utils.py          # 图像转 PDF
├── deepseek_ocr_v1/            # DeepSeek-OCR v1
│   ├── run_ocr.py
│   ├── requirements.txt
│   └── setup_env.sh
├── deepseek_ocr_v2/            # DeepSeek-OCR v2
├── marker/                     # Marker
└── markitdown/                 # MarkItDown
```

### 使用流程

```bash
# 1. 设置环境
cd scripts/utils/markitdown && bash setup_env.sh

# 2. 快速测试（5 个样本）
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5

# 3. 检查输出
ls data/ocr_precompute/markitdown/arxivqa/

# 4. 完整运行
.venv/bin/python run_ocr.py --dataset arxivqa
.venv/bin/python run_ocr.py --dataset slidevqa
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | `arxivqa` 或 `slidevqa` |
| `--output-dir` | `data/ocr_precompute/{model}/{dataset}` | 输出目录 |
| `--cache-dir` | `data/benchmark_data` | 共享图像缓存 |
| `--max-samples` | `0`（全部） | 最大样本数 |
| `--gpu` | `0` | GPU ID（仅 DeepSeek） |

### 输出结构

```
data/ocr_precompute/{model_name}/{dataset}/
├── arxivqa_0/arxivqa_0.md
├── arxivqa_1/arxivqa_1.md
└── ...

data/benchmark_data/ocr_pdfs/{dataset}/   # Marker/MarkItDown 共享 PDF 缓存
├── arxivqa_0.pdf
└── ...
```

脚本支持断点续传——如果 `{sample_id}.md` 已存在则自动跳过。

## 9. 命令速查表

| 操作 | 命令 |
|------|------|
| 环境配置 | `scripts/bench.sh setup` |
| 快速验证 | `scripts/bench.sh quick` |
| 完整实验 | `scripts/bench.sh full` |
| 只跑 arxivqa | `scripts/bench.sh full --datasets arxivqa` |
| 启动服务 | `scripts/benchmark/start_services.sh` |
| 停止服务 | `scripts/benchmark/stop_services.sh` |
| 查看状态 | `scripts/bench.sh status` |
| 运行评估 | `scripts/benchmark/run_benchmark.sh` |
| 合并报告 | `scripts/benchmark/generate_report.sh` |
| 单模型调试 | `scripts/bench.sh single deepseek-ocr` |
| 清理数据 | `scripts/bench.sh clean` |

## 9. 常见问题

**Q: vLLM 启动超时怎么办？**

查看日志：
```bash
cat logs/benchmark/latest/vllm_a.log
```
常见原因：模型路径错误、GPU 显存不足、CUDA 版本不兼容。可通过 `HEALTH_TIMEOUT=600` 延长超时时间。

**Q: 某个 OCR 模型失败了，需要重跑全部吗？**

不需要。每个模型的结果在独立的 `run_*` 目录中缓存（JSONL 格式）。修复问题后只重跑失败的模型：
```bash
scripts/benchmark/run_benchmark.sh --models deepseek-ocr-v2
scripts/benchmark/generate_report.sh
```

**Q: 端口被占用怎么办？**

方法 1：找到并停止占用进程：
```bash
scripts/benchmark/stop_services.sh
```

方法 2：修改 `.bench.env` 中的端口：
```bash
VLLM_PORT_A=8010
VLLM_PORT_B=8011
```

**Q: 数据集首次运行会自动下载吗？**

是的。ArxivQA 和 SlideVQA 通过 HuggingFace `datasets` 库加载，首次运行自动下载到 HuggingFace cache。确保网络可用或提前设置 `HF_HOME`。

**Q: 如何在不同机器上使用不同的 GPU 分配？**

每台机器维护自己的 `.bench.env`（已在 `.gitignore` 中忽略），修改 `VLLM_GPU_A`、`VLLM_GPU_B` 即可。

**Q: `bench.sh` 和 `scripts/benchmark/*.sh` 的关系是什么？**

`bench.sh` 是一个薄编排层，`start`/`stop`/`run`/`report` 命令直接委托给 `scripts/benchmark/` 下的对应脚本。`quick`/`full` 是便捷组合，按顺序调用多个子脚本。你可以完全不使用 `bench.sh`，直接调用子脚本。
