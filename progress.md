# Ryze-Data 项目进度

> 最后更新: 2026-02-11 | 当前版本: v1.0.0

## 项目概述

Ryze-Data 是一个面向科学论文的综合数据处理框架，涵盖从论文抓取、PDF 下载、OCR 转换、图表提取到 QA 数据生成的完整流水线，并提供 OCR 质量基准评估系统。

## 模块完成状态

| 模块 | 状态 | 任务 ID | 说明 |
|------|------|---------|------|
| **配置管理** (`config_manager.py`) | ✅ 完成 | — | 单例 ConfigManager，JSON + 环境变量 |
| **数据抓取** (`scrapers/`) | ✅ 完成 | — | Nature 论文元数据抓取 |
| **PDF 下载** (`downloaders/`) | ✅ 完成 | — | 多线程并行下载 |
| **OCR 框架** (`ocr/`) | ✅ 完成 | RD-019..RD-026 | 可扩展 OOP 设计 |
| **Marker OCR** (`ocr/marker_ocr.py`) | ✅ 完成 | RD-022 | CLI wrapper (marker_single / marker_chunk_convert) |
| **DeepSeek OCR v1** (`ocr/deepseek_ocr.py`) | ✅ 完成 | RD-027..RD-028 | 本地 Transformers 推理 |
| **DeepSeek OCR v2** (`ocr/deepseek_ocr_v2.py`) | ✅ 完成 | RD-029 | 本地 Transformers 推理 |
| **MarkItDown OCR** (`ocr/markitdown_ocr.py`) | ✅ 完成 | RD-034 | Microsoft markitdown 库集成 |
| **pdf2md OCR** (`ocr/pdf2md_ocr.py`) | 🔲 Stub | RD-023 | 未实现，仅注册占位 |
| **图表提取** (`processors/`) | ✅ 完成 | — | Markdown → 图表 JSON |
| **QA 生成器** (`generators/`) | ✅ 完成 | RD-013..RD-018 | Text + Vision 双模式 |
| **API 负载均衡** (`api_key_balancer.py`) | ✅ 完成 | — | 多 API Key 线程池 |
| **CLI** (`cli/main.py`) | ✅ 完成 | RD-024, RD-041 | Click 命令组 |
| **Benchmark 评估系统** (`benchmark/`) | ✅ 完成 | RD-034..RD-043 | 完整 OCR 基准评估框架 |
| **文档** (`docs/`) | ✅ 完成 | RD-001..RD-012, RD-033 | 中英文文档 |

## 里程碑时间线

### Phase 1: 基础设施 (已完成)

- `82a301d` 项目初始化
- `7eec545` 配置系统、环境搭建
- `3973b45` chunked OCR 处理 + 状态监控
- `52eefb1` API Key 负载均衡器
- `05e927a` 并行 Vision 数据生成

### Phase 2: 模块化重构 (已完成)

- `9e4c088` 文档同步 [RD-001..RD-012]
- `494e0fc` QA 生成器模块 (text + vision) [RD-013..RD-018]
- `fa6ed62` OCR 模块 OOP 重构 [RD-019..RD-026]
- `08c67da` DeepSeek OCR v1/v2 实现 [RD-027..RD-032]
- `5028c1f` OCR 模型选择文档 [RD-033]

### Phase 3: Benchmark 评估系统 (已完成)

- `75f0002` OCR Benchmark 评估系统 [RD-034..RD-043]

## Benchmark 评估系统详情

### 实验路径设计

```
Path 0 (Baseline): PDF/Image → Qwen3-VL-8B (vision)     → Score
Path 1:            PDF/Image → DeepSeek OCR v1 → MD → Qwen3-VL-8B (text) → Score
Path 2:            PDF/Image → DeepSeek OCR v2 → MD → Qwen3-VL-8B (text) → Score
Path 3:            PDF/Image → MarkItDown      → MD → Qwen3-VL-8B (text) → Score
Path 4 (Ours):     PDF/Image → Marker          → MD → Qwen3-VL-8B (text) → Score
```

### 数据集

| 数据集 | 来源 | 样本量 | 题型 |
|--------|------|--------|------|
| ArxivQA | `MMInstruction/ArxivQA` | 5000+ | 多选题 |
| SlideVQA | `NTT-hil-insight/SlideVQA` | 5000+ | 自由文本 |

### 评估指标

| 指标 | 适用场景 |
|------|----------|
| Accuracy | 多选题 (ArxivQA) |
| Exact Match | 自由文本 (SlideVQA) |
| BLEU-4 | 自由文本 (SlideVQA) |
| ROUGE-L | 自由文本 (SlideVQA) |
| Token F1 | 自由文本 (SlideVQA) |
| Avg OCR Time | 所有路径 |

### 文件结构

```
src/benchmark/
├── __init__.py
├── datasets/
│   ├── __init__.py
│   ├── base.py              # BenchmarkSample, BaseBenchmarkDataset
│   ├── arxivqa.py            # ArxivQA loader
│   └── slidevqa.py           # SlideVQA loader
├── evaluator.py              # BenchmarkEvaluator (主编排器)
├── qa_client.py              # QwenQAClient (vision + text)
├── image_utils.py            # images_to_pdf
├── metrics.py                # 全部指标 (纯函数)
└── report.py                 # Rich table / CSV / Markdown

prompts/benchmark/
├── multiple_choice.txt
└── free_text.txt
```

### CLI 用法

```bash
# 运行评估
uv run python -m src.cli.main benchmark run \
  --dataset arxivqa \
  --ocr-models "marker,deepseek-ocr,deepseek-ocr-v2,markitdown" \
  --include-baseline \
  --max-samples 100 \
  --qa-model Qwen3-VL-8B \
  --qa-api-base http://localhost:8000/v1

# 查看报告
uv run python -m src.cli.main benchmark report \
  --dataset arxivqa --format csv
```

## 测试状态

```
总计: 177 passed, 3 failed (pre-existing), 36 errors (pre-existing config)
```

| 测试文件 | 测试数 | 状态 |
|----------|--------|------|
| `test_benchmark_metrics.py` | 36 | ✅ 全部通过 |
| `test_benchmark_datasets.py` | 9 | ✅ 全部通过 |
| `test_benchmark_qa_client.py` | 6 | ✅ 全部通过 |
| `test_benchmark_evaluator.py` | 7 | ✅ 全部通过 |
| `test_markitdown_ocr.py` | 6 | ✅ 全部通过 |
| `test_marker_ocr.py` | 11 | ✅ 全部通过 |
| `test_ocr.py` | 12 | ✅ 全部通过 |
| `test_ocr_stubs.py` | 8 | ✅ 全部通过 |
| `test_deepseek_ocr.py` | 17 | ✅ 全部通过 |
| `test_generators.py` | 26 | ✅ 全部通过 |
| `unit/test_config_manager.py` | 11 | ⚠️ 3 failed (pre-existing) |
| `unit/test_data_inspector.py` | 14 | ⚠️ errors (test config JSON) |
| `integration/test_full_pipeline.py` | 12 | ⚠️ errors (test config JSON) |
| `integration/test_ocr_cli.py` | 2 | ✅ 全部通过 |

**Pre-existing 问题**: `tests/config.test.json` 格式错误导致 36 个 integration/unit 测试 setup 失败，与当前开发工作无关。

## OCR 模型矩阵

| 模型 | MODEL_NAME | 依赖 | GPU | 状态 |
|------|-----------|------|-----|------|
| Marker | `marker` | `marker_single` CLI | 可选 | ✅ 完整实现 |
| DeepSeek v1 | `deepseek-ocr` | `torch`, `transformers` | 必须 | ✅ 完整实现 |
| DeepSeek v2 | `deepseek-ocr-v2` | `torch`, `transformers` | 必须 | ✅ 完整实现 |
| MarkItDown | `markitdown` | `markitdown>=0.1.0` | 否 | ✅ 完整实现 |
| pdf2md | `pdf2md` | — | — | 🔲 Stub |

## 依赖管理

```toml
# pyproject.toml optional dependencies
[project.optional-dependencies]
deepseek-ocr = ["torch", "transformers==4.46.3", ...]
markitdown = ["markitdown>=0.1.0"]
benchmark = ["datasets>=2.14.0"]
```

## 下一步计划

- [ ] 修复 `tests/config.test.json` 格式错误 (消除 pre-existing 测试失败)
- [ ] 实现 `pdf2md` OCR stub
- [ ] 在实际数据集上运行 benchmark 端到端评估
- [ ] 基于 benchmark 结果优化 OCR 管线选择策略
- [ ] 添加 benchmark 结果可视化 (图表)
