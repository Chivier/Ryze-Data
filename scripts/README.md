# Scripts Guide

All runnable scripts live under `scripts/`. Always run from the **project root** (`Ryze-Data/`).

## Overview

```
scripts/
├── README.md                   # This file
├── QUICKSTART.md               # Custom inference detailed guide
│
├── bench.sh                    # Benchmark orchestrator (one-click)
├── benchmark/                  # Benchmark sub-scripts
│   ├── _env.sh                # Shared config (sourced by others)
│   ├── start_services.sh     # Start vLLM QA servers
│   ├── stop_services.sh      # Stop vLLM QA servers
│   ├── run_benchmark.sh      # Run parallel benchmark evaluation
│   ├── generate_report.sh    # Merge and display reports
│   ├── start_vllm_pool.sh    # Start multi-GPU vLLM pool
│   ├── run_ocr_benchmark.sh  # Run OCR-precompute benchmark
│   └── stop_vllm_pool.sh     # Stop vLLM pool
│
├── run_custom_inference.py     # Custom model inference (OpenAI-compatible)
├── reorganize_glm_ocr.py       # Reorganize raw GLM-OCR output
├── calculate_results.py        # Calculate accuracy from results.jsonl
├── llm_based_verifier.py       # LLM-based answer verification
│
├── test_ocr_real.py            # Real-file OCR test runner
├── test_ocr_all.sh             # Run tests for all OCR models
│
└── utils/                      # Standalone OCR precompute scripts
    ├── README.md              # Detailed OCR precompute guide
    ├── _shared/               # Shared code (dataset loader, image utils)
    ├── deepseek_ocr_v1/       # DeepSeek-OCR v1
    ├── deepseek_ocr_v2/       # DeepSeek-OCR v2
    ├── marker/                # Marker
    ├── markitdown/            # MarkItDown
    ├── paddleocr/             # PaddleOCR (PP-OCRv5)
    └── glm_ocr/               # GLM-OCR (vLLM nightly)
```

---

## 1. Benchmark Toolchain

The benchmark system evaluates OCR models by running QA inference on ArxivQA and SlideVQA datasets through a Qwen3-VL-8B VLM served by vLLM.

### Prerequisites

| Item | Requirement |
|------|-------------|
| GPU | 4x NVIDIA A6000 (48GB) |
| Python | 3.10+ |
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| QA Model | Qwen3-VL-8B weights (local path) |
| vLLM | Installed and available |

### 1.1 First-Time Setup

```bash
scripts/bench.sh setup
```

This will check Python/uv/GPU/vLLM, run `uv sync`, and create `.bench.env` from the template.

Then edit `.bench.env`:

```bash
# .bench.env — required
QA_MODEL_PATH=/data/models/Qwen3-VL-8B

# Optional (defaults shown)
QA_MODEL_NAME=Qwen3-VL-8B
VLLM_GPU_A=2              # vLLM Server A GPU
VLLM_GPU_B=3              # vLLM Server B GPU
VLLM_PORT_A=8000
VLLM_PORT_B=8001
DATASETS="arxivqa slidevqa"
MAX_SAMPLES=0              # 0 = all samples
HEALTH_TIMEOUT=300         # vLLM startup timeout (seconds)
RESULTS_BASE=data/benchmark_results
SHARED_DATA_DIR=data/benchmark_data
```

All variables can also be overridden via environment, e.g. `MAX_SAMPLES=5 scripts/bench.sh run`.

### 1.2 One-Click Run (Recommended)

```bash
# Quick validation (5 samples per dataset)
scripts/bench.sh quick

# Full benchmark (all samples)
scripts/bench.sh full

# Specific dataset
scripts/bench.sh full --datasets arxivqa
```

`quick` and `full` automatically: start vLLM -> run evaluation -> merge reports -> stop vLLM. Press Ctrl+C to safely abort.

### 1.3 Step-by-Step Execution

For finer control or keeping vLLM running between iterations:

```bash
# Start vLLM servers
scripts/benchmark/start_services.sh

# Check status
scripts/bench.sh status

# Run benchmark (all models, all samples)
scripts/benchmark/run_benchmark.sh

# Limit samples or models
scripts/benchmark/run_benchmark.sh --max-samples 20
scripts/benchmark/run_benchmark.sh --datasets arxivqa --models deepseek-ocr,marker

# Generate report
scripts/benchmark/generate_report.sh

# Stop vLLM servers
scripts/benchmark/stop_services.sh
```

**Available models:**

| Model | GPU | vLLM Server |
|-------|-----|-------------|
| `baseline` | CPU | Server A (8000) |
| `deepseek-ocr` | GPU 0 | Server A (8000) |
| `deepseek-ocr-v2` | GPU 1 | Server B (8001) |
| `marker` | CPU | Server B (8001) |
| `markitdown` | CPU | Server A (8000) |
| `paddleocr` | GPU (optional) | Server A (8000) |
| `glm-ocr` | GPU | Server B (8001) |

### 1.4 Single-Model Debugging

```bash
scripts/bench.sh start
scripts/bench.sh single deepseek-ocr
scripts/bench.sh single marker --datasets slidevqa --max-samples 5
scripts/bench.sh stop
```

### 1.5 OCR-Precompute Benchmark

When OCR results are already precomputed in `data/ocr_precompute/`, run QA evaluation directly without real-time OCR:

```bash
# Start multi-GPU vLLM pool
scripts/benchmark/start_vllm_pool.sh \
  --model-path /data/models/Qwen3-VL-8B \
  --model-name Qwen3-VL-8B \
  --gpus 0,1,2,3 \
  --base-port 8000

# Run all 6 experiments
scripts/benchmark/run_ocr_benchmark.sh --dataset arxivqa

# Run specific experiments
scripts/benchmark/run_ocr_benchmark.sh \
  --dataset arxivqa \
  --experiments baseline,us \
  --max-samples 100

# Stop vLLM pool
scripts/benchmark/stop_vllm_pool.sh
```

Experiment names: `baseline` (vision-only), `baseline1` (DeepSeek v1), `baseline2` (DeepSeek v2), `baseline3` (MarkItDown), `baseline4` (GLM-OCR), `us` (Marker).

### 1.6 GPU Allocation

```
GPU 0  ──  DeepSeek-OCR v1      (~6 GB)
GPU 1  ──  DeepSeek-OCR v2      (~6 GB)
GPU 2  ──  vLLM Server A        (port 8000, Qwen3-VL-8B, ~38 GB)
GPU 3  ──  vLLM Server B        (port 8001, Qwen3-VL-8B, ~38 GB)
CPU    ──  Baseline / Marker / MarkItDown
```

Adjust GPU assignment in `.bench.env` (`VLLM_GPU_A`, `VLLM_GPU_B`). OCR model GPU assignment is managed by `run_benchmark.sh`.

### 1.7 Output Structure

```
data/benchmark_results/
├── run_baseline/              # Per-model result directories
│   ├── pdfs/                  # PDF cache
│   ├── baseline/
│   │   └── qa_results.jsonl   # QA cache (supports resume)
│   └── arxivqa_results.csv
├── run_deepseek-ocr/
├── run_deepseek-ocr-v2/
├── run_marker/
├── run_markitdown/
├── arxivqa_results.csv        # Merged report
├── arxivqa_results.md
├── slidevqa_results.csv
└── slidevqa_results.md

logs/benchmark/
├── latest -> 20260211_143000  # Symlink to latest logs
└── 20260211_143000/
    ├── vllm_a.log
    ├── vllm_b.log
    ├── arxivqa_baseline.log
    └── ...
```

### 1.8 Quick Reference

| Action | Command |
|--------|---------|
| Setup | `scripts/bench.sh setup` |
| Quick validation | `scripts/bench.sh quick` |
| Full benchmark | `scripts/bench.sh full` |
| ArxivQA only | `scripts/bench.sh full --datasets arxivqa` |
| Start vLLM | `scripts/benchmark/start_services.sh` |
| Stop vLLM | `scripts/benchmark/stop_services.sh` |
| Check status | `scripts/bench.sh status` |
| Run evaluation | `scripts/benchmark/run_benchmark.sh` |
| Merge report | `scripts/benchmark/generate_report.sh` |
| Start vLLM pool | `scripts/benchmark/start_vllm_pool.sh --model-path ... --gpus 0,1` |
| Precompute eval | `scripts/benchmark/run_ocr_benchmark.sh --dataset arxivqa` |
| Stop vLLM pool | `scripts/benchmark/stop_vllm_pool.sh` |
| Single model debug | `scripts/bench.sh single deepseek-ocr` |
| Clean data | `scripts/bench.sh clean` |

---

## 2. Custom Inference

`run_custom_inference.py` runs VLM inference on ArxivQA/SlideVQA with optional OCR augmentation, using any OpenAI-compatible endpoint.

See [QUICKSTART.md](QUICKSTART.md) for the full guide including data layout, all CLI options, and API key configuration.

### Quick Examples

```bash
# Baseline (vision-only, local vLLM)
python scripts/run_custom_inference.py \
    --dataset arxivqa \
    --experiment baseline \
    --endpoints http://localhost:8000/v1 \
    --max-samples 50

# With DeepSeek-OCR v1 augmentation
python scripts/run_custom_inference.py \
    --dataset arxivqa \
    --experiment baseline1 \
    --endpoints http://localhost:8000/v1

# GLM-OCR (reorganize first)
python scripts/reorganize_glm_ocr.py --dataset all
python scripts/run_custom_inference.py \
    --dataset arxivqa \
    --experiment baseline4 \
    --endpoints http://localhost:8000/v1

# Public API (OpenAI, SiliconFlow, etc.)
python scripts/run_custom_inference.py \
    --dataset arxivqa \
    --experiment baseline \
    --endpoints https://api.openai.com/v1 \
    --model gpt-4o \
    --api-key sk-xxxxxxxxxxxx
```

**Experiments:** `baseline` (vision-only), `baseline1` (DeepSeek v1), `baseline2` (DeepSeek v2), `baseline3` (MarkItDown), `baseline4` (GLM-OCR), `us` (Marker).

**Output:** `data/custom_inference/{dataset}/{experiment}/results.jsonl`

---

## 3. OCR Precompute (Standalone Scripts)

Each OCR model has an isolated venv under `scripts/utils/`. See [utils/README.md](utils/README.md) for the full guide including all CLI flags and troubleshooting.

### Quick Examples

```bash
# Setup any model
cd scripts/utils/markitdown && bash setup_env.sh

# Run OCR (5 samples)
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5

# Marker with pipelined workers
cd scripts/utils/marker && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --workers 4 --gpu cpu

# DeepSeek v1 on GPU 0
cd scripts/utils/deepseek_ocr_v1 && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 0

# PaddleOCR on GPU 2
cd scripts/utils/paddleocr && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 2

# GLM-OCR on GPU 3
cd scripts/utils/glm_ocr && bash setup_env.sh
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 3
```

**Output:** `data/ocr_precompute/{model}/{dataset}/{sample_id}/{sample_id}.md`

Scripts automatically skip existing output (resume-safe). Delete a sample's output directory to re-process it.

---

## 4. Testing Scripts

### test_ocr_all.sh — Run All OCR Model Tests

```bash
# Test all 5 OCR models with 3 PDFs each
bash scripts/test_ocr_all.sh 3
```

Runs `test_ocr_real.py` for each model sequentially. Reports PASS/FAIL/SKIP per model.

### test_ocr_real.py — Single Model Test

```bash
# Test a specific model
python scripts/test_ocr_real.py --model markitdown --num-pdfs 3
python scripts/test_ocr_real.py --model deepseek-ocr --num-pdfs 5
```

Uses real ArxivQA PDFs from `data/benchmark_data/ocr_pdfs/arxivqa/`. Validates that the model can process PDFs end-to-end without crashing.

---

## 5. Utility Scripts

### reorganize_glm_ocr.py

Reorganizes raw GLM-OCR output (flat file layout) into the standard `{sample_id}/{sample_id}.md` structure expected by the inference pipeline.

```bash
# Preview changes
python scripts/reorganize_glm_ocr.py --dataset all --dry-run

# Execute
python scripts/reorganize_glm_ocr.py --dataset all
```

### calculate_results.py

Calculates accuracy metrics from inference result files.

```bash
python scripts/calculate_results.py
```

### llm_based_verifier.py

Uses an LLM to verify answer correctness (for free-text QA where exact match is insufficient).

```bash
python scripts/llm_based_verifier.py
```

---

## 6. FAQ

**Q: vLLM startup timeout?**
Check `logs/benchmark/latest/vllm_a.log`. Common causes: wrong model path, insufficient GPU memory, CUDA version mismatch. Increase timeout with `HEALTH_TIMEOUT=600`.

**Q: One OCR model failed — do I need to re-run everything?**
No. Each model's results are cached independently. Fix the issue and re-run only that model:
```bash
scripts/benchmark/run_benchmark.sh --models deepseek-ocr-v2
scripts/benchmark/generate_report.sh
```

**Q: Port already in use?**
```bash
scripts/benchmark/stop_services.sh
# Or change ports in .bench.env
```

**Q: Datasets download automatically?**
Yes. ArxivQA and SlideVQA are loaded via HuggingFace `datasets`. Ensure network access or pre-set `HF_HOME`.

**Q: Different GPU allocation on different machines?**
Each machine has its own `.bench.env` (git-ignored). Adjust `VLLM_GPU_A`, `VLLM_GPU_B` as needed.

**Q: What's the relationship between `bench.sh` and `scripts/benchmark/*.sh`?**
`bench.sh` is a thin orchestrator. `start`/`stop`/`run`/`report` delegate directly to `scripts/benchmark/` sub-scripts. `quick`/`full` are convenience combos that call multiple sub-scripts in sequence. You can skip `bench.sh` entirely and call the sub-scripts directly.
