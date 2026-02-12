# OCR Precompute — Quick Start

Standalone scripts to run 4 OCR models on ArxivQA / SlideVQA datasets. Each model has its own isolated venv so there are no dependency conflicts.

## Directory Layout

```
scripts/utils/
├── _shared/              # Shared code (dataset loader, image→PDF)
├── deepseek_ocr_v1/      # DeepSeek-OCR v1  (GPU, direct image)
├── deepseek_ocr_v2/      # DeepSeek-OCR v2  (GPU, direct image)
├── marker/               # Marker           (pipelined image→PDF→OCR, CPU/GPU selectable)
└── markitdown/           # MarkItDown        (CPU, needs PDF)
```

## 1. Pick a model and set up its environment

```bash
# From the project root:
cd scripts/utils/markitdown   # or marker / deepseek_ocr_v1 / deepseek_ocr_v2
bash setup_env.sh
```

This creates a local `.venv` with all dependencies via `uv`.

## 2. Run OCR

### MarkItDown (CPU)

```bash
# Quick test — 5 samples from ArxivQA
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5

# Full run — all ArxivQA samples
.venv/bin/python run_ocr.py --dataset arxivqa

# SlideVQA
.venv/bin/python run_ocr.py --dataset slidevqa
```

### Marker (Pipelined OCR, CPU/GPU selectable)

```bash
cd scripts/utils/marker
bash setup_env.sh

# Quick test with pipelined workers
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 5 --workers 4

# Pin Marker to GPU 0
.venv/bin/python run_ocr.py --dataset arxivqa --workers 4 --gpu 0

# Force CPU mode
.venv/bin/python run_ocr.py --dataset arxivqa --workers 4 --gpu cpu
```

### GPU models (DeepSeek v1 / v2)

```bash
cd scripts/utils/deepseek_ocr_v1
bash setup_env.sh

# Specify which GPU to use (requires ~6 GB VRAM)
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 0

# Use a different GPU
.venv/bin/python run_ocr.py --dataset slidevqa --gpu 1

# Optional backend control (DeepSeek only)
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 1 --backend transformers
```

## 3. Check output

```bash
ls data/ocr_precompute/markitdown/arxivqa/
# arxivqa_0/  arxivqa_1/  arxivqa_2/  ...

cat data/ocr_precompute/markitdown/arxivqa/arxivqa_0/arxivqa_0.md
```

## CLI Reference

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *required* | `arxivqa` or `slidevqa` |
| `--output-dir` | `data/ocr_precompute/{model}/{dataset}` | Where to write `.md` files |
| `--cache-dir` | `data/benchmark_data` | Shared image cache (downloaded from HF) |
| `--max-samples` | `0` (all) | Limit number of samples |
| `--hf-endpoint` | *(unset)* | Optional HF endpoint override (e.g. `https://hf-mirror.com`) |

### Marker-Only Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | `0` | Pipeline worker count (`0` = auto from detected devices) |
| `--gpu` | *(unset)* | Sets `CUDA_VISIBLE_DEVICES` (e.g. `0`, `0,1`, or `cpu`) |

### DeepSeek-Only Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | `0` | GPU device ID |
| `--backend` | `v1:transformers / v2:transformers` | `auto`, `vllm`, `transformers` |

## Output Structure

```
data/ocr_precompute/
├── markitdown/
│   ├── arxivqa/
│   │   ├── arxivqa_0/arxivqa_0.md
│   │   ├── arxivqa_1/arxivqa_1.md
│   │   └── ...
│   └── slidevqa/
│       └── ...
├── marker/
├── deepseek_ocr_v1/
└── deepseek_ocr_v2/
```

Marker and MarkItDown also cache intermediate PDFs at:

```
data/benchmark_data/ocr_pdfs/{dataset}/
├── arxivqa_0.pdf
└── ...
```

## Resume / Re-run

Scripts **automatically skip** any sample whose output `.md` already exists. To re-process a sample, delete its output directory first:

```bash
rm -rf data/ocr_precompute/markitdown/arxivqa/arxivqa_0
.venv/bin/python run_ocr.py --dataset arxivqa --max-samples 1
```

## Run All 4 Models in Parallel

```bash
# Terminal 1 — MarkItDown (CPU)
cd scripts/utils/markitdown
.venv/bin/python run_ocr.py --dataset arxivqa

# Terminal 2 — Marker (pipelined, CPU)
cd scripts/utils/marker
.venv/bin/python run_ocr.py --dataset arxivqa --workers 4 --gpu cpu

# Terminal 3 — DeepSeek v1 (GPU 0)
cd scripts/utils/deepseek_ocr_v1
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 0

# Terminal 4 — DeepSeek v2 (GPU 1)
cd scripts/utils/deepseek_ocr_v2
.venv/bin/python run_ocr.py --dataset arxivqa --gpu 1
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| HuggingFace download fails | `export HF_ENDPOINT=https://hf-mirror.com` |
| Marker model download fails (`models.datalab.to`) | Check DNS/network or run with pre-cached Marker/Surya models |
| `vLLM` import/installation fails | Install a CUDA-matching `vllm` wheel (see vLLM install docs) |
| DeepSeek backend mismatch | Use `--backend auto|vllm|transformers` (v2 defaults to `transformers`) |
| CUDA OOM on DeepSeek | Use a GPU with ≥ 6 GB VRAM, or try `--gpu <other_id>` |
| `marker_single` not found | Activate the marker `.venv`: `.venv/bin/marker_single` |
| Stale output / want re-run | Delete `data/ocr_precompute/{model}/{dataset}/{sample_id}/` |
