# OCR Module Extension Guide

Register custom OCR models in three steps.

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Subclass `BaseOCRModel` |
| 2 | Decorate with `@OCRRegistry.register` |
| 3 | Import in `src/ocr/__init__.py` |

## 1. Create Your Model

Create a file at `src/ocr/my_model_ocr.py`:

```python
from src.ocr.base_ocr import BaseOCRModel, OCRResult
from src.ocr.registry import OCRRegistry


@OCRRegistry.register
class MyModelOCR(BaseOCRModel):
    MODEL_NAME = "my-model"  # Used in --ocr-model CLI flag

    @property
    def name(self) -> str:
        return "My Model"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_model_lib
            return True
        except ImportError:
            return False

    def process_single(self, pdf_path: str) -> OCRResult:
        paper_name = Path(pdf_path).stem
        output_dir = self.get_paper_output_dir(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ... your conversion logic ...

        return self._make_result(
            pdf_path,
            status="success",
            result_path=str(output_dir),
        )
```

## 2. Register the Import

Add one line to `src/ocr/__init__.py`:

```python
import src.ocr.my_model_ocr  # noqa: F401
```

## 3. Use It

```bash
# Check it shows up
uv run python -m src.cli.main list-ocr-models

# Run OCR with your model
uv run python -m src.cli.main ocr \
  --input-dir data/pdfs \
  --output-dir data/ocr_results \
  --ocr-model my-model
```

## Optional: Batch Processing

Override `supports_batch()` and `process_batch()` if your model can process multiple files more efficiently than one-at-a-time:

```python
def supports_batch(self) -> bool:
    return True

def process_batch(self, pdf_paths, gpu_count=1, workers_per_gpu=1):
    # Your batch logic here
    return [self.process_single(p) for p in pdf_paths]
```

## Output Format

Each model must produce this directory structure:

```
{output_dir}/{paper_name}/{paper_name}.md    # Markdown output
{output_dir}/{paper_name}/*.png              # Extracted images (optional)
```

The status tracker automatically writes `{output_dir}/ocr_status.csv`.

## Abstract Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `name` | `@property -> str` | Human-readable model name |
| `process_single` | `(pdf_path: str) -> OCRResult` | Convert one PDF |
| `is_available` | `@classmethod -> bool` | Check dependencies |
