"""Image-to-PDF conversion utility for OCR scripts.

OCR models like Marker and MarkItDown expect PDF input, but benchmark
datasets provide images. This module converts images to multi-page PDFs.
"""

from pathlib import Path
from typing import List


def images_to_pdf(image_paths: List[str], output_path: str) -> str:
    """Convert one or more images to a multi-page PDF.

    Args:
        image_paths: List of image file paths.
        output_path: Path for the output PDF file.

    Returns:
        The output_path string.

    Raises:
        ValueError: If image_paths is empty.
        FileNotFoundError: If an image file does not exist.
    """
    if not image_paths:
        raise ValueError("At least one image path is required")

    from PIL import Image

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    images = []
    for path in image_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        images.append(img)

    images[0].save(
        output_path,
        "PDF",
        save_all=True,
        append_images=images[1:],
    )

    return output_path
