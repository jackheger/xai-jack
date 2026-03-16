from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def infer_image_format(source_name: str | None) -> str:
    suffix = Path(source_name or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return suffix.removeprefix(".")
    return "unknown"


def image_bytes_to_array(
    image_bytes: bytes,
    target_size: tuple[int, int],
    source_name: str | None = None,
) -> tuple[np.ndarray, dict[str, str | int]]:
    with Image.open(io.BytesIO(image_bytes)) as handle:
        original_mode = handle.mode
        original_size = handle.size
        grayscale = handle.convert("L")
        resized = grayscale.resize(target_size)
        image = resized.convert("RGB")

    array = np.asarray(image, dtype=np.float32)
    metadata = {
        "source_format": infer_image_format(source_name),
        "original_mode": original_mode,
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "processed_width": int(target_size[0]),
        "processed_height": int(target_size[1]),
    }
    return array, metadata
