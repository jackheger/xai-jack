from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_image_model
from unified_xai.modalities.image.model import ImagePredictionResult, predict_image
from unified_xai.modalities.image.preprocess import image_bytes_to_array


@dataclass(frozen=True)
class ImageInferenceResult:
    model_id: str
    image: np.ndarray
    image_metadata: dict[str, str | int]
    prediction: ImagePredictionResult


def run_image_inference(
    model_id: str,
    image_bytes: bytes,
    source_name: str | None = None,
) -> ImageInferenceResult:
    spec = get_image_model(model_id)
    image, image_metadata = image_bytes_to_array(
        image_bytes=image_bytes,
        target_size=spec.input_size,
        source_name=source_name,
    )
    prediction = predict_image(model_id=model_id, image_array=image)
    return ImageInferenceResult(
        model_id=model_id,
        image=image,
        image_metadata=image_metadata,
        prediction=prediction,
    )
