from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unified_xai.core.registry import get_image_model
from unified_xai.modalities.image.explainers import explain_image_with_gradcam
from unified_xai.services.image_service import run_image_inference


SAMPLE_IMAGE_PATH = Path(__file__).resolve().parents[1] / "sample_inputs" / "images" / "jsrt_malignant_jpcln047.png"
IMAGE_MODEL_IDS = ("jsrt_densenet121", "jsrt_alexnet")


@pytest.mark.parametrize("model_id", IMAGE_MODEL_IDS)
def test_image_inference_smoke(model_id: str) -> None:
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip(f"Missing sample image file: {SAMPLE_IMAGE_PATH}")

    spec = get_image_model(model_id)
    if not spec.is_available:
        pytest.skip(f"Image checkpoint '{model_id}' is not available.")

    result = run_image_inference(
        model_id=model_id,
        image_bytes=SAMPLE_IMAGE_PATH.read_bytes(),
        source_name=SAMPLE_IMAGE_PATH.name,
    )

    assert result.image.shape == (*spec.input_size, 3)
    assert result.prediction.label in spec.labels
    assert np.isclose(sum(result.prediction.probabilities.values()), 1.0, atol=1e-5)
    assert result.image_metadata["original_width"] > 0
    assert result.image_metadata["original_height"] > 0


@pytest.mark.parametrize("model_id", IMAGE_MODEL_IDS)
def test_image_gradcam_smoke(model_id: str) -> None:
    if not SAMPLE_IMAGE_PATH.exists():
        pytest.skip(f"Missing sample image file: {SAMPLE_IMAGE_PATH}")

    spec = get_image_model(model_id)
    if not spec.is_available:
        pytest.skip(f"Image checkpoint '{model_id}' is not available.")

    result = run_image_inference(
        model_id=model_id,
        image_bytes=SAMPLE_IMAGE_PATH.read_bytes(),
        source_name=SAMPLE_IMAGE_PATH.name,
    )
    explanation = explain_image_with_gradcam(
        model_id=model_id,
        image_array=result.image,
        target_label=result.prediction.label,
    )

    assert explanation.target_label in spec.labels
    assert explanation.overlay.shape == result.image.shape
    assert explanation.heatmap.shape == result.image.shape[:2]
