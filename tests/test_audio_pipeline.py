from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from unified_xai.core.registry import get_audio_model
from unified_xai.modalities.audio.explainers import explain_with_gradcam, explain_with_lime
from unified_xai.services.audio_service import run_audio_inference


SAMPLE_AUDIO_PATH = Path(__file__).resolve().parents[1] / "sample_inputs" / "audio" / "file_example_WAV_1MG.wav"


def _select_audio_model_id() -> str:
    for model_id in ("vgg16", "deepfake_melspec_cnn", "custom_cnn"):
        if get_audio_model(model_id).is_available:
            return model_id
    pytest.skip("No installed audio checkpoint is available for smoke testing.")


def test_audio_inference_smoke() -> None:
    if not SAMPLE_AUDIO_PATH.exists():
        pytest.skip(f"Missing sample audio file: {SAMPLE_AUDIO_PATH}")

    model_id = _select_audio_model_id()
    result = run_audio_inference(
        model_id=model_id,
        audio_bytes=SAMPLE_AUDIO_PATH.read_bytes(),
        source_name=SAMPLE_AUDIO_PATH.name,
    )

    assert result.spectrogram.shape == (224, 224, 3)
    assert result.prediction.label in get_audio_model(model_id).labels
    assert np.isclose(sum(result.prediction.probabilities.values()), 1.0, atol=1e-5)
    assert result.audio_metadata["sample_rate"]
    assert result.audio_metadata["duration_seconds"] > 0


def test_audio_lime_smoke() -> None:
    if not SAMPLE_AUDIO_PATH.exists():
        pytest.skip(f"Missing sample audio file: {SAMPLE_AUDIO_PATH}")

    model_id = _select_audio_model_id()
    spec = get_audio_model(model_id)
    result = run_audio_inference(
        model_id=model_id,
        audio_bytes=SAMPLE_AUDIO_PATH.read_bytes(),
        source_name=SAMPLE_AUDIO_PATH.name,
    )

    explanation = explain_with_lime(
        model_id=model_id,
        spectrogram=result.spectrogram,
        labels=spec.labels,
        target_label=result.prediction.label,
        num_samples=16,
        num_features=4,
    )

    assert explanation.target_label in spec.labels
    assert explanation.visualization.shape == result.spectrogram.shape
    assert explanation.num_samples == 16
    assert explanation.num_features == 4


def test_audio_gradcam_smoke() -> None:
    if not SAMPLE_AUDIO_PATH.exists():
        pytest.skip(f"Missing sample audio file: {SAMPLE_AUDIO_PATH}")

    model_id = _select_audio_model_id()
    spec = get_audio_model(model_id)
    result = run_audio_inference(
        model_id=model_id,
        audio_bytes=SAMPLE_AUDIO_PATH.read_bytes(),
        source_name=SAMPLE_AUDIO_PATH.name,
    )

    explanation = explain_with_gradcam(
        model_id=model_id,
        spectrogram=result.spectrogram,
        target_label=result.prediction.label,
    )

    assert explanation.target_label in spec.labels
    assert explanation.visualization.shape == result.spectrogram.shape
    assert explanation.heatmap.shape == result.spectrogram.shape[:2]
    assert explanation.target_layer
