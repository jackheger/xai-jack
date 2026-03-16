from __future__ import annotations

from pathlib import Path

import pytest

from unified_xai.core.registry import get_audio_model, get_image_model
from unified_xai.services.audio_service import run_audio_inference
from unified_xai.services.image_service import run_image_inference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SAMPLE_DIR = PROJECT_ROOT / "sample_inputs" / "audio"
IMAGE_SAMPLE_DIR = PROJECT_ROOT / "sample_inputs" / "images"


def test_recommended_audio_demo_samples_match_packaged_model() -> None:
    spec = get_audio_model("deepfake_melspec_cnn")
    if not spec.is_available:
        pytest.skip("Packaged audio checkpoint is not available.")

    expected_samples = {
        "demo_real_packaged_correct.wav": "real",
        "demo_fake_packaged_correct.wav": "fake",
    }
    for sample_name, expected_label in expected_samples.items():
        sample_path = AUDIO_SAMPLE_DIR / sample_name
        if not sample_path.exists():
            pytest.skip(f"Missing audio demo sample: {sample_path}")

        result = run_audio_inference(
            model_id="deepfake_melspec_cnn",
            audio_bytes=sample_path.read_bytes(),
            source_name=sample_path.name,
        )
        assert result.prediction.label == expected_label


def test_recommended_image_demo_samples_match_densenet() -> None:
    spec = get_image_model("jsrt_densenet121")
    if not spec.is_available:
        pytest.skip("DenseNet image checkpoint is not available.")

    expected_samples = {
        "jsrt_malignant_jpcln047.png": "malignant",
        "jsrt_benign_jpcln036.png": "non_malignant",
        "jsrt_non_nodule_jpcnn086.png": "non_malignant",
    }
    for sample_name, expected_label in expected_samples.items():
        sample_path = IMAGE_SAMPLE_DIR / sample_name
        if not sample_path.exists():
            pytest.skip(f"Missing image demo sample: {sample_path}")

        result = run_image_inference(
            model_id="jsrt_densenet121",
            image_bytes=sample_path.read_bytes(),
            source_name=sample_path.name,
        )
        assert result.prediction.label == expected_label
