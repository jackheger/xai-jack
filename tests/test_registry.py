from __future__ import annotations

from unified_xai.core.registry import (
    get_image_model,
    list_available_audio_models,
    list_available_image_models,
)


def test_available_model_registry_has_installed_entries() -> None:
    available_audio_models = list_available_audio_models()
    available_image_models = list_available_image_models()

    assert available_audio_models, "Expected at least one installed audio checkpoint."
    assert available_image_models, "Expected at least one installed image checkpoint."
    assert len(available_image_models) >= 2, "Expected the JSRT DenseNet and AlexNet checkpoints."


def test_jsrt_image_model_is_available() -> None:
    spec = get_image_model("jsrt_densenet121")

    assert spec.is_available
    assert spec.gradcam_target_layer == "jsrt_gradcam_target"
    assert "gradcam" in spec.supported_explainers


def test_jsrt_alexnet_model_is_available() -> None:
    spec = get_image_model("jsrt_alexnet")

    assert spec.is_available
    assert spec.gradcam_target_layer == "jsrt_alexnet_gradcam_target"
    assert spec.input_size == (227, 227)
    assert "gradcam" in spec.supported_explainers
