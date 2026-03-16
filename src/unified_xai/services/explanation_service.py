from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_audio_model
from unified_xai.modalities.audio.explainers import (
    explain_with_gradcam,
    explain_with_lime,
    explain_with_shap,
)


@dataclass(frozen=True)
class AudioExplainerSpec:
    explainer_id: str
    display_name: str
    description: str


@dataclass(frozen=True)
class AudioExplanationResult:
    explainer_id: str
    display_name: str
    summary: str
    visualization: np.ndarray
    details: dict[str, str | int | float]


IMPLEMENTED_AUDIO_EXPLAINERS: dict[str, AudioExplainerSpec] = {
    "gradcam": AudioExplainerSpec(
        explainer_id="gradcam",
        display_name="Grad-CAM",
        description=(
            "Gradient-based localization over the generated mel spectrogram. "
            "Useful when the selected audio model exposes convolutional feature maps."
        ),
    ),
    "lime": AudioExplainerSpec(
        explainer_id="lime",
        display_name="LIME",
        description=(
            "Perturbation-based explanation over the spectrogram image. "
            "Useful as a model-agnostic first explainer for all installed audio checkpoints."
        ),
    ),
    "shap": AudioExplainerSpec(
        explainer_id="shap",
        display_name="SHAP",
        description=(
            "Shapley-style image attribution over the spectrogram. "
            "More expensive than LIME, but useful for feature-contribution views."
        ),
    ),
}


def get_audio_explainer(explainer_id: str) -> AudioExplainerSpec:
    try:
        return IMPLEMENTED_AUDIO_EXPLAINERS[explainer_id]
    except KeyError as exc:
        raise KeyError(f"Unknown audio explainer '{explainer_id}'.") from exc


def list_available_audio_explainers(model_id: str) -> list[AudioExplainerSpec]:
    spec = get_audio_model(model_id)
    if not spec.is_available:
        return []

    return [
        explainer
        for explainer_id, explainer in IMPLEMENTED_AUDIO_EXPLAINERS.items()
        if explainer_id in spec.supported_explainers
    ]


def run_audio_explanation(
    model_id: str,
    explainer_id: str,
    spectrogram: np.ndarray,
    predicted_label: str | None = None,
) -> AudioExplanationResult:
    model_spec = get_audio_model(model_id)
    explainer_spec = get_audio_explainer(explainer_id)

    if explainer_id not in model_spec.supported_explainers:
        raise ValueError(
            f"Explainer '{explainer_id}' is not supported by model '{model_id}'."
        )

    if explainer_id == "lime":
        explanation = explain_with_lime(
            model_id=model_id,
            spectrogram=spectrogram,
            labels=model_spec.labels,
            target_label=predicted_label,
        )
        return AudioExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "LIME perturbs regions of the mel spectrogram and estimates which segments "
                f"most influenced the predicted label `{explanation.target_label}`."
            ),
            visualization=explanation.visualization,
            details={
                "target_label": explanation.target_label,
                "num_samples": explanation.num_samples,
                "num_features": explanation.num_features,
            },
        )

    if explainer_id == "gradcam":
        explanation = explain_with_gradcam(
            model_id=model_id,
            spectrogram=spectrogram,
            target_label=predicted_label,
        )
        return AudioExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "Grad-CAM backpropagates gradients to the selected spectrogram feature map and "
                f"highlights the regions that most influenced the predicted label `{explanation.target_label}`."
            ),
            visualization=explanation.visualization,
            details={
                "target_label": explanation.target_label,
                "target_layer": explanation.target_layer,
            },
        )

    if explainer_id == "shap":
        explanation = explain_with_shap(
            model_id=model_id,
            spectrogram=spectrogram,
            labels=model_spec.labels,
            target_label=predicted_label,
        )
        return AudioExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "SHAP estimates the contribution of spectrogram regions to the predicted label "
                f"`{explanation.target_label}`."
            ),
            visualization=explanation.visualization,
            details={
                "target_label": explanation.target_label,
                "max_evals": explanation.max_evals,
            },
        )

    raise ValueError(f"Explainer '{explainer_id}' is not implemented yet.")
