from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_image_model
from unified_xai.modalities.image.explainers import (
    explain_image_with_gradcam,
    explain_image_with_lime,
    explain_image_with_shap,
)


@dataclass(frozen=True)
class ImageExplainerSpec:
    explainer_id: str
    display_name: str
    description: str


@dataclass(frozen=True)
class ImageExplanationResult:
    explainer_id: str
    display_name: str
    summary: str
    visualization: np.ndarray
    details: dict[str, str | int | float]


IMPLEMENTED_IMAGE_EXPLAINERS: dict[str, ImageExplainerSpec] = {
    "gradcam": ImageExplainerSpec(
        explainer_id="gradcam",
        display_name="Grad-CAM",
        description="Gradient-based localization heatmap over the chest X-ray.",
    ),
    "lime": ImageExplainerSpec(
        explainer_id="lime",
        display_name="LIME",
        description="Perturbation-based explanation over the X-ray image.",
    ),
    "shap": ImageExplainerSpec(
        explainer_id="shap",
        display_name="SHAP",
        description="Shapley-style attribution over the chest X-ray image.",
    ),
}


def get_image_explainer(explainer_id: str) -> ImageExplainerSpec:
    try:
        return IMPLEMENTED_IMAGE_EXPLAINERS[explainer_id]
    except KeyError as exc:
        raise KeyError(f"Unknown image explainer '{explainer_id}'.") from exc


def list_available_image_explainers(model_id: str) -> list[ImageExplainerSpec]:
    spec = get_image_model(model_id)
    if not spec.is_available:
        return []

    return [
        explainer
        for explainer_id, explainer in IMPLEMENTED_IMAGE_EXPLAINERS.items()
        if explainer_id in spec.supported_explainers
    ]


def run_image_explanation(
    model_id: str,
    explainer_id: str,
    image: np.ndarray,
    predicted_label: str | None = None,
) -> ImageExplanationResult:
    model_spec = get_image_model(model_id)
    explainer_spec = get_image_explainer(explainer_id)

    if explainer_id not in model_spec.supported_explainers:
        raise ValueError(
            f"Explainer '{explainer_id}' is not supported by model '{model_id}'."
        )

    if explainer_id == "gradcam":
        explanation = explain_image_with_gradcam(
            model_id=model_id,
            image_array=image,
            target_label=predicted_label,
        )
        return ImageExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "Grad-CAM backpropagates gradients to the selected chest X-ray feature map and highlights "
                f"the regions that most influenced the predicted label `{explanation.target_label}`."
            ),
            visualization=explanation.overlay,
            details={"target_label": explanation.target_label},
        )

    if explainer_id == "lime":
        explanation = explain_image_with_lime(
            model_id=model_id,
            image_array=image,
            labels=model_spec.labels,
            target_label=predicted_label,
        )
        return ImageExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "LIME perturbs regions of the chest X-ray and estimates which segments most "
                f"influenced the predicted label `{explanation.target_label}`."
            ),
            visualization=explanation.visualization,
            details={
                "target_label": explanation.target_label,
                "num_samples": explanation.num_samples,
                "num_features": explanation.num_features,
            },
        )

    if explainer_id == "shap":
        explanation = explain_image_with_shap(
            model_id=model_id,
            image_array=image,
            labels=model_spec.labels,
            target_label=predicted_label,
        )
        return ImageExplanationResult(
            explainer_id=explainer_spec.explainer_id,
            display_name=explainer_spec.display_name,
            summary=(
                "SHAP estimates the contribution of chest X-ray regions to the predicted label "
                f"`{explanation.target_label}`."
            ),
            visualization=explanation.visualization,
            details={
                "target_label": explanation.target_label,
                "max_evals": explanation.max_evals,
            },
        )

    raise ValueError(f"Explainer '{explainer_id}' is not implemented yet.")
