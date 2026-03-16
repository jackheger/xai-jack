from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_audio_model
from unified_xai.modalities.audio.model import (
    load_audio_model,
    predict_spectrogram_probabilities,
)


@dataclass(frozen=True)
class LIMEExplanation:
    visualization: np.ndarray
    target_label: str
    target_index: int
    num_samples: int
    num_features: int


@dataclass(frozen=True)
class SHAPExplanation:
    visualization: np.ndarray
    target_label: str
    max_evals: int


@dataclass(frozen=True)
class GradCAMExplanation:
    visualization: np.ndarray
    heatmap: np.ndarray
    target_label: str
    target_layer: str


def _build_overlay_from_attributions(image: np.ndarray, attributions: np.ndarray) -> np.ndarray:
    import matplotlib

    importance = np.mean(np.abs(attributions), axis=-1)
    max_value = float(np.max(importance))
    if max_value > 0:
        importance = importance / max_value

    colormap = matplotlib.colormaps["magma"]
    colored = colormap(np.clip(importance, 0.0, 1.0))[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    overlay = np.clip((0.6 * base) + (0.4 * colored), 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def _build_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    import matplotlib

    colormap = matplotlib.colormaps["jet"]
    colored = colormap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    base = np.asarray(image, dtype=np.float32) / 255.0
    overlay = np.clip((0.6 * base) + (0.4 * colored), 0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def _infer_default_gradcam_layer(model) -> str:
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None:
            continue
        if isinstance(output_shape, list):
            continue
        if len(output_shape) == 4:
            return layer.name
    raise ValueError("Unable to infer a 4D feature map layer for Grad-CAM.")


def _resolve_layer_output_tensor(layer):
    try:
        return layer.get_output_at(0)
    except (AttributeError, RuntimeError, ValueError):
        return layer.output


def explain_with_gradcam(
    model_id: str,
    spectrogram: np.ndarray,
    target_label: str | None = None,
) -> GradCAMExplanation:
    import tensorflow as tf

    spec = get_audio_model(model_id)
    model = load_audio_model(model_id)
    target_layer_name = spec.gradcam_target_layer or _infer_default_gradcam_layer(model)

    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Grad-CAM target layer '{target_layer_name}' was not found in model '{model_id}'."
        ) from exc

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[_resolve_layer_output_tensor(target_layer), model.output],
    )

    image_batch = np.expand_dims(np.asarray(spectrogram, dtype=np.float32), axis=0)
    if float(np.max(image_batch)) > 1.0:
        image_batch = image_batch / 255.0

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if predictions.shape[-1] == 1:
            positive_score = predictions[:, 0]
            if target_label == spec.labels[0]:
                class_channel = 1.0 - positive_score
                target_index = 0
            else:
                class_channel = positive_score
                target_index = 1 if len(spec.labels) > 1 else 0
        else:
            if target_label and target_label in spec.labels:
                target_index = spec.labels.index(target_label)
            else:
                target_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, target_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    if gradients is None:
        raise ValueError("Grad-CAM gradients were not available for the selected audio model.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        spectrogram.shape[:2],
    ).numpy().squeeze()
    overlay = _build_gradcam_overlay(spectrogram, heatmap)

    return GradCAMExplanation(
        visualization=overlay,
        heatmap=(np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8),
        target_label=spec.labels[target_index],
        target_layer=target_layer_name,
    )


def explain_with_shap(
    model_id: str,
    spectrogram: np.ndarray,
    labels: tuple[str, ...],
    target_label: str | None = None,
    max_evals: int = 64,
) -> SHAPExplanation:
    import shap

    image = np.asarray(spectrogram, dtype=np.float32)
    normalized = image / 255.0

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        return predict_spectrogram_probabilities(model_id, np.asarray(images))

    prediction = classifier_fn(normalized[np.newaxis, ...])[0]
    if target_label and target_label in labels:
        target_index = labels.index(target_label)
    else:
        target_index = int(np.argmax(prediction))

    masker = shap.maskers.Image("blur(32,32)", normalized.shape)
    explainer = shap.Explainer(classifier_fn, masker, output_names=list(labels))
    shap_values = explainer(normalized[np.newaxis, ...], max_evals=max_evals, batch_size=16)

    values = np.asarray(shap_values.values)
    if values.ndim == 5:
        attributions = values[0, ..., target_index]
    elif values.ndim == 4:
        attributions = values[0]
    else:
        raise ValueError("SHAP returned an unexpected attribution shape for audio.")

    visualization = _build_overlay_from_attributions(image, attributions)
    return SHAPExplanation(
        visualization=visualization,
        target_label=labels[target_index],
        max_evals=max_evals,
    )


def explain_with_lime(
    model_id: str,
    spectrogram: np.ndarray,
    labels: tuple[str, ...],
    target_label: str | None = None,
    num_samples: int = 128,
    num_features: int = 8,
) -> LIMEExplanation:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    image = np.asarray(spectrogram, dtype=np.float32) / 255.0

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        return predict_spectrogram_probabilities(model_id, np.asarray(images))

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image=image.astype("double"),
        classifier_fn=classifier_fn,
        top_labels=len(labels),
        hide_color=0,
        batch_size=16,
        num_samples=num_samples,
    )

    if target_label and target_label in labels:
        target_index = labels.index(target_label)
    else:
        target_index = int(explanation.top_labels[0])

    temp, mask = explanation.get_image_and_mask(
        label=target_index,
        positive_only=False,
        num_features=num_features,
        hide_rest=False,
        min_weight=0.0,
    )
    overlay = mark_boundaries(
        np.clip(temp, 0.0, 1.0),
        mask,
        color=(1.0, 0.55, 0.0),
        mode="thick",
    )
    visualization = (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)

    return LIMEExplanation(
        visualization=visualization,
        target_label=labels[target_index],
        target_index=target_index,
        num_samples=num_samples,
        num_features=num_features,
    )
