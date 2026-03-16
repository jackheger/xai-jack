from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_image_model
from unified_xai.modalities.image.model import load_image_model, predict_image_probabilities


@dataclass(frozen=True)
class ImageLIMEExplanation:
    visualization: np.ndarray
    target_label: str
    num_samples: int
    num_features: int


@dataclass(frozen=True)
class ImageGradCAMExplanation:
    overlay: np.ndarray
    heatmap: np.ndarray
    target_label: str


@dataclass(frozen=True)
class ImageSHAPExplanation:
    visualization: np.ndarray
    target_label: str
    max_evals: int


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


def explain_image_with_lime(
    model_id: str,
    image_array: np.ndarray,
    labels: tuple[str, ...],
    target_label: str | None = None,
    num_samples: int = 128,
    num_features: int = 8,
) -> ImageLIMEExplanation:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    image = np.asarray(image_array, dtype=np.float32) / 255.0

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        return predict_image_probabilities(model_id, np.asarray(images))

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

    return ImageLIMEExplanation(
        visualization=visualization,
        target_label=labels[target_index],
        num_samples=num_samples,
        num_features=num_features,
    )


def explain_image_with_gradcam(
    model_id: str,
    image_array: np.ndarray,
    target_label: str | None = None,
) -> ImageGradCAMExplanation:
    import matplotlib
    import tensorflow as tf

    spec = get_image_model(model_id)
    model = load_image_model(model_id)

    if not spec.gradcam_target_layer:
        raise ValueError(f"Model '{model_id}' does not define a Grad-CAM target layer.")

    backbone = model.get_layer(spec.gradcam_target_layer)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[backbone.output, model.output],
    )

    image_batch = np.expand_dims(np.asarray(image_array, dtype=np.float32), axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if target_label and target_label in spec.labels:
            class_index = spec.labels.index(target_label)
        else:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = tf.image.resize(heatmap[..., tf.newaxis], image_array.shape[:2]).numpy().squeeze()
    colormap = matplotlib.colormaps["jet"]
    colored = colormap(np.clip(heatmap, 0.0, 1.0))[..., :3]

    base = np.asarray(image_array, dtype=np.float32) / 255.0
    overlay = np.clip((0.6 * base) + (0.4 * colored), 0.0, 1.0)

    return ImageGradCAMExplanation(
        overlay=(overlay * 255).astype(np.uint8),
        heatmap=(np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8),
        target_label=spec.labels[class_index],
    )


def explain_image_with_shap(
    model_id: str,
    image_array: np.ndarray,
    labels: tuple[str, ...],
    target_label: str | None = None,
    max_evals: int = 64,
) -> ImageSHAPExplanation:
    import shap

    image = np.asarray(image_array, dtype=np.float32)
    normalized = image / 255.0

    def classifier_fn(images: np.ndarray) -> np.ndarray:
        return predict_image_probabilities(model_id, np.asarray(images))

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
        raise ValueError("SHAP returned an unexpected attribution shape for images.")

    visualization = _build_overlay_from_attributions(image, attributions)
    return ImageSHAPExplanation(
        visualization=visualization,
        target_label=labels[target_index],
        max_evals=max_evals,
    )
