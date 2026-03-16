from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from unified_xai.core.registry import get_audio_model


@dataclass(frozen=True)
class PredictionResult:
    label: str
    probabilities: dict[str, float]
    raw_output: tuple[float, ...]


@lru_cache(maxsize=4)
def load_audio_model(model_id: str):
    spec = get_audio_model(model_id)
    if not spec.artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{spec.artifact_path}'."
        )

    import tensorflow as tf

    return tf.keras.models.load_model(spec.artifact_path)


def _prepare_spectrogram_batch(spectrogram_batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(spectrogram_batch, dtype=np.float32)

    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    if batch.ndim != 4:
        raise ValueError(
            "Expected a spectrogram batch with shape (batch, height, width, channels)."
        )

    if float(np.max(batch)) > 1.0:
        batch = batch / 255.0

    return batch


def _ensure_2d_predictions(raw_predictions: np.ndarray, batch_size: int) -> np.ndarray:
    predictions = np.asarray(raw_predictions, dtype=np.float32)

    if predictions.ndim == 1:
        if predictions.shape[0] == batch_size:
            predictions = predictions.reshape(batch_size, 1)
        else:
            predictions = predictions.reshape(1, -1)

    if predictions.ndim != 2:
        raise ValueError("Model prediction returned an unexpected output shape.")

    return predictions


def _normalize_probabilities(raw_output: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    flattened = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    if flattened.size == 1 and len(labels) == 2:
        positive = float(np.clip(flattened[0], 0.0, 1.0))
        return np.array([1.0 - positive, positive], dtype=np.float32)

    if flattened.size != len(labels):
        raise ValueError(
            f"Expected {len(labels)} outputs but received {flattened.size}."
        )

    total = float(flattened.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Model prediction contains invalid values.")

    return flattened / total


def _predict_raw_and_probabilities(
    model_id: str,
    spectrogram_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spec = get_audio_model(model_id)
    model = load_audio_model(model_id)
    batch = _prepare_spectrogram_batch(spectrogram_batch)
    raw_predictions = np.asarray(model.predict(batch, verbose=0))
    raw_predictions = _ensure_2d_predictions(raw_predictions, batch_size=batch.shape[0])
    probabilities = np.stack(
        [_normalize_probabilities(row, spec.labels) for row in raw_predictions],
        axis=0,
    )
    return raw_predictions, probabilities


def predict_spectrogram_probabilities(
    model_id: str,
    spectrogram_batch: np.ndarray,
) -> np.ndarray:
    _, probabilities = _predict_raw_and_probabilities(model_id, spectrogram_batch)
    return probabilities


def predict_spectrogram(model_id: str, spectrogram: np.ndarray) -> PredictionResult:
    spec = get_audio_model(model_id)
    raw_prediction, probabilities_batch = _predict_raw_and_probabilities(model_id, spectrogram)
    probabilities = probabilities_batch[0]

    predicted_index = int(np.argmax(probabilities))
    return PredictionResult(
        label=spec.labels[predicted_index],
        probabilities={
            label: float(probabilities[idx]) for idx, label in enumerate(spec.labels)
        },
        raw_output=tuple(float(value) for value in np.asarray(raw_prediction[0]).reshape(-1)),
    )
