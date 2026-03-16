from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unified_xai.core.registry import get_audio_model
from unified_xai.modalities.audio.model import PredictionResult, predict_spectrogram
from unified_xai.modalities.audio.preprocess import waveform_to_mel_spectrogram


@dataclass(frozen=True)
class AudioInferenceResult:
    model_id: str
    spectrogram: np.ndarray
    audio_metadata: dict[str, float | str | bool | bytes]
    prediction: PredictionResult


def run_audio_inference(
    model_id: str,
    audio_bytes: bytes,
    source_name: str | None = None,
) -> AudioInferenceResult:
    spec = get_audio_model(model_id)
    spectrogram, audio_metadata = waveform_to_mel_spectrogram(
        audio_bytes=audio_bytes,
        target_size=spec.input_size,
        source_name=source_name,
    )
    prediction = predict_spectrogram(model_id=model_id, spectrogram=spectrogram)
    return AudioInferenceResult(
        model_id=model_id,
        spectrogram=spectrogram,
        audio_metadata=audio_metadata,
        prediction=prediction,
    )
