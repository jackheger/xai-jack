from __future__ import annotations

import io
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from PIL import Image


def infer_audio_format(source_name: str | None) -> str:
    suffix = Path(source_name or "").suffix.lower()
    if suffix in {".wav", ".mp3"}:
        return suffix.removeprefix(".")
    return "unknown"


def _load_waveform(
    audio_bytes: bytes,
    source_name: str | None,
) -> tuple[np.ndarray, int, str]:
    source_format = infer_audio_format(source_name)
    temp_suffix = Path(source_name or "uploaded.wav").suffix.lower() or ".wav"

    with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)

    try:
        waveform, sample_rate = librosa.load(temp_path, sr=None, mono=True)
    finally:
        temp_path.unlink(missing_ok=True)

    return waveform, sample_rate, source_format


def convert_waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return buffer.getvalue()


def waveform_to_mel_spectrogram(
    audio_bytes: bytes,
    target_size: tuple[int, int],
    source_name: str | None = None,
) -> tuple[np.ndarray, dict[str, float | str | bool | bytes]]:
    waveform, sample_rate, source_format = _load_waveform(audio_bytes, source_name)
    converted_to_wav = source_format == "mp3"

    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    librosa.display.specshow(log_mel, sr=sample_rate, ax=ax)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB").resize(target_size)
    spectrogram = np.array(image)

    duration_seconds = float(len(waveform) / sample_rate) if sample_rate else 0.0
    metadata = {
        "sample_rate": float(sample_rate),
        "duration_seconds": duration_seconds,
        "source_format": source_format,
        "converted_to_wav": converted_to_wav,
        "conversion_warning": (
            "MP3 input was converted to WAV/PCM before inference. Lossy compression "
            "and decoding artifacts can reduce classification reliability."
            if converted_to_wav
            else ""
        ),
        "converted_wav_bytes": convert_waveform_to_wav_bytes(waveform, sample_rate)
        if converted_to_wav
        else audio_bytes,
    }
    return spectrogram, metadata
