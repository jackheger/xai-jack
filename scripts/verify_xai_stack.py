from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from unified_xai.services.audio_service import run_audio_inference
from unified_xai.services.explanation_service import run_audio_explanation
from unified_xai.services.image_explanation_service import run_image_explanation
from unified_xai.services.image_service import run_image_inference


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "verification"
RECOMMENDED_AUDIO_MODEL_ID = "deepfake_melspec_cnn"
RECOMMENDED_IMAGE_MODEL_ID = "jsrt_densenet121"
RECOMMENDED_AUDIO_SAMPLE = (
    PROJECT_ROOT / "sample_inputs" / "audio" / "demo_real_packaged_correct.wav"
)
RECOMMENDED_IMAGE_SAMPLE = (
    PROJECT_ROOT / "sample_inputs" / "images" / "jsrt_malignant_jpcln047.png"
)


@dataclass(frozen=True)
class VerificationResult:
    modality: str
    model_id: str
    explainer_id: str
    status: str
    duration_seconds: float
    output_path: str | None
    details: dict[str, str | int | float]


def save_image(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.asarray(array).astype(np.uint8))
    image.save(path)


def verify_audio_stack() -> list[VerificationResult]:
    audio_bytes = RECOMMENDED_AUDIO_SAMPLE.read_bytes()
    inference = run_audio_inference(
        RECOMMENDED_AUDIO_MODEL_ID,
        audio_bytes,
        source_name=RECOMMENDED_AUDIO_SAMPLE.name,
    )
    save_image(inference.spectrogram, OUTPUT_DIR / "audio_reference_spectrogram.png")

    results: list[VerificationResult] = []
    for explainer_id in ("gradcam", "lime", "shap"):
        output_path = OUTPUT_DIR / f"audio_{explainer_id}.png"
        started = time.perf_counter()
        explanation = run_audio_explanation(
            model_id=RECOMMENDED_AUDIO_MODEL_ID,
            explainer_id=explainer_id,
            spectrogram=inference.spectrogram,
            predicted_label=inference.prediction.label,
        )
        duration_seconds = time.perf_counter() - started
        save_image(explanation.visualization, output_path)
        results.append(
            VerificationResult(
                modality="audio",
                model_id=RECOMMENDED_AUDIO_MODEL_ID,
                explainer_id=explainer_id,
                status="ok",
                duration_seconds=duration_seconds,
                output_path=str(output_path.relative_to(PROJECT_ROOT)),
                details=explanation.details,
            )
        )
    return results


def verify_image_stack() -> list[VerificationResult]:
    image_bytes = RECOMMENDED_IMAGE_SAMPLE.read_bytes()
    inference = run_image_inference(
        RECOMMENDED_IMAGE_MODEL_ID,
        image_bytes,
        source_name=RECOMMENDED_IMAGE_SAMPLE.name,
    )
    save_image(inference.image.astype(np.uint8), OUTPUT_DIR / "image_reference_xray.png")

    results: list[VerificationResult] = []
    for explainer_id in ("gradcam", "lime", "shap"):
        output_path = OUTPUT_DIR / f"image_{explainer_id}.png"
        started = time.perf_counter()
        explanation = run_image_explanation(
            model_id=RECOMMENDED_IMAGE_MODEL_ID,
            explainer_id=explainer_id,
            image=inference.image,
            predicted_label=inference.prediction.label,
        )
        duration_seconds = time.perf_counter() - started
        save_image(explanation.visualization, output_path)
        results.append(
            VerificationResult(
                modality="image",
                model_id=RECOMMENDED_IMAGE_MODEL_ID,
                explainer_id=explainer_id,
                status="ok",
                duration_seconds=duration_seconds,
                output_path=str(output_path.relative_to(PROJECT_ROOT)),
                details=explanation.details,
            )
        )
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommended_audio_model_id": RECOMMENDED_AUDIO_MODEL_ID,
        "recommended_audio_sample": str(RECOMMENDED_AUDIO_SAMPLE.relative_to(PROJECT_ROOT)),
        "recommended_image_model_id": RECOMMENDED_IMAGE_MODEL_ID,
        "recommended_image_sample": str(RECOMMENDED_IMAGE_SAMPLE.relative_to(PROJECT_ROOT)),
        "results": [
            result.__dict__ for result in (verify_audio_stack() + verify_image_stack())
        ],
    }
    (OUTPUT_DIR / "xai_stack_verification.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Saved XAI verification outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
