from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


DEFAULT_DATA_DIR = Path(
    r"C:\Users\jackj\Work\A6\Explanability\train_datasets\for-2sec\for-2seconds\testing"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\jackj\Work\A6\Explanability\artifacts\evaluation"
)


@dataclass(frozen=True)
class EvaluationConfig:
    data_dir: Path
    output_dir: Path
    model_ids: tuple[str, ...]
    limit_per_class: int | None


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate installed audio models on the local FoR testing subset."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepfake_melspec_cnn", "vgg16", "custom_cnn"],
        help="Audio model ids to evaluate.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional limit per class for quicker exploratory runs.",
    )
    args = parser.parse_args()
    return EvaluationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_ids=tuple(args.models),
        limit_per_class=args.limit_per_class,
    )


def collect_audio_files(data_dir: Path, limit_per_class: int | None) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for label in ("fake", "real"):
        class_dir = data_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing expected class directory: {class_dir}")
        wav_files = sorted(class_dir.glob("*.wav"))
        if limit_per_class is not None:
            wav_files = wav_files[:limit_per_class]
        rows.extend((label, wav_file) for wav_file in wav_files)
    return rows


def evaluate_model(config: EvaluationConfig, model_id: str, files: list[tuple[str, Path]]) -> tuple[dict, list[dict]]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from unified_xai.core.registry import get_audio_model
    from unified_xai.services.audio_service import run_audio_inference

    spec = get_audio_model(model_id)
    if not spec.is_available:
        raise FileNotFoundError(f"Audio model '{model_id}' is not available.")

    records: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score_fake: list[float] = []

    for true_label, wav_path in files:
        result = run_audio_inference(
            model_id=model_id,
            audio_bytes=wav_path.read_bytes(),
            source_name=wav_path.name,
        )
        predicted_label = result.prediction.label
        fake_probability = float(result.prediction.probabilities["fake"])

        y_true.append(1 if true_label == "fake" else 0)
        y_pred.append(1 if predicted_label == "fake" else 0)
        y_score_fake.append(fake_probability)
        records.append(
            {
                "model_id": model_id,
                "file_name": wav_path.name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "fake_probability": fake_probability,
                "real_probability": float(result.prediction.probabilities["real"]),
                "correct": predicted_label == true_label,
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc_fake": float(roc_auc_score(y_true, y_score_fake)),
        "num_examples": len(files),
        "num_fake": int(sum(y_true)),
        "num_real": int(len(y_true) - sum(y_true)),
    }
    if model_id == "deepfake_melspec_cnn":
        metrics["evaluation_note"] = (
            "Packaged original checkpoint evaluated on the local FoR testing subset."
        )
    else:
        metrics["evaluation_note"] = (
            "Local reconstruction checkpoint. Metrics on this testing subset are not directly comparable "
            "because the local training artifacts were generated from a demo subset workflow."
        )

    return metrics, records


def main() -> None:
    config = parse_args()
    files = collect_audio_files(config.data_dir, config.limit_per_class)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(config.data_dir),
        "limit_per_class": config.limit_per_class,
        "models": {},
    }
    record_rows: list[dict] = []

    for model_id in config.model_ids:
        metrics, rows = evaluate_model(config, model_id, files)
        summary["models"][model_id] = metrics
        record_rows.extend(rows)
        print(f"{model_id}: {metrics}")

    (config.output_dir / "audio_model_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (config.output_dir / "audio_model_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "file_name",
                "true_label",
                "predicted_label",
                "fake_probability",
                "real_probability",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(record_rows)

    print(f"Saved evaluation outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
