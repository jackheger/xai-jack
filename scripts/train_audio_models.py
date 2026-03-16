from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import tensorflow as tf
from tensorflow import keras

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_DIR = Path(
    r"C:\Users\jackj\Work\A6\Explanability\train_datasets\for-2sec\for-2seconds\testing"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\jackj\Work\A6\Explanability\artifacts\audio_models"
)
DEFAULT_SPECTROGRAM_DIR = Path(
    r"C:\Users\jackj\Work\A6\Explanability\artifacts\generated_spectrograms\for_2seconds_testing"
)
SUPPORTED_MODELS = ("vgg16", "custom_cnn")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    spectrogram_dir: Path
    models: tuple[str, ...]
    image_size: int
    batch_size: int
    epochs: int
    validation_split: float
    seed: int
    regenerate_spectrograms: bool


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train and export audio deepfake classifiers from spectrogram datasets. "
            "This script is intended for quick artifact creation from the local subset."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing class folders such as fake/ and real/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where SavedModel artifacts will be exported.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=DEFAULT_SPECTROGRAM_DIR,
        help="Directory where spectrogram PNGs will be generated or reused.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help="Models to train and export.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regenerate-spectrograms",
        action="store_true",
        help="Regenerate spectrogram PNGs even if they already exist.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        spectrogram_dir=args.spectrogram_dir,
        models=tuple(args.models),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        seed=args.seed,
        regenerate_spectrograms=args.regenerate_spectrograms,
    )


def ensure_dataset_layout(data_dir: Path) -> None:
    for class_name in ("fake", "real"):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing expected class directory: {class_dir}")


def create_spectrogram(audio_file: Path, image_file: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")

    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(log_mel, sr=sample_rate, ax=ax)

    image_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def ensure_spectrogram_dataset(config: TrainingConfig) -> Path:
    spectrogram_root = config.spectrogram_dir
    spectrogram_root.mkdir(parents=True, exist_ok=True)

    for class_name in ("fake", "real"):
        source_dir = config.data_dir / class_name
        target_dir = spectrogram_root / class_name
        wav_files = sorted(source_dir.glob("*.wav"))
        print(f"Preparing spectrograms for {class_name}: {len(wav_files)} audio files")

        for index, wav_file in enumerate(wav_files, start=1):
            image_path = target_dir / f"{wav_file.stem}.png"
            if image_path.exists() and not config.regenerate_spectrograms:
                continue
            create_spectrogram(wav_file, image_path)
            if index % 100 == 0 or index == len(wav_files):
                print(f"  {class_name}: {index}/{len(wav_files)} spectrograms ready")

    return spectrogram_root


def build_datasets(
    config: TrainingConfig,
    spectrogram_dir: Path,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int]:
    dataset_kwargs = {
        "directory": str(spectrogram_dir),
        "labels": "inferred",
        "label_mode": "binary",
        "color_mode": "rgb",
        "batch_size": config.batch_size,
        "image_size": (config.image_size, config.image_size),
        "shuffle": True,
        "seed": config.seed,
        "validation_split": config.validation_split,
    }

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        **dataset_kwargs,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        **dataset_kwargs,
    )

    class_names = list(train_ds.class_names)
    total_samples = sum(
        len(list((spectrogram_dir / class_name).glob("*.png"))) for class_name in class_names
    )
    return train_ds, val_ds, class_names, total_samples


def prepare_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE

    def _map(images, labels):
        images = tf.cast(images, tf.float32)
        return images, labels

    prepared = dataset.map(_map, num_parallel_calls=autotune)
    if training:
        prepared = prepared.shuffle(256, reshuffle_each_iteration=True)
    return prepared.prefetch(autotune)


def build_vgg16(image_size: int) -> keras.Model:
    base_model = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="vgg16_audio_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn(image_size: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(image_size, image_size, 3)),
            keras.layers.Rescaling(1.0 / 255.0),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="custom_cnn_audio_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, image_size: int) -> keras.Model:
    if model_name == "vgg16":
        return build_vgg16(image_size)
    if model_name == "custom_cnn":
        return build_custom_cnn(image_size)
    raise ValueError(f"Unsupported model '{model_name}'.")


def export_model_artifact(model: keras.Model, output_path: Path) -> None:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def save_report(
    output_path: Path,
    config: TrainingConfig,
    class_names: list[str],
    total_samples: int,
    history: keras.callbacks.History,
    evaluation: list[float],
) -> None:
    history_data = history.history
    config_data = {
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "spectrogram_dir": str(config.spectrogram_dir),
        "models": list(config.models),
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "validation_split": config.validation_split,
        "seed": config.seed,
        "regenerate_spectrograms": config.regenerate_spectrograms,
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_data,
        "class_names": class_names,
        "total_samples_in_source_dir": total_samples,
        "history": history_data,
        "evaluation": {
            "loss": float(evaluation[0]),
            "accuracy": float(evaluation[1]),
        },
    }
    (output_path / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    config = parse_args()
    ensure_dataset_layout(config.data_dir)
    print(f"Using dataset directory: {config.data_dir}")
    print("Warning: this run trains on the local subset only for quick artifact generation.")
    spectrogram_dir = ensure_spectrogram_dataset(config)
    print(f"Using spectrogram cache: {spectrogram_dir}")

    train_raw, val_raw, class_names, total_samples = build_datasets(config, spectrogram_dir)
    print(f"Classes: {class_names}")
    print(f"Total files in subset: {total_samples}")

    for model_name in config.models:
        print(f"\n=== Training {model_name} ===")
        model = build_model(model_name, config.image_size)
        train_ds = prepare_dataset(train_raw, training=True)
        val_ds = prepare_dataset(val_raw, training=False)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            verbose=1,
        )
        evaluation = model.evaluate(val_ds, verbose=0)

        output_path = config.output_dir / model_name
        export_model_artifact(model, output_path)
        save_report(output_path, config, class_names, total_samples, history, evaluation)
        print(
            f"Saved {model_name} artifact to {output_path} "
            f"(val_accuracy={evaluation[1]:.4f})"
        )
        keras.backend.clear_session()


if __name__ == "__main__":
    main()
