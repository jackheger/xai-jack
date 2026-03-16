from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras


DEFAULT_DATA_ROOT = Path(r"C:\Users\jackj\Work\A6\Explanability\train_datasets\archive")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\jackj\Work\A6\Explanability\artifacts\image_models")
LABELS = ("non_malignant", "malignant")
MODEL_CHOICES = ("densenet121", "alexnet")


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    batch_size: int
    epochs_densenet: int
    epochs_densenet_finetune: int
    epochs_alexnet: int
    seed: int
    models: tuple[str, ...]


@dataclass(frozen=True)
class ModelTrainingSpec:
    model_name: str
    artifact_dir_name: str
    image_size: int
    epochs: int
    use_class_weights: bool = True
    initial_learning_rate: float = 1e-3
    fine_tune_learning_rate: float | None = None
    fine_tune_epochs: int = 0
    fine_tune_last_layers: int = 0


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train and export JSRT chest X-ray classifiers for the unified XAI app."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-densenet", type=int, default=4)
    parser.add_argument("--epochs-densenet-finetune", type=int, default=0)
    parser.add_argument("--epochs-alexnet", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["densenet121"],
        help="One or more image architectures to train and export.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs_densenet=args.epochs_densenet,
        epochs_densenet_finetune=args.epochs_densenet_finetune,
        epochs_alexnet=args.epochs_alexnet,
        seed=args.seed,
        models=tuple(args.models),
    )


def build_training_specs(config: TrainingConfig) -> list[ModelTrainingSpec]:
    spec_lookup = {
        "densenet121": ModelTrainingSpec(
            model_name="densenet121",
            artifact_dir_name="jsrt_densenet121",
            image_size=224,
            epochs=config.epochs_densenet,
            use_class_weights=True,
            initial_learning_rate=1e-3,
            fine_tune_learning_rate=1e-5,
            fine_tune_epochs=config.epochs_densenet_finetune,
            fine_tune_last_layers=48,
        ),
        "alexnet": ModelTrainingSpec(
            model_name="alexnet",
            artifact_dir_name="jsrt_alexnet",
            image_size=227,
            epochs=config.epochs_alexnet,
            use_class_weights=True,
            initial_learning_rate=5e-4,
        ),
    }
    return [spec_lookup[model_name] for model_name in config.models]


def load_metadata(config: TrainingConfig) -> pd.DataFrame:
    metadata_path = config.data_root / "jsrt_metadata.csv"
    image_root = config.data_root / "images" / "images"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing image directory: {image_root}")

    df = pd.read_csv(metadata_path)
    df["image_path"] = df["study_id"].map(lambda name: str((image_root / name).resolve()))
    missing_files = df.loc[~df["image_path"].map(lambda value: Path(value).exists())]
    if not missing_files.empty:
        raise FileNotFoundError("Some JSRT image files referenced in metadata are missing.")

    df["label"] = np.where(df["state"].eq("malignant"), "malignant", "non_malignant")
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    df["label_id"] = df["label"].map(label_to_id)
    return df


def split_metadata(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_id"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _load_example(path: tf.Tensor, label: tf.Tensor, image_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_png(image_bytes, channels=1)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    return image, label


def make_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    seed: int,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].to_numpy(), df["label_id"].to_numpy(dtype=np.int32))
    )
    if training:
        dataset = dataset.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: _load_example(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    counts = train_df["label_id"].value_counts().to_dict()
    total = float(len(train_df))
    class_count = float(len(LABELS))
    return {
        int(label_id): total / (class_count * float(count))
        for label_id, count in counts.items()
    }


def build_common_augmentation(name: str) -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.02),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomTranslation(0.03, 0.03),
        ],
        name=name,
    )


def build_densenet121(image_size: int) -> keras.Model:
    augmentation = build_common_augmentation(name="jsrt_densenet_augmentation")

    base = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    backbone_inputs = keras.Input(shape=(image_size, image_size, 3))
    backbone_outputs = base(backbone_inputs)
    backbone = keras.Model(
        inputs=backbone_inputs,
        outputs=backbone_outputs,
        name="jsrt_densenet_backbone",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = keras.applications.densenet.preprocess_input(x)
    x = backbone(x, training=False)
    x = keras.layers.Lambda(lambda tensor: tensor, name="jsrt_gradcam_target")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="jsrt_densenet121")
    return model


def build_alexnet(image_size: int) -> keras.Model:
    augmentation = build_common_augmentation(name="jsrt_alexnet_augmentation")

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = keras.layers.Rescaling(1.0 / 255.0, name="jsrt_alexnet_rescaling")(x)

    x = keras.layers.Conv2D(
        64,
        kernel_size=11,
        strides=4,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_conv1",
    )(x)
    x = keras.layers.BatchNormalization(name="alexnet_bn1")(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="alexnet_pool1")(x)

    x = keras.layers.Conv2D(
        192,
        kernel_size=5,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_conv2",
    )(x)
    x = keras.layers.BatchNormalization(name="alexnet_bn2")(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="alexnet_pool2")(x)

    x = keras.layers.Conv2D(
        384,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_conv3",
    )(x)
    x = keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_conv4",
    )(x)
    x = keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_conv5",
    )(x)
    x = keras.layers.Lambda(lambda tensor: tensor, name="jsrt_alexnet_gradcam_target")(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="alexnet_pool5")(x)

    x = keras.layers.GlobalAveragePooling2D(name="alexnet_gap")(x)
    x = keras.layers.Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal",
        name="alexnet_fc1",
    )(x)
    x = keras.layers.Dropout(0.5, name="alexnet_dropout1")(x)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="jsrt_alexnet")
    return model


def build_model(model_name: str, image_size: int) -> keras.Model:
    if model_name == "densenet121":
        return build_densenet121(image_size)
    if model_name == "alexnet":
        return build_alexnet(image_size)
    raise ValueError(f"Unsupported model '{model_name}'.")


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_model(
    model: keras.Model,
    test_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
) -> dict[str, float]:
    test_ds = make_dataset(test_df, image_size=image_size, batch_size=batch_size, seed=42, training=False)
    probabilities = np.asarray(model.predict(test_ds, verbose=0))
    labels = test_df["label_id"].to_numpy(dtype=np.int32)
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def write_model_summary(model: keras.Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    (output_path / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_training_report(
    output_path: Path,
    config: TrainingConfig,
    training_spec: ModelTrainingSpec,
    history: keras.callbacks.History,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_metrics: dict[str, float],
) -> None:
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": training_spec.model_name,
        "artifact_dir_name": training_spec.artifact_dir_name,
        "config": {
            "data_root": str(config.data_root),
            "output_dir": str(config.output_dir),
            "image_size": training_spec.image_size,
            "batch_size": config.batch_size,
            "epochs": training_spec.epochs,
            "fine_tune_epochs": training_spec.fine_tune_epochs,
            "seed": config.seed,
            "models_requested": list(config.models),
        },
        "training_strategy": {
            "use_class_weights": training_spec.use_class_weights,
            "initial_learning_rate": training_spec.initial_learning_rate,
            "fine_tune_learning_rate": training_spec.fine_tune_learning_rate,
            "fine_tune_last_layers": training_spec.fine_tune_last_layers,
        },
        "labels": list(LABELS),
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "class_balance": {
            "train": train_df["label"].value_counts().to_dict(),
            "validation": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "history": _json_safe(history.history),
        "test_metrics": _json_safe(test_metrics),
    }
    (output_path / "training_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )


def save_split_files(
    output_path: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_df.to_csv(output_path / "train_split.csv", index=False)
    val_df.to_csv(output_path / "validation_split.csv", index=False)
    test_df.to_csv(output_path / "test_split.csv", index=False)


def reset_output_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def merge_histories(*histories: keras.callbacks.History) -> keras.callbacks.History:
    merged = keras.callbacks.History()
    merged.history = {}
    for history in histories:
        for key, values in history.history.items():
            merged.history.setdefault(key, []).extend(values)
    return merged


def make_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]


def configure_densenet_for_finetuning(model: keras.Model, fine_tune_last_layers: int) -> None:
    backbone = model.get_layer("jsrt_densenet_backbone")
    backbone.trainable = True

    if fine_tune_last_layers > 0:
        frozen_boundary = max(0, len(backbone.layers) - fine_tune_last_layers)
    else:
        frozen_boundary = 0

    for index, layer in enumerate(backbone.layers):
        if index < frozen_boundary:
            layer.trainable = False
            continue
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def train_and_export_model(
    training_spec: ModelTrainingSpec,
    config: TrainingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    print()
    print(f"Training {training_spec.model_name} on full JSRT replacement dataset...")
    print(
        f"Image size={training_spec.image_size}, batch_size={config.batch_size}, epochs={training_spec.epochs}"
    )

    train_ds = make_dataset(
        train_df,
        image_size=training_spec.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        training=True,
    )
    val_ds = make_dataset(
        val_df,
        image_size=training_spec.image_size,
        batch_size=config.batch_size,
        seed=config.seed,
        training=False,
    )

    model = build_model(training_spec.model_name, image_size=training_spec.image_size)
    compile_model(model, training_spec.initial_learning_rate)
    class_weights = compute_class_weights(train_df) if training_spec.use_class_weights else None

    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_spec.epochs,
        verbose=1,
        callbacks=make_callbacks(),
        class_weight=class_weights,
    )
    combined_history = initial_history

    if (
        training_spec.model_name == "densenet121"
        and training_spec.fine_tune_epochs > 0
        and training_spec.fine_tune_learning_rate is not None
    ):
        print(
            "Starting DenseNet fine-tuning phase "
            f"(last {training_spec.fine_tune_last_layers} backbone layers, "
            f"lr={training_spec.fine_tune_learning_rate})"
        )
        configure_densenet_for_finetuning(model, training_spec.fine_tune_last_layers)
        compile_model(model, training_spec.fine_tune_learning_rate)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_spec.fine_tune_epochs,
            verbose=1,
            callbacks=make_callbacks(),
            class_weight=class_weights,
        )
        combined_history = merge_histories(initial_history, fine_tune_history)

    test_metrics = evaluate_model(
        model=model,
        test_df=test_df,
        image_size=training_spec.image_size,
        batch_size=config.batch_size,
    )

    output_path = config.output_dir / training_spec.artifact_dir_name
    reset_output_path(output_path)
    model.save(output_path)
    save_split_files(output_path, train_df, val_df, test_df)
    save_training_report(
        output_path=output_path,
        config=config,
        training_spec=training_spec,
        history=combined_history,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        test_metrics=test_metrics,
    )
    write_model_summary(model, output_path)

    print(f"Saved image model artifact to {output_path}")
    print("Test metrics:", test_metrics)
    return test_metrics


def main() -> None:
    config = parse_args()
    metadata = load_metadata(config)
    train_df, val_df, test_df = split_metadata(metadata, config.seed)
    training_specs = build_training_specs(config)

    print("Using JSRT replacement dataset:")
    print(metadata["label"].value_counts().to_string())
    print("Train/val/test split:", len(train_df), len(val_df), len(test_df))
    print("Requested models:", ", ".join(spec.model_name for spec in training_specs))

    metrics_by_model: dict[str, dict[str, float]] = {}
    for training_spec in training_specs:
        metrics_by_model[training_spec.model_name] = train_and_export_model(
            training_spec=training_spec,
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

    print()
    print("Finished image training run.")
    for model_name, metrics in metrics_by_model.items():
        print(f"- {model_name}: {metrics}")


if __name__ == "__main__":
    main()
