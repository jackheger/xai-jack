from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_MODEL_ARTIFACTS = PROJECT_ROOT / "artifacts" / "audio_models"
IMAGE_MODEL_ARTIFACTS = PROJECT_ROOT / "artifacts" / "image_models"


@dataclass(frozen=True)
class AudioModelSpec:
    model_id: str
    display_name: str
    artifact_path: Path
    labels: tuple[str, ...]
    input_size: tuple[int, int]
    supported_explainers: tuple[str, ...]
    description: str
    source_note: str
    gradcam_target_layer: str | None = None

    @property
    def is_available(self) -> bool:
        return self.artifact_path.exists()

    @property
    def relative_artifact_path(self) -> str:
        try:
            return str(self.artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.artifact_path)


@dataclass(frozen=True)
class ImageModelSpec:
    model_id: str
    display_name: str
    artifact_path: Path
    labels: tuple[str, ...]
    input_size: tuple[int, int]
    supported_explainers: tuple[str, ...]
    description: str
    source_note: str
    gradcam_target_layer: str | None = None

    @property
    def is_available(self) -> bool:
        return self.artifact_path.exists()

    @property
    def relative_artifact_path(self) -> str:
        try:
            return str(self.artifact_path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.artifact_path)


AUDIO_MODELS: dict[str, AudioModelSpec] = {
    "deepfake_melspec_cnn": AudioModelSpec(
        model_id="deepfake_melspec_cnn",
        display_name="MobileNet-style classifier (packaged)",
        artifact_path=AUDIO_MODEL_ARTIFACTS / "deepfake_melspec_cnn",
        labels=("real", "fake"),
        input_size=(224, 224),
        supported_explainers=("lime", "gradcam", "shap"),
        description=(
            "Packaged TensorFlow SavedModel copied from the original audio repository into the "
            "unified artifacts directory so fresh clones can run the audio demo path."
        ),
        source_note="Bundled checkpoint included in the repository for the default audio demo flow.",
        gradcam_target_layer="conv_pw_13_relu",
    ),
    "inception_v3": AudioModelSpec(
        model_id="inception_v3",
        display_name="InceptionV3",
        artifact_path=AUDIO_MODEL_ARTIFACTS / "inception_v3",
        labels=("real", "fake"),
        input_size=(224, 224),
        supported_explainers=("lime", "gradcam", "shap"),
        description="Audio classifier described in the original notebooks.",
        source_note="Notebook references exist, but no checkpoint is installed yet.",
    ),
    "vgg16": AudioModelSpec(
        model_id="vgg16",
        display_name="VGG16",
        artifact_path=AUDIO_MODEL_ARTIFACTS / "vgg16",
        labels=("real", "fake"),
        input_size=(224, 224),
        supported_explainers=("lime", "gradcam", "shap"),
        description="Transfer-learning classifier described in the original notebooks.",
        source_note="Checkpoint trained locally from the downloaded subset and exported as a SavedModel.",
        gradcam_target_layer="vgg16",
    ),
    "custom_cnn": AudioModelSpec(
        model_id="custom_cnn",
        display_name="Custom CNN",
        artifact_path=AUDIO_MODEL_ARTIFACTS / "custom_cnn",
        labels=("real", "fake"),
        input_size=(224, 224),
        supported_explainers=("lime", "gradcam", "shap"),
        description="Custom convolutional network described in the original notebooks.",
        source_note="Checkpoint trained locally from the downloaded subset and exported as a SavedModel.",
        gradcam_target_layer="conv2d_2",
    ),
    "resnet50": AudioModelSpec(
        model_id="resnet50",
        display_name="ResNet50",
        artifact_path=AUDIO_MODEL_ARTIFACTS / "resnet50",
        labels=("real", "fake"),
        input_size=(224, 224),
        supported_explainers=("lime", "gradcam", "shap"),
        description="ResNet-based classifier described in the original notebooks.",
        source_note="Notebook references exist, but no checkpoint is installed yet.",
    ),
}

IMAGE_MODELS: dict[str, ImageModelSpec] = {
    "jsrt_densenet121": ImageModelSpec(
        model_id="jsrt_densenet121",
        display_name="DenseNet121 (JSRT)",
        artifact_path=IMAGE_MODEL_ARTIFACTS / "jsrt_densenet121",
        labels=("non_malignant", "malignant"),
        input_size=(224, 224),
        supported_explainers=("gradcam", "lime", "shap"),
        description=(
            "DenseNet121 transfer-learning classifier trained on the local JSRT replacement dataset "
            "to detect malignant chest X-rays."
        ),
        source_note=(
            "Local replacement for the missing CheXpert-based code path described in the lung repo."
        ),
        gradcam_target_layer="jsrt_gradcam_target",
    ),
    "jsrt_alexnet": ImageModelSpec(
        model_id="jsrt_alexnet",
        display_name="AlexNet-style (JSRT)",
        artifact_path=IMAGE_MODEL_ARTIFACTS / "jsrt_alexnet",
        labels=("non_malignant", "malignant"),
        input_size=(227, 227),
        supported_explainers=("gradcam", "lime", "shap"),
        description=(
            "AlexNet-style chest X-ray classifier trained on the full local JSRT replacement dataset "
            "to match the original lung repo architecture family."
        ),
        source_note=(
            "Local reconstruction of the AlexNet checkpoint described in the lung repo README."
        ),
        gradcam_target_layer="jsrt_alexnet_gradcam_target",
    ),
}


def list_audio_models() -> list[AudioModelSpec]:
    return list(AUDIO_MODELS.values())


def list_available_audio_models() -> list[AudioModelSpec]:
    return [spec for spec in AUDIO_MODELS.values() if spec.is_available]


def list_unavailable_audio_models() -> list[AudioModelSpec]:
    return [spec for spec in AUDIO_MODELS.values() if not spec.is_available]


def get_audio_model(model_id: str) -> AudioModelSpec:
    try:
        return AUDIO_MODELS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown audio model '{model_id}'.") from exc


def list_image_models() -> list[ImageModelSpec]:
    return list(IMAGE_MODELS.values())


def list_available_image_models() -> list[ImageModelSpec]:
    return [spec for spec in IMAGE_MODELS.values() if spec.is_available]


def list_unavailable_image_models() -> list[ImageModelSpec]:
    return [spec for spec in IMAGE_MODELS.values() if not spec.is_available]


def get_image_model(model_id: str) -> ImageModelSpec:
    try:
        return IMAGE_MODELS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown image model '{model_id}'.") from exc
