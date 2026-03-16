# Technical Report

## 1. Project Goal

The project objective was to unify two separate Explainable AI systems into one interface:

- deepfake audio detection
- chest X-ray malignancy detection

The final unified system allows the user to:

- upload audio or chest X-ray data
- select a compatible model
- run one XAI method or several side by side
- avoid invalid XAI choices through compatibility filtering

## 2. Source Repositories And Reconstruction Scope

### Audio Repository

The downloaded audio repository already contained:

- notebooks for several architectures
- an older Streamlit prototype
- one packaged TensorFlow SavedModel

That made the audio integration partly a refactor:

- reuse the packaged checkpoint
- extract the preprocessing and inference logic into reusable modules
- rebuild the UI inside the unified application
- reimplement the explainers in the new architecture

### Lung Repository

The downloaded lung repository did not contain runnable inference code or exported checkpoints. It mainly contained documentation and figures. Because of that, the chest X-ray pipeline in this repo is a reconstruction based on:

- the original README description
- the assignment brief
- the locally available replacement dataset

This is why the chest X-ray code path in the unified app is locally built rather than directly reused from the original repository.

## 3. Architecture

The project uses a thin Streamlit UI and a layered Python backend.

### UI Layer

- [app.py](../app.py)
- [src/unified_xai/ui/app.py](../src/unified_xai/ui/app.py)

Responsibilities:

- model selection
- file upload and sample selection
- compatibility-aware explainer selection
- presentation of prediction, metadata, and explanation overlays
- comparison view

### Registry Layer

- [src/unified_xai/core/registry.py](../src/unified_xai/core/registry.py)

Responsibilities:

- define all known model checkpoints
- store expected artifact paths
- advertise supported explainers
- expose availability status to the UI

### Service Layer

- [src/unified_xai/services/audio_service.py](../src/unified_xai/services/audio_service.py)
- [src/unified_xai/services/explanation_service.py](../src/unified_xai/services/explanation_service.py)
- [src/unified_xai/services/image_service.py](../src/unified_xai/services/image_service.py)
- [src/unified_xai/services/image_explanation_service.py](../src/unified_xai/services/image_explanation_service.py)

Responsibilities:

- orchestrate preprocessing, model loading, prediction, and explanation
- keep Streamlit code thin
- make explainers run against the same checkpoint selected in the UI

### Modality Layers

Audio-specific modules:

- [src/unified_xai/modalities/audio/preprocess.py](../src/unified_xai/modalities/audio/preprocess.py)
- [src/unified_xai/modalities/audio/model.py](../src/unified_xai/modalities/audio/model.py)
- [src/unified_xai/modalities/audio/explainers.py](../src/unified_xai/modalities/audio/explainers.py)

Image-specific modules:

- [src/unified_xai/modalities/image/preprocess.py](../src/unified_xai/modalities/image/preprocess.py)
- [src/unified_xai/modalities/image/model.py](../src/unified_xai/modalities/image/model.py)
- [src/unified_xai/modalities/image/explainers.py](../src/unified_xai/modalities/image/explainers.py)

## 4. Dataset Decisions

### Audio Dataset

The audio work uses the downloaded Fake-or-Real two-second subset under `train_datasets/for-2sec`.

Important limitation:

- the local data is incomplete for a full clean retraining story
- `testing/fake`, `testing/real`, and `training/fake` are available locally
- `training/real` was not present in the workspace during this submission-readiness pass

Because of that, the packaged original checkpoint is treated as the primary audio model, while the locally trained `VGG16` and `Custom CNN` checkpoints are documented as demo reconstructions rather than final academic results.

### Chest X-ray Dataset

The original lung repository referenced CheXpert, but that dataset was not used locally because of size and availability constraints. The local replacement dataset is the JSRT-based dataset in `train_datasets/archive`.

The reconstructed task was adapted to:

- `malignant`
- `non_malignant`

The local JSRT metadata contains 247 images:

- 100 malignant
- 54 benign
- 93 non-nodule

This mapping is a practical replacement for the original README description, but it is not equivalent to the original CheXpert-scale setup.

## 5. Selected Models

### Audio Models

- `deepfake_melspec_cnn`
  - packaged original checkpoint
  - recommended audio demo model
- `vgg16`
  - local reconstruction checkpoint
- `custom_cnn`
  - local reconstruction checkpoint

Planned but not installed:

- `inception_v3`
- `resnet50`

### Chest X-ray Models

- `jsrt_densenet121`
  - transfer-learning style DenseNet checkpoint
  - recommended chest X-ray demo model
- `jsrt_alexnet`
  - AlexNet-style reconstruction to match the original lung README description

## 6. Selected XAI Methods

The assignment required `LIME`, `Grad-CAM`, and `SHAP`. All three are implemented in the unified project.

### Audio

The audio pipeline first converts the waveform to a mel spectrogram image. XAI then operates on that 2D spectrogram.

- `Grad-CAM`
  - model-specific heatmap on the selected audio CNN
- `LIME`
  - perturbation-based explanation over spectrogram regions
- `SHAP`
  - additive contribution visualization over the spectrogram

### Chest X-ray

- `Grad-CAM`
  - model-specific heatmap for the selected image checkpoint
- `LIME`
  - perturbation-based explanation over image regions
- `SHAP`
  - additive contribution explanation over image regions

## 7. What Was Reused Vs Rebuilt

### Reused

- original packaged audio checkpoint
- architecture and preprocessing ideas from the original audio notebooks
- model family requirements from the two original READMEs

### Rebuilt

- unified Streamlit interface
- registry-driven model selection
- compatibility-aware explainer selection
- audio MP3 preprocessing pipeline
- chest X-ray training, export, and inference pipeline
- chest X-ray explainers
- comparison view
- verification scripts and smoke tests

## 8. Evaluation Summary

### Audio

Audio evaluation was run with [evaluate_audio_models.py](../scripts/evaluate_audio_models.py) on the local FoR testing subset. The local summary output is written to `artifacts/evaluation/audio_model_evaluation_summary.json`.

Key result:

- `deepfake_melspec_cnn`
  - accuracy: `0.6985`
  - fake recall: `0.8732`
  - fake F1: `0.7433`
  - ROC AUC: `0.8116`

The local `VGG16` and `Custom CNN` checkpoints underperform the packaged model and are therefore not recommended as the primary demo path.

### Chest X-ray

DenseNet report:

- source: [training_report.json](../artifacts/image_models/jsrt_densenet121/training_report.json)
- test accuracy: `0.5789`
- precision: `0.4815`
- recall: `0.8667`
- F1: `0.6190`
- ROC AUC: `0.6435`

AlexNet report:

- source: [training_report.json](../artifacts/image_models/jsrt_alexnet/training_report.json)
- test accuracy: `0.6053`
- precision: `0.5000`
- recall: `0.4667`
- F1: `0.4828`
- ROC AUC: `0.5623`

Interpretation:

- `jsrt_densenet121` is the better demo model because it captures malignant cases more reliably
- `jsrt_alexnet` is valuable for architectural coverage and assignment alignment, but not the strongest clinical-style demo path

An additional DenseNet fine-tuning attempt was tested during this pass and did not improve the recommended checkpoint enough to replace the current exported DenseNet artifact.

## 9. Verification Strategy

### Fast Verification

- [tests](../tests)
- run with `uv run pytest`

These tests cover:

- installed model discovery
- inference for both modalities
- selected explainer smoke paths
- recommended demo samples for audio and chest X-ray

### Full Required-XAI Verification

- script: [verify_xai_stack.py](../scripts/verify_xai_stack.py)
- output: local files written under `artifacts/verification/`

This script verifies:

- audio `Grad-CAM`
- audio `LIME`
- audio `SHAP`
- chest X-ray `Grad-CAM`
- chest X-ray `LIME`
- chest X-ray `SHAP`

`SHAP` is documented outside the main test suite because it is materially slower than the other explainers on CPU.

## 10. Demo Recommendations

Recommended live-demo path:

- audio:
  - model: `deepfake_melspec_cnn`
  - sample: `demo_real_packaged_correct.wav` or `demo_fake_packaged_correct.wav`
  - explainer: `Grad-CAM` first, `LIME` second
- chest X-ray:
  - model: `jsrt_densenet121`
  - sample: `jsrt_malignant_jpcln047.png`
  - explainer: `Grad-CAM` first
- comparison view:
  - use `Grad-CAM` plus `LIME`
  - add `SHAP` only if there is time during the demo

## 11. Current Limitations

- The lung repository was documentation-only, so the image pipeline is a reconstruction rather than direct integration.
- JSRT is much smaller than CheXpert, so image-model performance is limited.
- The audio retraining story is incomplete with the locally available dataset.
- The probability outputs shown in the UI should be read as model scores, not calibrated certainty.
- `SHAP` is slower than `Grad-CAM` and `LIME`.

## 12. Improvements Over The Original Repositories

- one interface instead of two disconnected sources
- modality-aware model selection
- compatibility-aware XAI selection
- comparison view for multiple explainers
- reusable training and verification scripts
- explicit artifact registry
- better demo safety through curated sample files and smoke tests

