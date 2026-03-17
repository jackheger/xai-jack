# Project Structure

This document describes where the important parts of the unified project live and how the current audio and chest X-ray pipelines work.

## Top-Level Directories

- [src/unified_xai](../src/unified_xai): main application package
- [scripts](../scripts): helper scripts for preprocessing and training
- [artifacts](../artifacts): generated model files and cached outputs
- [sample_inputs](../sample_inputs): local files for manual testing
- [docs](../docs): structure notes and technical report

Additional submission-oriented docs:

- [technical-report.md](../docs/technical-report.md)
- [pipeline-overview.md](../docs/pipeline-overview.md)
- [pipeline-technical.md](../docs/pipeline-technical.md)

Local-only resources used during development, but intentionally not committed to Git:

- `train_datasets/`
- `Deepfake-Audio-Detection-with-XAI/`
- `LungCancerDetection/`
- internal demo-only notes

## Unified App Code

### UI Layer

- [app.py](../app.py): lightweight bootstrap used by Streamlit
- [src/unified_xai/ui/app.py](../src/unified_xai/ui/app.py): unified audio and chest X-ray interface

This is where the user:

- uploads audio
- uploads chest X-ray images
- selects a model
- sees preprocessing warnings
- runs inference
- sees the prediction and spectrogram
- sees the processed X-ray image
- switches between `Single Analysis` and `Compare XAI` tabs

### Registry Layer

- [src/unified_xai/core/registry.py](../src/unified_xai/core/registry.py)

This file is the source of truth for audio model metadata:

- model id
- display name
- artifact path
- labels
- expected input size
- advertised XAI methods
- availability status

If a model shows `Missing checkpoint` in the UI, the reason is almost always here: the registry expects a SavedModel at the configured path, and that folder is missing.

### Service Layer

- [src/unified_xai/services/audio_service.py](../src/unified_xai/services/audio_service.py)
- [src/unified_xai/services/explanation_service.py](../src/unified_xai/services/explanation_service.py)
- [src/unified_xai/services/image_service.py](../src/unified_xai/services/image_service.py)
- [src/unified_xai/services/image_explanation_service.py](../src/unified_xai/services/image_explanation_service.py)

This file orchestrates the audio inference pipeline:

1. take audio bytes from the UI
2. preprocess them into a mel spectrogram
3. call the selected model
4. return prediction data and metadata to the UI

`explanation_service.py` is the matching orchestration layer for XAI:

1. determine which explainers are implemented for the selected model
2. dispatch the explanation request
3. return the visualization and explanation metadata to the UI

`image_service.py` and `image_explanation_service.py` do the equivalent work for chest X-ray inference and XAI.

### Audio Modality Layer

- [src/unified_xai/modalities/audio/preprocess.py](../src/unified_xai/modalities/audio/preprocess.py)
- [src/unified_xai/modalities/audio/model.py](../src/unified_xai/modalities/audio/model.py)
- [src/unified_xai/modalities/audio/explainers.py](../src/unified_xai/modalities/audio/explainers.py)

Responsibilities:

- `preprocess.py`
  - detect input format
  - decode `.wav` or `.mp3`
  - convert MP3 input to WAV/PCM in memory
  - generate mel spectrogram images
- `model.py`
  - load TensorFlow SavedModels
  - cache loaded models
  - normalize output probabilities
  - return the final predicted label
- `explainers.py`
  - implement model-facing XAI methods for audio
  - run `Grad-CAM` against the selected checkpoint
  - run `LIME` against the selected checkpoint
  - run `SHAP` against the selected checkpoint
  - return explanation overlays ready for Streamlit rendering

### Image Modality Layer

- [src/unified_xai/modalities/image/preprocess.py](../src/unified_xai/modalities/image/preprocess.py)
- [src/unified_xai/modalities/image/model.py](../src/unified_xai/modalities/image/model.py)
- [src/unified_xai/modalities/image/explainers.py](../src/unified_xai/modalities/image/explainers.py)

Responsibilities:

- `preprocess.py`
  - decode uploaded chest X-ray images
  - normalize them into grayscale RGB tensors for the classifier
  - capture image metadata for the UI
- `model.py`
  - load the exported DenseNet121 SavedModel
  - run batch or single-image inference
  - normalize prediction probabilities
- `explainers.py`
  - implement `Grad-CAM` for the chest X-ray model
  - implement `LIME` for the chest X-ray model
  - return overlays ready for Streamlit rendering

## Training And Model Artifacts

### Exported Checkpoints

- [artifacts/audio_models](../artifacts/audio_models)

Expected local artifact folders:

- `artifacts/audio_models/deepfake_melspec_cnn`
- `artifacts/audio_models/vgg16`
- `artifacts/audio_models/custom_cnn`
- `artifacts/image_models/jsrt_densenet121`
- `artifacts/image_models/jsrt_alexnet`

The Git repository intentionally keeps only the default runnable checkpoints:

- `artifacts/audio_models/deepfake_melspec_cnn`
- `artifacts/image_models/jsrt_densenet121`

The larger local reconstruction checkpoints can be regenerated locally and are not required for the default demo path.

Each artifact directory should contain a real TensorFlow SavedModel:

- `saved_model.pb`
- `variables/`
- optional metadata such as `training_report.json`

### Training Scripts

- [scripts/train_audio_models.py](../scripts/train_audio_models.py)
- [scripts/train_image_models.py](../scripts/train_image_models.py)
- [scripts/convert_audio_to_wav.py](../scripts/convert_audio_to_wav.py)
- [scripts/evaluate_audio_models.py](../scripts/evaluate_audio_models.py)
- [scripts/verify_xai_stack.py](../scripts/verify_xai_stack.py)

`train_audio_models.py` is the script to rerun if you want to regenerate local audio artifacts from the downloaded dataset subset.

`train_image_models.py` reconstructs the missing image pipeline from the lung repo README using the local JSRT replacement dataset in `train_datasets/archive`. It now supports:

- `DenseNet121`
- `AlexNet-style`

`evaluate_audio_models.py` evaluates the currently installed audio checkpoints on the local FoR testing subset and writes summary artifacts under `artifacts/evaluation`.

`verify_xai_stack.py` is the slower full-stack verification pass for all required XAI methods on the recommended audio and chest X-ray demo paths.

### Smoke Tests

- [tests/test_registry.py](../tests/test_registry.py)
- [tests/test_audio_pipeline.py](../tests/test_audio_pipeline.py)
- [tests/test_image_pipeline.py](../tests/test_image_pipeline.py)
- [tests/test_demo_recommendations.py](../tests/test_demo_recommendations.py)

These tests are intentionally small and demo-oriented. They validate:

- installed checkpoint discovery
- audio inference on a bundled sample
- audio `Grad-CAM` generation
- audio `LIME` generation
- chest X-ray inference on a curated JSRT sample
- chest X-ray `Grad-CAM` generation
- recommended demo sample correctness for the safest audio and chest X-ray demo models

Run them with:

```bash
uv run pytest
```

## Original Repositories

### Audio Source Repo

The original audio source repo was `Deepfake-Audio-Detection-with-XAI`.

Important parts:

- notebooks under `Code/`
- old demo app at `Streamlit/app.py`
- packaged model at `Streamlit/saved_model/model`

This repo is the source for:

- original model/training ideas
- original XAI experiments
- original demo code

### Image Source Repo

The original image source repo was `LungCancerDetection`.

This repo is still mainly reference material. The working image pipeline in the unified app is therefore a local implementation inspired by that README rather than code reused directly from the original repo.

## What "Wire XAI For These Checkpoints" Means

For this project, it does not mean only "show a dropdown with LIME or SHAP".

It means:

1. the UI must let the user choose an explainer for the selected model
2. the backend must run that explainer against the exact checkpoint chosen in the UI
3. the explanation must be generated from the same spectrogram used for prediction
4. the output must be rendered in the app as an image, mask, heatmap, or overlay
5. unsupported explainers must be filtered automatically

In practice for the current project:

- audio has `Grad-CAM`, `LIME`, and `SHAP`
- chest X-ray has `Grad-CAM`, `LIME`, and `SHAP`
- `SHAP` is implemented, but it is slower than `LIME`

The clean next place for this code is likely:

- `src/unified_xai/services/explanation_service.py`
- `src/unified_xai/modalities/audio/explainers.py`

That keeps the UI thin and prevents explainer logic from being mixed directly into Streamlit callbacks.

