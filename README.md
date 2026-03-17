# Unified Explainable AI Interface

### Jack Heger 2026


## Project Overview

This project refactors two separate Explainable AI repositories into one Streamlit interface that supports:

- audio deepfake classification from `.wav` and `.mp3`
- chest X-ray classification from `.png`, `.jpg`, and `.jpeg`
- model selection per modality
- XAI selection with automatic compatibility filtering
- side-by-side comparison of multiple XAI methods on the same input

The unified app now supports both required modalities from the assignment:

- audio deepfake detection reconstructed from the original `Deepfake-Audio-Detection-with-XAI` source material
- chest X-ray malignancy detection reconstructed from the README-only `LungCancerDetection` repository

## Assignment Coverage

Current status against the course brief:

- Unified interface for audio and chest X-ray inputs: complete
- Model selection: complete
- Required XAI methods:
  - audio: `Grad-CAM`, `LIME`, `SHAP`
  - chest X-ray: `Grad-CAM`, `LIME`, `SHAP`
- Automatic filtering of incompatible methods: complete
- Comparison tab with multiple explainers: complete
- Local training/export scripts for reconstructed checkpoints: complete
- Documentation and verification artifacts: complete, with a few placeholders for final submission metadata

## Recommended Demo Setup

These are the safest demo paths in the current workspace.

### Recommended Audio Demo Path

- Model: `MobileNet-style classifier (packaged)`
- Model id: `deepfake_melspec_cnn`
- Good sample files:
  - [demo_real_packaged_correct.wav](sample_inputs/audio/demo_real_packaged_correct.wav)
  - [demo_fake_packaged_correct.wav](sample_inputs/audio/demo_fake_packaged_correct.wav)
- Best live-demo explainers:
  - `Grad-CAM`
  - `LIME`
- `SHAP` works, but it is slower on CPU

The packaged audio checkpoint is the recommended demo model because it performs materially better than the locally reconstructed audio checkpoints on the local FoR testing subset. The evaluation script writes its local summary to `artifacts/evaluation/audio_model_evaluation_summary.json`.

### Recommended Chest X-ray Demo Path

- Model: `DenseNet121 (JSRT)`
- Model id: `jsrt_densenet121`
- Good sample files:
  - [jsrt_malignant_jpcln047.png](sample_inputs/images/jsrt_malignant_jpcln047.png)
  - [jsrt_benign_jpcln036.png](sample_inputs/images/jsrt_benign_jpcln036.png)
  - [jsrt_non_nodule_jpcnn086.png](sample_inputs/images/jsrt_non_nodule_jpcnn086.png)
- Best live-demo explainer:
  - `Grad-CAM`
- Comparison-friendly explainers:
  - `Grad-CAM`
  - `LIME`
  - `SHAP`

The DenseNet checkpoint is the recommended chest X-ray demo model because it remains the strongest balance of recall and F1 among the current image checkpoints. See [technical-report.md](docs/technical-report.md).

## Installed Models

### Audio

- Installed:
  - `MobileNet-style classifier (packaged)`
  - `VGG16`
  - `Custom CNN`
- Not installed:
  - `InceptionV3`
  - `ResNet50`

### Chest X-ray

- Installed:
  - `DenseNet121 (JSRT)`
  - `AlexNet-style (JSRT)`

Bundled directly in this Git repository for a runnable clone:

- audio: `deepfake_melspec_cnn`
- chest X-ray: `jsrt_densenet121`

## Repository Structure

High-level entry points:

- [app.py](app.py): Streamlit entrypoint
- [src/unified_xai/ui/app.py](src/unified_xai/ui/app.py): main interface
- [src/unified_xai/core/registry.py](src/unified_xai/core/registry.py): model registry and compatibility metadata
- [src/unified_xai/services](src/unified_xai/services): inference and XAI orchestration
- [scripts](scripts): local training, evaluation, and verification scripts
- [tests](tests): smoke and demo-safety tests

For the full map, see [project-structure.md](docs/project-structure.md).

## Git Repository Notes

This Git repository is intentionally lighter than the local development workspace. The following local-only resources are excluded from version control:

- downloaded datasets under `train_datasets/`
- the two original downloaded source repositories
- TensorFlow SavedModel weight files and other large generated artifacts
- internal demo-only notes

To keep the repository runnable after clone, one audio checkpoint and one image checkpoint are included:

- `artifacts/audio_models/deepfake_melspec_cnn`
- `artifacts/image_models/jsrt_densenet121`

The larger local reconstruction checkpoints such as `VGG16`, `Custom CNN`, and `jsrt_alexnet` stay local and are not required to test the main demo path.

To reproduce the full local setup after cloning, you need to add the datasets and model artifacts locally or regenerate them with the training/export scripts.

## Setup And Installation

### Requirements

- Windows machine
- `uv` installed
- Python resolved by `uv`

### Install Dependencies

From the project root:

```bash
uv sync
```

## Run The Interface

```bash
uv run streamlit run app.py --server.port 8503
```

If that port is already taken, use another free port.

## Demo Instructions

### Audio Demo

1. Start the app with `uv run streamlit run app.py --server.port 8503`.
2. Choose `Audio`.
3. Select `MobileNet-style classifier (packaged)`.
4. Pick one of these samples:
   - [demo_real_packaged_correct.wav](sample_inputs/audio/demo_real_packaged_correct.wav)
   - [demo_fake_packaged_correct.wav](sample_inputs/audio/demo_fake_packaged_correct.wav)
5. In `Single Analysis`, choose `Grad-CAM` or `LIME`.
6. Run the analysis.
7. For the comparison part of the demo, switch to `Compare XAI` and select at least two explainers.

### Chest X-ray Demo

1. Choose `Chest X-ray`.
2. Select `DenseNet121 (JSRT)`.
3. Pick one of these samples:
   - [jsrt_malignant_jpcln047.png](sample_inputs/images/jsrt_malignant_jpcln047.png)
   - [jsrt_benign_jpcln036.png](sample_inputs/images/jsrt_benign_jpcln036.png)
   - [jsrt_non_nodule_jpcnn086.png](sample_inputs/images/jsrt_non_nodule_jpcnn086.png)
4. In `Single Analysis`, choose `Grad-CAM`.
5. Run the analysis.
6. For the comparison part of the demo, switch to `Compare XAI` and select `Grad-CAM`, `LIME`, and optionally `SHAP`.

## Verification

### Test Suite

```bash
uv run pytest
```

The tests cover:

- registry/model availability
- audio inference
- audio `Grad-CAM`
- audio `LIME`
- chest X-ray inference
- chest X-ray `Grad-CAM`
- curated demo sample checks for the recommended audio and image demo paths

### Full XAI Verification

`SHAP` is slower than the rest, so there is a separate verification script for all required explainers:

```bash
uv run python scripts/verify_xai_stack.py
```

This writes local outputs under `artifacts/verification/`, including `xai_stack_verification.json` and the generated explanation overlays.

### Audio Evaluation

```bash
uv run python scripts/evaluate_audio_models.py
```

This writes local outputs under `artifacts/evaluation/`, including `audio_model_evaluation_summary.json` and `audio_model_predictions.csv`.

## Training Scripts

### Audio Checkpoints

```bash
uv run python scripts/train_audio_models.py
```

Important limitation: the locally reconstructed `VGG16` and `Custom CNN` artifacts were generated from a small demo-oriented subset workflow, not a complete final FoR training protocol. The packaged original checkpoint is therefore treated as the primary audio demo model in this repo.

### Chest X-ray Checkpoints

```bash
uv run python scripts/train_image_models.py
```

To train only the AlexNet-style checkpoint:

```bash
uv run python scripts/train_image_models.py --models alexnet --epochs-alexnet 16
```

## Documentation

- [project-structure.md](docs/project-structure.md)
- [technical-report.md](docs/technical-report.md)
- [pipeline-overview.md](docs/pipeline-overview.md)
- [pipeline-technical.md](docs/pipeline-technical.md)

## Generative AI Usage Statement

This project made extensive use of Generative AI during development.

- Tool/model used:
  - `GPT-5.4 Codex`
- Scope of use:
  - project planning and task breakdown
  - code generation and refactoring
  - reconstruction of missing code paths from README-only source material
  - debugging and implementation iteration
  - documentation drafting and cleanup
  - training, evaluation, and verification script support
- Working method:
  - `GPT-5.4 Codex` was used as an assistant throughout the project
  - I remained the main orchestrator of the project and decided what needed to be built, changed, or kept
  - Codex was used heavily across the different stages of the project to accelerate implementation, propose code, and help structure the work
  - all important code changes, design choices, and outputs were reviewed by me before being accepted
- Personal learning rationale:
  - this project was also an opportunity to learn how to use modern AI development tools in a practical and responsible way
  - these tools are not available in my company's environment, and I had never used Codex before this project
  - Given the AI situation, I took this opportunity to learn how to use these tools and adopt the best practices, it is an important part of my personal and professional training

## Limitations

- The original lung repository did not ship runnable inference code, so the chest X-ray pipeline is a local reconstruction from its README and the available JSRT replacement dataset.
- The local JSRT replacement dataset is much smaller than CheXpert, so the chest X-ray metrics are demo-usable but not clinically strong.
- The local audio reconstructions are weaker than the packaged original checkpoint.
- `SHAP` is significantly slower than `Grad-CAM` and `LIME` in a live demo.

