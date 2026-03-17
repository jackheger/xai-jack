This directory contains audio TensorFlow SavedModels used by the unified app.

Bundled in the repository:

- `artifacts/audio_models/deepfake_melspec_cnn`

Optional local export paths for additional checkpoints:

- `artifacts/audio_models/inception_v3`
- `artifacts/audio_models/vgg16`
- `artifacts/audio_models/custom_cnn`
- `artifacts/audio_models/resnet50`

Each model directory should contain a TensorFlow SavedModel, for example:

- `saved_model.pb`
- `variables/`
