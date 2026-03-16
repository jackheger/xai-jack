Place additional TensorFlow SavedModel directories here when you export more audio checkpoints.

Expected paths:
- artifacts/audio_models/inception_v3
- artifacts/audio_models/vgg16
- artifacts/audio_models/custom_cnn
- artifacts/audio_models/resnet50

Each model directory should contain a TensorFlow SavedModel, for example:
- saved_model.pb
- variables/
