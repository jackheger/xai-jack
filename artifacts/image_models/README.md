This directory contains chest X-ray TensorFlow SavedModels used by the unified app.

Bundled in the repository:

- `artifacts/image_models/jsrt_densenet121`

Optional local export paths for additional checkpoints:

- `artifacts/image_models/jsrt_alexnet`

Those artifacts are produced by:

- `uv run python scripts/train_image_models.py`
- `uv run python scripts/train_image_models.py --models alexnet --epochs-alexnet 16`
