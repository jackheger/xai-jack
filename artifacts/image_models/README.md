Place exported chest X-ray TensorFlow SavedModels in this directory.

Expected paths for the current image prototype:

- `artifacts/image_models/jsrt_densenet121`
- `artifacts/image_models/jsrt_alexnet`

Those artifacts are produced by:

- `uv run python scripts/train_image_models.py`
- `uv run python scripts/train_image_models.py --models alexnet --epochs-alexnet 16`
