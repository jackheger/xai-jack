Curated chest X-ray demo files live in this folder so the Streamlit UI and smoke tests do not depend on the saved test split.

Current bundled files:

- `jsrt_malignant_jpcln047.png`: malignant case jointly classified correctly by the current DenseNet and AlexNet checkpoints
- `jsrt_benign_jpcln036.png`: benign case jointly classified correctly by the current DenseNet and AlexNet checkpoints
- `jsrt_non_nodule_jpcnn086.png`: non-nodule case jointly classified correctly by the current DenseNet and AlexNet checkpoints

The recommended live-demo model for these files is `jsrt_densenet121`.

See [manifest.csv](../../sample_inputs/images/manifest.csv) for the source study IDs and model-confidence snapshot used when these demo files were selected.

You can still add more local images here if you want custom demo samples in the UI.

