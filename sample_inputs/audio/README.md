Curated audio demo files live in this folder so the Streamlit UI and the demo workflow do not depend only on ad hoc uploads.

Recommended demo files:

- `demo_real_packaged_correct.wav`: real sample correctly classified by the packaged audio checkpoint
- `demo_fake_packaged_correct.wav`: fake sample correctly classified by the packaged audio checkpoint

Other utility files:

- `file_example_WAV_1MG.wav`: generic WAV sample
- `file_example_WAV_1MG.mp3`: generic MP3 sample for the conversion path
- `converted_from_mp3.wav`: example conversion output

The recommended live-demo model for these files is `deepfake_melspec_cnn`.

See [manifest.csv](../../sample_inputs/audio/manifest.csv) for the true labels and packaged-model confidence snapshot used when these demo files were selected.

