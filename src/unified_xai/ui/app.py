from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

from unified_xai.core.registry import (
    get_audio_model,
    get_image_model,
    list_audio_models,
    list_available_audio_models,
    list_available_image_models,
    list_image_models,
    list_unavailable_audio_models,
    list_unavailable_image_models,
)
from unified_xai.modalities.audio.preprocess import infer_audio_format
from unified_xai.services.audio_service import run_audio_inference
from unified_xai.services.explanation_service import (
    get_audio_explainer,
    list_available_audio_explainers,
    run_audio_explanation,
)
from unified_xai.services.image_explanation_service import (
    get_image_explainer,
    list_available_image_explainers,
    run_image_explanation,
)
from unified_xai.services.image_service import run_image_inference


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_AUDIO_DIR = PROJECT_ROOT / "sample_inputs" / "audio"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "sample_inputs" / "images"
IMAGE_TEST_SPLIT_PATH = (
    PROJECT_ROOT / "artifacts" / "image_models" / "jsrt_densenet121" / "test_split.csv"
)
RECOMMENDED_AUDIO_MODEL_ID = "deepfake_melspec_cnn"
RECOMMENDED_IMAGE_MODEL_ID = "jsrt_densenet121"
RECOMMENDED_AUDIO_SAMPLE_NAMES = (
    "demo_fake_packaged_correct.wav",
    "demo_real_packaged_correct.wav",
)
RECOMMENDED_IMAGE_SAMPLE_NAMES = (
    "jsrt_malignant_jpcln047.png",
    "jsrt_benign_jpcln036.png",
    "jsrt_non_nodule_jpcnn086.png",
)


def _read_selected_audio(uploaded_file, sample_file: str) -> tuple[bytes | None, str | None]:
    if uploaded_file is not None:
        return uploaded_file.read(), uploaded_file.name

    if sample_file:
        sample_path = SAMPLE_AUDIO_DIR / sample_file
        if sample_path.exists():
            return sample_path.read_bytes(), sample_path.name

    return None, None


def _list_image_sample_paths(limit: int = 20) -> dict[str, Path]:
    sample_paths: dict[str, Path] = {}

    if SAMPLE_IMAGE_DIR.exists():
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            for path in sorted(SAMPLE_IMAGE_DIR.glob(suffix)):
                sample_paths[path.name] = path
        if sample_paths:
            return sample_paths

    if IMAGE_TEST_SPLIT_PATH.exists():
        with IMAGE_TEST_SPLIT_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                path = Path(row["image_path"])
                if path.exists():
                    sample_paths[path.name] = path
                if len(sample_paths) >= limit:
                    break

    return sample_paths


def _read_selected_image(
    uploaded_file,
    selected_sample: str,
    sample_paths: dict[str, Path],
) -> tuple[bytes | None, str | None]:
    if uploaded_file is not None:
        return uploaded_file.read(), uploaded_file.name

    if selected_sample:
        sample_path = sample_paths.get(selected_sample)
        if sample_path and sample_path.exists():
            return sample_path.read_bytes(), sample_path.name

    return None, None


def _probability_rows(probabilities: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {"label": label, "probability": probability}
        for label, probability in probabilities.items()
    ]


def _recommended_first(names: list[str], recommended_names: tuple[str, ...]) -> list[str]:
    recommended = [name for name in recommended_names if name in names]
    remaining = sorted(name for name in names if name not in recommended_names)
    return recommended + remaining


def _format_explainer_ids(explainer_ids: tuple[str, ...] | list[str]) -> str:
    display_names = {
        "lime": "LIME",
        "gradcam": "Grad-CAM",
        "shap": "SHAP",
    }
    return ", ".join(display_names.get(explainer_id, explainer_id) for explainer_id in explainer_ids)


def _format_available_explainers(explainers: list) -> str:
    if not explainers:
        return "None yet"
    return ", ".join(explainer.display_name for explainer in explainers)


def _format_audio_model_option(model_id: str) -> str:
    spec = get_audio_model(model_id)
    if model_id == RECOMMENDED_AUDIO_MODEL_ID:
        return f"{spec.display_name} (Recommended demo)"
    return spec.display_name


def _format_image_model_option(model_id: str) -> str:
    spec = get_image_model(model_id)
    if model_id == RECOMMENDED_IMAGE_MODEL_ID:
        return f"{spec.display_name} (Recommended demo)"
    return spec.display_name


def _render_probability_table(probabilities: dict[str, float]) -> None:
    st.dataframe(
        _probability_rows(probabilities),
        use_container_width=True,
        column_config={
            "label": "Label",
            "probability": st.column_config.ProgressColumn(
                "Probability",
                format="%.4f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
        hide_index=True,
    )
    st.caption(
        "These values are model scores for the selected checkpoint. They help compare labels, "
        "but they should not be interpreted as calibrated certainty."
    )


def _render_status_table(rows: list[dict[str, str]]) -> None:
    ordered_rows = sorted(
        rows,
        key=lambda row: (0 if row["status"] == "Installed" else 1, row["model"].lower()),
    )

    def _format_model(model: str, status: str) -> str:
        if status == "Installed":
            return f"<span style='color:#15803d;font-weight:600'>{model}</span>"
        return model

    def _format_status(status: str) -> str:
        if status == "Installed":
            return "<span style='color:#15803d;font-weight:600'>Installed</span>"
        return "<span style='color:#b91c1c;font-weight:600'>Missing checkpoint</span>"

    def _format_path(path: str, status: str) -> str:
        if status == "Installed":
            return f"<code style='color:#15803d'>{path}</code>"
        return f"<code style='color:#6b7280'>{path}</code>"

    html_rows = "".join(
        (
            "<tr>"
            f"<td>{_format_model(row['model'], row['status'])}</td>"
            f"<td>{_format_status(row['status'])}</td>"
            f"<td>{_format_path(row['expected_path'], row['status'])}</td>"
            "</tr>"
        )
        for row in ordered_rows
    )
    st.markdown(
        (
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left;border-bottom:1px solid #d1d5db;padding:0.4rem'>Model</th>"
            "<th style='text-align:left;border-bottom:1px solid #d1d5db;padding:0.4rem'>Status</th>"
            "<th style='text-align:left;border-bottom:1px solid #d1d5db;padding:0.4rem'>Expected path</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{html_rows}</tbody>"
            "</table>"
        ),
        unsafe_allow_html=True,
    )


def _render_explanation_details(explanation) -> None:
    st.write(explanation.summary)
    st.write(f"Target label: `{explanation.details['target_label']}`")
    if "target_layer" in explanation.details:
        st.write(f"Grad-CAM layer: `{explanation.details['target_layer']}`")
    if "num_samples" in explanation.details:
        st.write(f"Perturbation samples: `{explanation.details['num_samples']}`")
    if "num_features" in explanation.details:
        st.write(f"Highlighted segments: `{explanation.details['num_features']}`")
    if "max_evals" in explanation.details:
        st.write(f"SHAP evaluations: `{explanation.details['max_evals']}`")


def _chunked(items: list, chunk_size: int) -> list[list]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def _render_explanation_grid(explanations: list) -> None:
    for row in _chunked(explanations, 3):
        columns = st.columns(len(row))
        for column, explanation in zip(columns, row):
            with column:
                st.subheader(explanation.display_name)
                st.image(
                    explanation.visualization,
                    caption=(
                        f"{explanation.display_name} overlay for the predicted label "
                        f"`{explanation.details['target_label']}`"
                    ),
                )
                _render_explanation_details(explanation)


def _render_failures(failures: list[tuple[str, Exception]]) -> None:
    if not failures:
        return

    with st.expander("Explanation Failures"):
        for display_name, exception in failures:
            st.write(f"`{display_name}` failed: `{exception}`")


def _run_audio_explanations_batch(
    model_id: str,
    spectrogram,
    predicted_label: str,
    explainer_ids: list[str],
) -> tuple[list, list[tuple[str, Exception]]]:
    explanations = []
    failures: list[tuple[str, Exception]] = []
    for explainer_id in explainer_ids:
        display_name = get_audio_explainer(explainer_id).display_name
        try:
            explanations.append(
                run_audio_explanation(
                    model_id=model_id,
                    explainer_id=explainer_id,
                    spectrogram=spectrogram,
                    predicted_label=predicted_label,
                )
            )
        except Exception as exc:  # pragma: no cover - UI guardrail
            failures.append((display_name, exc))
    return explanations, failures


def _run_image_explanations_batch(
    model_id: str,
    image,
    predicted_label: str,
    explainer_ids: list[str],
) -> tuple[list, list[tuple[str, Exception]]]:
    explanations = []
    failures: list[tuple[str, Exception]] = []
    for explainer_id in explainer_ids:
        display_name = get_image_explainer(explainer_id).display_name
        try:
            explanations.append(
                run_image_explanation(
                    model_id=model_id,
                    explainer_id=explainer_id,
                    image=image,
                    predicted_label=predicted_label,
                )
            )
        except Exception as exc:  # pragma: no cover - UI guardrail
            failures.append((display_name, exc))
    return explanations, failures


def _render_audio_prediction_and_reference(result, spec, audio_name: str) -> None:
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Prediction")
        st.write(f"Model: `{spec.display_name}`")
        st.write(f"File: `{audio_name}`")
        st.metric("Predicted label", result.prediction.label)
        st.metric("Duration", f"{result.audio_metadata['duration_seconds']:.2f} s")
        st.metric("Sample rate", f"{int(result.audio_metadata['sample_rate'])} Hz")
        st.metric("Input format", str(result.audio_metadata["source_format"]).upper())
        _render_probability_table(result.prediction.probabilities)
        st.write(
            "Raw model output: "
            + ", ".join(f"{value:.6f}" for value in result.prediction.raw_output)
        )

    with right_col:
        st.subheader("Generated Spectrogram")
        st.image(result.spectrogram, caption=f"Mel spectrogram for {audio_name}")
        if result.audio_metadata["converted_to_wav"]:
            st.warning(str(result.audio_metadata["conversion_warning"]))
            st.subheader("Converted WAV Preview")
            st.audio(result.audio_metadata["converted_wav_bytes"], format="audio/wav")


def _render_image_prediction_and_reference(result, spec, image_name: str) -> None:
    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.subheader("Prediction")
        st.write(f"Model: `{spec.display_name}`")
        st.write(f"File: `{image_name}`")
        st.metric("Predicted label", result.prediction.label)
        st.metric("Original width", str(result.image_metadata["original_width"]))
        st.metric("Original height", str(result.image_metadata["original_height"]))
        st.metric("Input format", str(result.image_metadata["source_format"]).upper())
        _render_probability_table(result.prediction.probabilities)
        st.write(
            "Raw model output: "
            + ", ".join(f"{value:.6f}" for value in result.prediction.raw_output)
        )

    with right_col:
        st.subheader("Processed X-ray")
        st.image(result.image.astype("uint8"), caption=f"Processed chest X-ray for {image_name}")


def _render_audio_single_tab(spec, selected_model_id: str, audio_bytes: bytes, audio_name: str, available_explainers: list) -> None:
    st.subheader("Audio XAI")
    selected_explainer_id = st.selectbox(
        "Explanation method",
        options=[""] + [explainer.explainer_id for explainer in available_explainers],
        format_func=lambda explainer_id: (
            "None" if explainer_id == "" else get_audio_explainer(explainer_id).display_name
        ),
        disabled=not spec.is_available,
        key="audio_single_explainer_select",
    )

    if selected_explainer_id:
        st.info(get_audio_explainer(selected_explainer_id).description)
    else:
        st.caption("Choose an XAI method if you want an explanation in addition to the prediction.")

    if not available_explainers and spec.is_available:
        st.warning("This model is installed, but no explainers are implemented for it yet in the unified app.")

    if st.button("Run audio analysis", type="primary", disabled=not spec.is_available, key="audio_single_run"):
        try:
            with st.spinner("Running audio inference..."):
                result = run_audio_inference(selected_model_id, audio_bytes, source_name=audio_name)
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error("Audio inference failed.")
            st.exception(exc)
            return

        _render_audio_prediction_and_reference(result, spec, audio_name)

        if selected_explainer_id:
            explanations, failures = _run_audio_explanations_batch(
                model_id=selected_model_id,
                spectrogram=result.spectrogram,
                predicted_label=result.prediction.label,
                explainer_ids=[selected_explainer_id],
            )
            if explanations:
                explanation = explanations[0]
                exp_left, exp_right = st.columns([1, 1])
                with exp_left:
                    st.subheader(f"{explanation.display_name} Explanation")
                    st.image(
                        explanation.visualization,
                        caption=(
                            f"{explanation.display_name} overlay for the predicted label "
                            f"`{explanation.details['target_label']}`"
                        ),
                    )
                with exp_right:
                    st.subheader("Explanation Details")
                    _render_explanation_details(explanation)
            _render_failures(failures)

        st.success("Audio analysis completed.")


def _render_audio_compare_tab(spec, selected_model_id: str, audio_bytes: bytes, audio_name: str, available_explainers: list) -> None:
    st.subheader("Compare Audio XAI")
    selected_explainer_ids = st.multiselect(
        "Explanation methods to compare",
        options=[explainer.explainer_id for explainer in available_explainers],
        default=[explainer.explainer_id for explainer in available_explainers[:2]],
        format_func=lambda explainer_id: get_audio_explainer(explainer_id).display_name,
        disabled=not spec.is_available,
        key="audio_compare_explainer_select",
    )
    st.caption(
        "Select at least two explainers to generate a side-by-side comparison. "
        "SHAP is supported but slower than Grad-CAM and LIME."
    )

    if st.button("Run audio comparison", type="primary", disabled=not spec.is_available, key="audio_compare_run"):
        if len(selected_explainer_ids) < 2:
            st.warning("Choose at least two explainers to use the comparison view.")
            return

        try:
            with st.spinner("Running audio inference..."):
                result = run_audio_inference(selected_model_id, audio_bytes, source_name=audio_name)
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error("Audio inference failed.")
            st.exception(exc)
            return

        with st.spinner("Generating selected audio explanations..."):
            explanations, failures = _run_audio_explanations_batch(
                model_id=selected_model_id,
                spectrogram=result.spectrogram,
                predicted_label=result.prediction.label,
                explainer_ids=selected_explainer_ids,
            )

        _render_audio_prediction_and_reference(result, spec, audio_name)

        if explanations:
            st.subheader("Audio XAI Comparison")
            _render_explanation_grid(explanations)
        _render_failures(failures)

        if explanations:
            st.success("Audio comparison completed.")


def _render_image_single_tab(spec, selected_model_id: str, image_bytes: bytes, image_name: str, available_explainers: list) -> None:
    st.subheader("Image XAI")
    selected_explainer_id = st.selectbox(
        "Explanation method",
        options=[""] + [explainer.explainer_id for explainer in available_explainers],
        format_func=lambda explainer_id: (
            "None" if explainer_id == "" else get_image_explainer(explainer_id).display_name
        ),
        disabled=not spec.is_available,
        key="image_single_explainer_select",
    )

    if selected_explainer_id:
        st.info(get_image_explainer(selected_explainer_id).description)
    else:
        st.caption("Choose an image explainer if you want Grad-CAM, LIME, or SHAP in addition to the prediction.")

    if st.button("Run chest X-ray analysis", type="primary", disabled=not spec.is_available, key="image_single_run"):
        try:
            with st.spinner("Running chest X-ray inference..."):
                result = run_image_inference(selected_model_id, image_bytes, source_name=image_name)
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error("Image inference failed.")
            st.exception(exc)
            return

        _render_image_prediction_and_reference(result, spec, image_name)

        if selected_explainer_id:
            explanations, failures = _run_image_explanations_batch(
                model_id=selected_model_id,
                image=result.image,
                predicted_label=result.prediction.label,
                explainer_ids=[selected_explainer_id],
            )
            if explanations:
                explanation = explanations[0]
                exp_left, exp_right = st.columns([1, 1])
                with exp_left:
                    st.subheader(f"{explanation.display_name} Explanation")
                    st.image(
                        explanation.visualization,
                        caption=(
                            f"{explanation.display_name} overlay for the predicted label "
                            f"`{explanation.details['target_label']}`"
                        ),
                    )
                with exp_right:
                    st.subheader("Explanation Details")
                    _render_explanation_details(explanation)
            _render_failures(failures)

        st.success("Chest X-ray analysis completed.")


def _render_image_compare_tab(spec, selected_model_id: str, image_bytes: bytes, image_name: str, available_explainers: list) -> None:
    st.subheader("Compare Image XAI")
    selected_explainer_ids = st.multiselect(
        "Explanation methods to compare",
        options=[explainer.explainer_id for explainer in available_explainers],
        default=[explainer.explainer_id for explainer in available_explainers[:2]],
        format_func=lambda explainer_id: get_image_explainer(explainer_id).display_name,
        disabled=not spec.is_available,
        key="image_compare_explainer_select",
    )
    st.caption(
        "Select at least two explainers to generate a side-by-side comparison. "
        "SHAP is supported but slower than Grad-CAM and LIME."
    )

    if st.button("Run chest X-ray comparison", type="primary", disabled=not spec.is_available, key="image_compare_run"):
        if len(selected_explainer_ids) < 2:
            st.warning("Choose at least two explainers to use the comparison view.")
            return

        try:
            with st.spinner("Running chest X-ray inference..."):
                result = run_image_inference(selected_model_id, image_bytes, source_name=image_name)
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error("Image inference failed.")
            st.exception(exc)
            return

        with st.spinner("Generating selected chest X-ray explanations..."):
            explanations, failures = _run_image_explanations_batch(
                model_id=selected_model_id,
                image=result.image,
                predicted_label=result.prediction.label,
                explainer_ids=selected_explainer_ids,
            )

        _render_image_prediction_and_reference(result, spec, image_name)

        if explanations:
            st.subheader("Chest X-ray XAI Comparison")
            _render_explanation_grid(explanations)
        _render_failures(failures)

        if explanations:
            st.success("Chest X-ray comparison completed.")


def _render_audio_view() -> None:
    st.caption(
        "Audio prototype with model selection, MP3 preprocessing, single-explainer analysis, and XAI comparison."
    )

    model_specs = list_audio_models()
    available_specs = list_available_audio_models()

    if not available_specs:
        st.error("No audio model checkpoints are installed yet.")
        st.info(
            "Add a TensorFlow SavedModel under `artifacts/audio_models/` or restore the packaged model path."
        )
        return

    selected_model_id = st.selectbox(
        "Audio model",
        options=[spec.model_id for spec in available_specs],
        format_func=_format_audio_model_option,
        key="audio_model_select",
    )
    spec = get_audio_model(selected_model_id)
    available_explainers = list_available_audio_explainers(selected_model_id)

    with st.sidebar:
        st.subheader("Selected Audio Model")
        st.write(spec.description)
        st.write(f"Labels: {', '.join(spec.labels)}")
        st.write(f"Advertised explainers: {_format_explainer_ids(spec.supported_explainers)}")
        st.write(f"Implemented in unified app: {_format_available_explainers(available_explainers)}")
        st.write(f"Status: {'Installed' if spec.is_available else 'Missing checkpoint'}")
        st.write(f"Artifact path: `{spec.relative_artifact_path}`")
        st.write(
            "Recommended for live demo: "
            + ("Yes" if selected_model_id == RECOMMENDED_AUDIO_MODEL_ID else "No")
        )
        st.write(spec.source_note)

    st.subheader("Audio Model Availability")
    st.caption("Only installed audio checkpoints appear in the selector above.")
    _render_status_table(
        [
            {
                "model": model_spec.display_name,
                "status": "Installed" if model_spec.is_available else "Missing checkpoint",
                "expected_path": model_spec.relative_artifact_path,
            }
            for model_spec in model_specs
        ]
    )

    if unavailable_audio_models := list_unavailable_audio_models():
        with st.expander("Missing Audio Checkpoints"):
            st.write(
                "These models are listed for completeness but cannot be selected until their "
                "SavedModel artifacts are added."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": missing.display_name,
                            "expected_path": missing.relative_artifact_path,
                        }
                        for missing in unavailable_audio_models
                    ]
                ),
                use_container_width=True,
            )

    if selected_model_id == RECOMMENDED_AUDIO_MODEL_ID:
        st.info(
            "Recommended audio demo model selected. This packaged checkpoint currently performs best "
            "on the local evaluation subset."
        )
    else:
        st.caption(
            "For the safest live demo, prefer `MobileNet-style classifier (packaged)` "
            "because it outperforms the local audio reconstructions in the current evaluation."
        )

    st.subheader("Audio Input")
    uploaded_file = st.file_uploader(
        "Upload a .wav or .mp3 file",
        type=["wav", "mp3"],
        key="audio_uploader",
    )
    sample_files = _recommended_first(
        [path.name for path in SAMPLE_AUDIO_DIR.glob("*.wav")],
        RECOMMENDED_AUDIO_SAMPLE_NAMES,
    )
    selected_sample = st.selectbox(
        "Or choose a sample from sample_inputs/audio",
        options=[""] + sample_files,
        format_func=lambda value: "None" if value == "" else value,
        key="audio_sample_select",
    )

    if uploaded_file is not None and selected_sample:
        st.warning("Uploaded file takes priority over the selected sample.")

    audio_bytes, audio_name = _read_selected_audio(uploaded_file, selected_sample)
    if audio_bytes is None:
        st.info("Upload a `.wav` or `.mp3` file, or place sample WAV files in `sample_inputs/audio/`.")
        return

    st.info(
        "Audio explainers operate on the generated mel spectrogram image. "
        "The waveform is first transformed into a 2D representation, then the selected XAI method highlights "
        "important spectrogram regions for the chosen model."
    )
    st.caption(
        "Compatible explainers for the selected model: "
        f"{_format_available_explainers(available_explainers)}. "
        "The Compare XAI tab only shows methods that are both implemented and compatible with this checkpoint."
    )
    st.caption(
        "Recommended audio demo samples: `demo_fake_packaged_correct.wav` and "
        "`demo_real_packaged_correct.wav`."
    )

    source_format = infer_audio_format(audio_name)
    st.subheader("Preprocessing")
    if source_format == "mp3":
        toast_key = f"mp3-conversion-toast::{audio_name}"
        if not st.session_state.get(toast_key):
            st.toast(
                "MP3 detected. The app will convert it to WAV/PCM before inference. Results may be less accurate."
            )
            st.session_state[toast_key] = True
        st.warning(
            "This file will be converted from MP3 to WAV/PCM before classification. "
            "Because MP3 is lossy, the prediction may be less reliable than a native WAV input."
        )
        st.markdown(
            "**Pipeline:** uploaded MP3 -> decode to mono waveform -> convert to WAV/PCM -> "
            "generate mel spectrogram -> model inference"
        )
        st.audio(audio_bytes, format="audio/mpeg")
    else:
        st.info("No format conversion is needed for WAV input.")
        st.markdown(
            "**Pipeline:** uploaded WAV -> decode to mono waveform -> generate mel spectrogram -> "
            "model inference"
        )
        st.audio(audio_bytes, format="audio/wav")

    single_tab, compare_tab = st.tabs(["Single Analysis", "Compare XAI"])
    with single_tab:
        _render_audio_single_tab(spec, selected_model_id, audio_bytes, audio_name, available_explainers)
    with compare_tab:
        _render_audio_compare_tab(spec, selected_model_id, audio_bytes, audio_name, available_explainers)


def _render_image_view() -> None:
    st.caption(
        "Chest X-ray prototype using the local JSRT replacement dataset, multiple image checkpoints, single-explainer analysis, and XAI comparison."
    )

    model_specs = list_image_models()
    available_specs = list_available_image_models()
    unavailable_specs = list_unavailable_image_models()

    if not available_specs:
        st.error("No image model checkpoints are installed yet.")
        st.info("Train and export one with `uv run python scripts/train_image_models.py`.")
        return

    selected_model_id = st.selectbox(
        "Chest X-ray model",
        options=[spec.model_id for spec in available_specs],
        format_func=_format_image_model_option,
        key="image_model_select",
    )
    spec = get_image_model(selected_model_id)
    available_explainers = list_available_image_explainers(selected_model_id)

    with st.sidebar:
        st.subheader("Selected Image Model")
        st.write(spec.description)
        st.write(f"Labels: {', '.join(spec.labels)}")
        st.write(f"Advertised explainers: {_format_explainer_ids(spec.supported_explainers)}")
        st.write(f"Implemented in unified app: {_format_available_explainers(available_explainers)}")
        st.write(f"Status: {'Installed' if spec.is_available else 'Missing checkpoint'}")
        st.write(f"Artifact path: `{spec.relative_artifact_path}`")
        st.write(
            "Recommended for live demo: "
            + ("Yes" if selected_model_id == RECOMMENDED_IMAGE_MODEL_ID else "No")
        )
        st.write(spec.source_note)

    if unavailable_specs:
        with st.expander("Missing Chest X-ray Checkpoints"):
            st.write(
                "These models are listed for completeness but cannot be selected until their "
                "SavedModel artifacts are added."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "model": missing.display_name,
                            "expected_path": missing.relative_artifact_path,
                        }
                        for missing in unavailable_specs
                    ]
                ),
                use_container_width=True,
            )

    st.subheader("Chest X-ray Model Availability")
    st.caption("Only installed chest X-ray checkpoints appear in the selector above.")
    _render_status_table(
        [
            {
                "model": model_spec.display_name,
                "status": "Installed" if model_spec.is_available else "Missing checkpoint",
                "expected_path": model_spec.relative_artifact_path,
            }
            for model_spec in model_specs
        ]
    )

    if selected_model_id == RECOMMENDED_IMAGE_MODEL_ID:
        st.info(
            "Recommended chest X-ray demo model selected. This DenseNet checkpoint is the strongest "
            "current balance of malignant-case recall and F1."
        )
    else:
        st.caption(
            "For the safest live demo, prefer `DenseNet121 (JSRT)` over the AlexNet-style checkpoint."
        )

    st.subheader("Chest X-ray Input")
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        key="image_uploader",
    )
    sample_paths = _list_image_sample_paths()
    selected_sample = st.selectbox(
        "Or choose a sample image",
        options=[""] + _recommended_first(list(sample_paths.keys()), RECOMMENDED_IMAGE_SAMPLE_NAMES),
        format_func=lambda value: "None" if value == "" else value,
        key="image_sample_select",
    )

    if uploaded_file is not None and selected_sample:
        st.warning("Uploaded image takes priority over the selected sample.")

    image_bytes, image_name = _read_selected_image(uploaded_file, selected_sample, sample_paths)
    if image_bytes is None:
        st.info(
            "Upload a `.png`, `.jpg`, or `.jpeg` chest X-ray. "
            "If the JSRT model artifact exists, a few dataset samples are also listed here."
        )
        return

    st.info(
        "Chest X-ray explainers operate on the processed X-ray image shown below. "
        "Grad-CAM is model-specific, while LIME and SHAP are model-agnostic image explainers."
    )
    st.caption(
        "Compatible explainers for the selected model: "
        f"{_format_available_explainers(available_explainers)}. "
        "The Compare XAI tab only shows methods that are both implemented and compatible with this checkpoint."
    )
    st.caption(
        "Recommended chest X-ray demo samples: `jsrt_malignant_jpcln047.png`, "
        "`jsrt_benign_jpcln036.png`, and `jsrt_non_nodule_jpcnn086.png`."
    )

    single_tab, compare_tab = st.tabs(["Single Analysis", "Compare XAI"])
    with single_tab:
        _render_image_single_tab(spec, selected_model_id, image_bytes, image_name, available_explainers)
    with compare_tab:
        _render_image_compare_tab(spec, selected_model_id, image_bytes, image_name, available_explainers)


def main() -> None:
    st.set_page_config(page_title="Unified XAI Interface", layout="wide")

    st.title("Unified Explainable AI Interface")
    modality = st.radio(
        "Dataset / modality",
        options=["Audio", "Chest X-ray"],
        horizontal=True,
    )

    if modality == "Audio":
        _render_audio_view()
    else:
        _render_image_view()
