"""Handwritten Text Recognition - Streamlit application.

Run with:
    streamlit run app.py
or
    python main.py app

Pages: Home, Handwriting Recognition, Model Comparison, Dataset Explorer,
       Model Information, About.

All predictions come from real trained checkpoints. When a checkpoint is
missing the page says so instead of showing fake output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments"
METRICS_DIR = RESULTS / "metrics"
FIG_DIR = RESULTS / "figures"
CKPT_DIR = ROOT / "checkpoints"

CRNN_CKPT = CKPT_DIR / "crnn" / "crnn_best.pt"
CRNN_VOCAB = CKPT_DIR / "crnn" / "vocab.json"
TROCR_DIR = CKPT_DIR / "trocr" / "final"

st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="✍️",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading IAM-line dataset...")
def get_dataset():
    from src.data.dataset import load_iam_splits

    return load_iam_splits()


@st.cache_resource(show_spinner="Loading CRNN checkpoint...")
def get_crnn(checkpoint: str, vocab: str | None):
    from src.inference.predict import CRNNInference

    return CRNNInference(checkpoint, vocab, device="auto")


@st.cache_resource(show_spinner="Loading TrOCR model...")
def get_trocr(model_dir: str):
    from src.inference.predict import TrOCRInference

    return TrOCRInference(model_dir, device="auto")


@st.cache_data(ttl=10)
def read_json(path: str):
    file = Path(path)
    if not file.is_file():
        return None
    with open(file, encoding="utf-8") as f:
        return json.load(f)


def available_models() -> list[str]:
    models = []
    if CRNN_CKPT.is_file():
        models.append("CRNN")
    if (TROCR_DIR / "config.json").is_file():
        models.append("TrOCR")
    return models


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home() -> None:
    st.title("✍️ Handwritten Text Recognition")
    st.caption("Line-level HTR on IAM with CRNN + BiLSTM + CTC and fine-tuned TrOCR")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
### What this system does
It reads a **complete handwritten English text line** from an image and returns
the transcription. This is sequence recognition, not character or word
classification: the model sees the whole line and predicts the whole sentence.

```
Input  :  [ handwritten line image ]
Output :  "This is an example of handwritten text recognition."
```

### The two models
| Model | Type | Decoding |
|-------|------|----------|
| **CRNN** | CNN → BiLSTM → CTC | greedy / prefix beam search |
| **TrOCR** | ViT encoder + Transformer decoder (pretrained) | beam search |

Both are trained on the same **Teklia/IAM-line** splits (train 6,482 /
validation 976 / test 2,915) and compared on identical test samples.
            """
        )
    with col2:
        st.subheader("Current status")
        models = available_models()
        if models:
            for model in models:
                st.success(f"{model} checkpoint available")
        else:
            st.warning("No trained checkpoints yet")
        st.info(
            "Dataset: Teklia/IAM-line\n\n"
            "Splits are the dataset's own writer-independent splits."
        )

    st.subheader("Quick start")
    st.code(
        "python src/train.py crnn --epochs 50\n"
        "python src/train.py trocr --epochs 10\n"
        "python src/evaluate.py\n"
        "python main.py app",
        language="bash",
    )


def page_recognition() -> None:
    st.title("Handwriting Recognition")
    st.write(
        "Upload a line image (PNG/JPG, ideally a single handwritten line on a "
        "light background) and run inference with a trained model."
    )

    models = available_models()
    if not models:
        st.error(
            "No trained checkpoints were found.\n\n"
            f"- CRNN: expected `{CRNN_CKPT.relative_to(ROOT)}`\n"
            f"- TrOCR: expected `{TROCR_DIR.relative_to(ROOT)}/config.json`\n\n"
            "Train a model first:\n"
            "```bash\npython scripts/train_crnn.py --epochs 50\n"
            "python scripts/train_trocr.py --epochs 10\n```"
        )
        return

    uploaded = st.file_uploader("Handwritten line image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    choice = st.selectbox("Model", models + (["Compare both"] if len(models) > 1 else []))

    if uploaded is not None:
        from PIL import Image

        image = Image.open(uploaded)
        st.image(image, caption=f"Uploaded image ({image.size[0]}×{image.size[1]} px)", use_container_width=True)

    run = st.button("Recognize text", type="primary", disabled=uploaded is None)
    if not run or uploaded is None:
        return

    from PIL import Image

    image = Image.open(uploaded)
    targets = models if choice == "Compare both" else [choice]
    for name in targets:
        try:
            with st.spinner(f"Running {name}..."):
                started = time.perf_counter()
                if name == "CRNN":
                    predictor = get_crnn(
                        str(CRNN_CKPT), str(CRNN_VOCAB) if CRNN_VOCAB.is_file() else None
                    )
                else:
                    predictor = get_trocr(str(TROCR_DIR))
                result = predictor.predict(image)
                elapsed = time.perf_counter() - started
            st.markdown(f"#### {name}")
            st.success(result.text if result.text else "(empty prediction)")
            st.caption(
                f"Inference: {result.inference_time * 1000:.1f} ms "
                f"(total incl. preprocessing: {elapsed * 1000:.1f} ms)"
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"{name} failed: {exc}")

    st.info(
        "Predictions are generated by the checkpoints on disk. Accuracy depends "
        "on how long each model was trained; see the Model Comparison page."
    )


def page_comparison() -> None:
    st.title("Model Comparison")
    comparison = read_json(str(METRICS_DIR / "comparison.json"))
    if not comparison:
        st.warning(
            "No evaluation results found. Run:\n\n"
            "```bash\npython scripts/evaluate.py\n```\n\n"
            "Results are stored in `experiments/metrics/`."
        )
        return

    st.caption(f"Evaluated on the IAM-line test split ({next(iter(comparison.values()))['num_samples']} samples)")

    columns = st.columns(len(comparison))
    for column, (name, values) in zip(columns, comparison.items()):
        column.metric(f"{name} CER", f"{values['cer'] * 100:.2f}%")
        column.metric(f"{name} WER", f"{values['wer'] * 100:.2f}%")
        column.metric(f"{name} inference", f"{values['avg_inference_time_per_line'] * 1000:.1f} ms/line")

    rows = []
    for name, values in comparison.items():
        rows.append(
            {
                "Model": name,
                "CER": round(values["cer"], 4),
                "WER": round(values["wer"], 4),
                "Best val loss": (
                    round(values["val_loss"], 4) if values.get("val_loss") is not None else "n/a"
                ),
                "Training time (s)": (
                    round(values["training_time_seconds"], 1)
                    if values.get("training_time_seconds")
                    else "n/a"
                ),
                "Avg inference (ms/line)": round(values["avg_inference_time_per_line"] * 1000, 2),
                "Samples": values["num_samples"],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    figures = [
        ("CER comparison", FIG_DIR / "cer_comparison.png"),
        ("WER comparison", FIG_DIR / "wer_comparison.png"),
        ("CER / WER grouped", FIG_DIR / "cer_wer_comparison.png"),
        ("Average inference time", FIG_DIR / "inference_time.png"),
        ("Training time", FIG_DIR / "training_time.png"),
        ("CRNN loss curves", FIG_DIR / "crnn_loss.png"),
        ("TrOCR loss curves", FIG_DIR / "trocr_loss.png"),
        ("Prediction examples", FIG_DIR / "prediction_examples.png"),
    ]
    existing = [(caption, path) for caption, path in figures if path.is_file()]
    if not existing:
        st.info("Evaluation figures have not been generated yet.")
        return

    st.subheader("Charts")
    for i in range(0, len(existing), 2):
        cols = st.columns(2)
        for col, (caption, path) in zip(cols, existing[i : i + 2]):
            col.image(str(path), caption=caption, use_container_width=True)

    per_model = [
        ("CRNN predictions", RESULTS / "predictions" / "crnn_predictions.txt"),
        ("TrOCR predictions", RESULTS / "predictions" / "trocr_predictions.txt"),
    ]
    for title, path in per_model:
        if path.is_file():
            with st.expander(title):
                st.text(path.read_text(encoding="utf-8")[:6000])


def page_dataset() -> None:
    st.title("Dataset Explorer")
    st.markdown(
        "**Teklia/IAM-line** — English handwritten text lines (128 px high, "
        "variable width) with transcriptions. The dataset already provides "
        "line-segmented images and writer-independent splits."
    )
    try:
        dataset = get_dataset()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the dataset: {exc}")
        return

    split_names = list(dataset.keys())
    split = st.selectbox("Split", split_names)
    data = dataset[split]

    lengths = [len(str(t)) for t in data["text"]]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{len(data):,}")
    col2.metric("Columns", ", ".join(data.column_names))
    col3.metric("Avg text length", f"{sum(lengths) / len(lengths):.1f}")
    col4.metric("Max text length", max(lengths))

    st.caption(
        f"Total across splits: {sum(len(dataset[s]) for s in split_names):,} samples "
        "(train 6,482 / validation 976 / test 2,915)."
    )

    index = st.slider("Sample index", 0, len(data) - 1, 0)
    sample = data[index]
    left, right = st.columns([2, 1])
    with left:
        st.image(sample["image"], caption=f"{split}[{index}]", use_container_width=True)
    with right:
        st.subheader("Transcription")
        st.write(sample["text"])
        st.caption(f"Image size: {sample['image'].size[0]}×{sample['image'].size[1]} px")

    st.subheader("More samples")
    cols = st.columns(3)
    for offset, col in enumerate(cols):
        idx = (index + offset * 37) % len(data)
        with col:
            st.image(dataset[split][idx]["image"], use_container_width=True)
            st.caption(str(dataset[split][idx]["text"])[:120])


def page_model_info() -> None:
    st.title("Model Information")
    st.markdown(
        """
### CRNN + BiLSTM + CTC
1. **CNN** — three conv/batch-norm/ReLU blocks with stride-2 max pooling.
   A 64×W line becomes a 256×(W/8) feature map, i.e. a sequence of
   `W/8` frames, each 256×8 features wide.
2. **BiLSTM** — two bidirectional LSTM layers (hidden 256) model left-to-right
   and right-to-left context across the line.
3. **Fully connected layer** — one score per character plus a CTC blank.
4. **CTC loss** — aligns the frame sequence with the transcription without
   needing per-character positions; blanks and repeated characters are handled
   by the CTC collapse rule. Decoding uses greedy best-path or prefix beam search.

Images keep their aspect ratio: the height is fixed at 64 px, the width scales,
and variable widths are padded per batch (with true lengths passed to CTC).

### TrOCR (Transformer OCR)
- **Encoder**: DeiT/ViT vision transformer on 384×384 inputs.
- **Decoder**: Transformer decoder (XLM-R vocabulary) generating tokens
  autoregressively, initialised from `microsoft/trocr-small-handwritten`.
- Fine-tuned on IAM-line with teacher forcing; padding labels are masked with
  `-100`, generation uses beam search.
- Pretrained weights are never trained from scratch.

### Metrics
- **CER** — character edit distance / reference characters (corpus level).
- **WER** — word edit distance / reference words (corpus level).
- **Inference time** — measured per line, model forward/generation only.

Both models share one normalisation (lowercase, punctuation removed, whitespace
collapsed) so the comparison is fair.
        """
    )


def page_about() -> None:
    st.title("About")
    st.markdown(
        """
**Project:** Handwritten Text Recognition using Deep Learning

A sequence-recognition system that transcribes complete handwritten English
text lines, comparing a classical CRNN/CTC pipeline with a pretrained
Transformer OCR model.

**Dataset** — [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line):
6,482 train / 976 validation / 2,915 test line images with transcriptions.

**Technologies**
- Python 3.10, PyTorch 2.5
- Hugging Face `transformers` + `datasets`
- `jiwer` for word error rate
- Streamlit for the interface
- OpenCV / Pillow for preprocessing

**Project layout**
```
HTR_Project/
├── configs/               YAML configuration (global, CRNN, TrOCR)
├── data/                  download + preprocessing scripts (no raw data)
├── src/
│   ├── datasets/          loaders, splits, augmentation
│   ├── models/            CRNN, CTC decoders, TrOCR wrapper (one file per family)
│   ├── train.py          shared training entry point (CRNN / TrOCR)
│   ├── evaluate.py       shared evaluation entry point (CER/WER for all models)
│   ├── training/          trainer, checkpoints, shared CTC loss
│   ├── evaluation/        CER/WER metrics, evaluation loops, plots
│   └── inference/         checkpoint loading + prediction
├── experiments/          logs, result tables, figures
├── checkpoints/          trained weights (not in version control)
├── demo/                 Streamlit application
├── tests/                unit and smoke tests
└── contributions/        individual logs
```

**Reproducibility** — fixed seed 42, the dataset's own splits, and every metric
in this app is read from files produced by `scripts/evaluate.py`.
        """
    )


PAGES = {
    "Home": page_home,
    "Handwriting Recognition": page_recognition,
    "Model Comparison": page_comparison,
    "Dataset Explorer": page_dataset,
    "Model Information": page_model_info,
    "About": page_about,
}


def main() -> None:
    st.sidebar.title("HTR Project")
    page = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.markdown("---")
    models = available_models()
    st.sidebar.caption(
        "Checkpoints: " + (", ".join(models) if models else "none yet")
    )
    st.sidebar.caption(f"Dataset cached: {'yes' if (Path.home() / '.cache/huggingface/datasets').is_dir() else 'unknown'}")
    PAGES[page]()


if __name__ == "__main__":
    main()
