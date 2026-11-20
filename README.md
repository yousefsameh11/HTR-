# Handwritten Text Recognition (HTR) using Deep Learning

**Team:** [yousef sameh,
momen aymen,
mahmoud amr, 
,motaz ahmed
,mohamed waled ]

**Idea chosen:** Line-level handwritten English text recognition comparing a
CRNN + BiLSTM + CTC model with a fine-tuned TrOCR (Transformer OCR) model on
the Teklia/IAM-line dataset.

**Reference implementation:**
Line-level handwritten English text recognition on the
[Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line) dataset,
comparing two architectures:

| Model | Architecture | Decoding |
|-------|--------------|----------|
| **CRNN** | CNN → BiLSTM → CTC | greedy / prefix beam search |
| **TrOCR** | DeiT ViT encoder + Transformer decoder, pretrained `microsoft/trocr-small-handwritten` | beam search |

**Input:** an image containing one handwritten English text line.
**Output:** the recognized text.

This is a sequence-recognition problem (whole lines), not character or word
classification.

---

---

## Best result

| Model | CER | WER | Inference |
|-------|-----|-----|----------|
| **TrOCR-small fine-tuned** (5 epochs) | **5.34 %** | **10.88 %** | 363 ms/line (4 beams) |
| CRNN + BiLSTM + CTC (49 epochs) | 14.06 % | 43.51 % | 0.6 ms/line (greedy) |

Evaluated on the full IAM-line test split (2,915 lines). Fine-tuned TrOCR is the
more accurate model; CRNN is ~600x faster per line.

---

## Dataset

`Teklia/IAM-line` — line images with transcriptions, already line-segmented.
The dataset ships writer-independent splits that this project preserves:

| Split | Samples |
|-------|---------|
| train | 6,482 |
| validation | 976 |
| test | 2,915 |

Images are 128 px high (grayscale, `.mode == "L"`) with variable width. Text
lengths range from 1 to 93 characters. The dataset exposes only `image` and
`text` columns, so writer identity itself is not available for analysis; the
train/validation/test separation provided by the dataset is used as-is and is
never mixed.

The uploaded dataset is cached by `datasets` in
`~/.cache/huggingface/hub/datasets--Teklia--IAM-line` (≈434 MB) and loads in
about 7 seconds. `E:\DL Project\archive` contains *word-level* IAM data; it is
intentionally not used because this project is line-level.

---

## Project structure

```
HTR_Project/
├── app.py                     Streamlit app (6 pages)
├── main.py                    CLI dispatcher (inspect/train/evaluate/predict/app)
├── configs/
│   ├── config.yaml            global paths, dataset columns, image settings
│   ├── crnn_config.yaml       CRNN + training + augmentation settings
│   └── trocr_config.yaml      TrOCR fine-tuning + generation settings
├── scripts/
│   ├── inspect_dataset.py     dataset audit (splits, columns, examples)
│   ├── train_crnn.py          CRNN training (backing implementation)
│   ├── train_trocr.py         TrOCR fine-tuning (HF Seq2SeqTrainer, backing impl.)
│   ├── evaluate.py            evaluation driver (backing implementation)
│   └── predict.py             single-image CLI inference
├── src/
│   ├── datasets/              loaders, splits, augmentation
│   ├── models/                CRNN, CTC decoders (greedy + beam), TrOCR wrapper
│   ├── train.py              shared training entry point (CRNN / TrOCR)
│   ├── evaluate.py           shared evaluation entry point (CER/WER for all models)
│   ├── training/              trainer, checkpoints, shared CTC loss
│   ├── evaluation/            CER/WER metrics, evaluation loops, plots
│   └── inference/             checkpoint loading + prediction
├── tests/                     unit + smoke tests (pytest)
├── checkpoints/               trained weights (git-ignored)
├── experiments/           metrics, predictions, figures (git-ignored)
└── demo/                 Streamlit application
```

---

## Installation

```bash
cd HTR_Project
python -m pip install -r requirements.txt
```

For an NVIDIA GPU install the CUDA build of PyTorch instead of the default CPU
wheel:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

All commands run from `HTR_Project/`.

```bash
# 1. Inspect the dataset (splits, columns, examples)
python main.py inspect

# 2. Train CRNN + BiLSTM + CTC
python src/train.py crnn --epochs 50
#    quick smoke test (CPU friendly)
python src/train.py crnn --epochs 1 --max-train 256 --max-val 64
#    (backing script, same options)
python scripts/train_crnn.py --epochs 50

# 3. Fine-tune pretrained TrOCR (never from scratch)
python src/train.py trocr --epochs 10
python src/train.py trocr --epochs 1 --max-train 32 --max-val 8
#    (backing script, same options)
python scripts/train_trocr.py --epochs 10

# 4. Evaluate both models on the same test samples
python src/evaluate.py --max-test 400
#    (backing script, same options)
python scripts/evaluate.py --max-test 400

# 5. Recognize a single line image
python scripts/predict.py --model crnn  --image path/to/line.png
python scripts/predict.py --model trocr --image path/to/line.png

# 6. Streamlit application
python main.py app          # or: streamlit run demo/app.py
```

### Training details

- **CRNN** (`configs/crnn_config.yaml`): 7.3 M parameters, image height 64 px,
  aspect-ratio-preserving resize, per-batch zero padding, BiLSTM 2×256,
  CTC loss with exact `floor(w/8)` input lengths, AdamW, ReduceLROnPlateau,
  gradient clipping, early stopping, best/latest checkpoints plus
  `crnn_history.json` and `training_summary.json`.
- **TrOCR** (`configs/trocr_config.yaml`): `VisionEncoderDecoderModel` fine-tuned
  with teacher forcing; labels tokenized with XLM-R, padding masked as `-100`,
  dynamic per-batch padding, AdamW + warmup, epoch-wise validation CER/WER,
  best model reloaded at the end and saved with its processor.

Both models use the dataset's original splits; test data is never used for
training or hyper-parameter selection.

### The checkpoint compatibility shims (important)

`microsoft/trocr-small-handwritten` is published only as a legacy
`pytorch_model.bin` and without a fast-tokenizer file:

1. transformers ≥ 5 refuses `torch.load` on torch < 2.6, so the weights are
   converted **once** to safetensors in
   `checkpoints/pretrained/trocr-small-handwritten` using transformers' own
   loader (which also applies the DeiT key renaming).
2. The checkpoint has no `tokenizer.json`, so the tokenizer is rebuilt from
   `sentencepiece.bpe.model` with the XLM-R converter. Verified token ids:
   `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3.
3. TrOCR decodes starting from `</s>` (id 2). Using `<s>` (0) produces
   degenerate repeated text — this is asserted in the test suite.

Without any fine-tuning, the loaded checkpoint already transcribes the sample
lines correctly (e.g. *"put down a resolution on the subject"*), which is the
strongest evidence that weights and tokenizer are wired up correctly.

---

## Evaluation

`scripts/evaluate.py` writes:

```
experiments/
├── metrics/crnn_metrics.json
├── metrics/trocr_metrics.json
├── metrics/comparison.json
├── predictions/crnn_predictions.txt
├── predictions/trocr_predictions.txt
└── figures/
    ├── cer_comparison.png
    ├── wer_comparison.png
    ├── cer_wer_comparison.png
    ├── inference_time.png
    ├── training_time.png
    ├── crnn_loss.png / trocr_loss.png
    └── prediction_examples.png   (image | ground truth | CRNN | TrOCR)
```

Text normalisation is shared by both models (lowercase, punctuation removed,
whitespace collapsed). CER is `sum character edits / sum reference characters`;
WER is the corpus-level word error rate from `jiwer`.

### Measured results (full test split, 2,915 lines)

| Model | CER | WER | Speed | Training |
|---|---|---|---|---|
| CRNN + BiLSTM + CTC (`checkpoints/crnn/crnn_best.pt`) | 14.06 % | 43.51 % | 0.6 ms/line (greedy) | 49 epochs, ~53 min on RTX 3050 |
| TrOCR-small fine-tuned (`checkpoints/trocr/final`) | **5.34 %** | **10.88 %** | 363 ms/line (4 beams) | 5 epochs, 14.3 min on RTX 3050 |

Validation: CRNN best val loss 0.2787; TrOCR val CER 1.85 % / WER 4.73 % on 300
lines. These are the numbers the Streamlit app reads from `experiments/`.

**No metrics are fabricated.** A model without a checkpoint is reported as
unavailable, and the Streamlit app says so instead of showing numbers.

---

## Tests

```bash
python -m pytest tests -q           # all tests (TrOCR tests load the checkpoint)
python -m pytest tests/test_core.py -q
HTR_SKIP_TROCR=1 python -m pytest tests -q
```

Covered: vocabulary encode/decode + CTC collapse, variable-width collation,
CNN output-length formula, CRNN forward + CTC loss/backward, invalid-length
rejection, greedy/beam decoding, preprocessing, CER/WER/jiwer, checkpoint
round-trip and inference, TrOCR tokenizer/ids, label masking, forward/backward,
generation and decoder start token.

---

## Streamlit application

Six pages: Home, Handwriting Recognition, Model Comparison, Dataset Explorer,
Model Information, About. Predictions always come from checkpoints on disk; if
none exist, the Recognition page explains what to train.

---

## Hardware used and expected runtimes

| Stage | Hardware | Runtime |
|-------|----------|---------|
| CRNN training (49 epochs) | NVIDIA RTX 3050 Laptop, 4.29 GB, CUDA 12.4, torch 2.5.1+cu124 | ~53 min (15 min epochs 1–7 + 38 min 8–49, resumed) |
| TrOCR fine-tuning (5 epochs) | Same GPU | 857 s (~14.3 min) |
| Full evaluation (2,915 test lines) | Same GPU | CRNN ~1.7 s; TrOCR ~17.6 min (4 beams) |
| CPU smoke test (CRNN 1 epoch, 256/64 samples) | CPU only | ~49 s |

Peak VRAM: CRNN 2.76 GB, TrOCR 1.92 GB. The CRNN uses FP16 on the conv stack
only (LSTM/CTC in FP32); full-model FP16 aborted the process on this CUDA stack.

### Reproducibility

- Fixed seed 42 everywhere.
- The dataset's own writer-independent splits are used as-is; no sample is ever
  moved between train/validation/test.
- Every number in this README is read from files in `experiments/` produced by
  real training and evaluation; nothing is fabricated.

---

## Credits

### Dataset
https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database
- **Teklia/IAM-line** — line-segmented IAM handwriting dataset, distributed via
  Hugging Face Hub: https://huggingface.co/datasets/Teklia/IAM-line
  Original IAM Handwriting Database: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

### Pretrained model
- **microsoft/trocr-small-handwritten** — TrOCR small handwritten model from
  Microsoft, published on Hugging Face Hub. Used as the initialisation for the
  fine-tuned TrOCR model. See the TrOCR paper:
  *"TrOCR: Transformer-based Optical Character Recognition with Pre-trained
  Models"* (Turc et al., 2021).

### External code / libraries
- PyTorch 2.5 — deep learning framework
- Hugging Face `transformers` 5.x, `datasets` 5.x, `huggingface-hub` — model
  loading, dataset loading, training
- `jiwer` 4.x — word error rate computation
- `sentencepiece` — XLM-R tokenizer reconstruction for the TrOCR checkpoint
- Streamlit — demo application
- OpenCV / Pillow / numpy / matplotlib / scikit-learn / pyyaml / tqdm — preprocessing,
  visualisation, utilities

### CRNN/CTC references
-Shi, B., Bai, X., & Yao, C. (2015). *"An End-to-End Trainable Neural Network
  for Image-based Sequence Recognition and Its Application to Scene Text
  Recognition."* arXiv:1507.05717 — the CRNN architecture.
- Hannun, A. (2017). *"Sequence Modeling with CTC."* Distill.

### Environment
- Python 3.10.11, Windows 10, NVIDIA GeForce RTX 3050 Laptop (4.29 GB VRAM).

---

## Project reference

This project follows the deliverable structure specified in the assignment:

```
project-name/
├── README.md
├── requirements.txt
├── configs/            # one config file per experiment
├── data/               # download + preprocessing scripts (no raw data)
├── src/
│   ├── datasets/       # loaders, splits, augmentation
│   ├── models/         # one file per model family
│   ├── train.py       # shared training entry point
│   └── evaluate.py    # shared evaluation for all models
├── experiments/       # logs, result tables, figures
├── checkpoints/       # trained weights / download link
├── demo/              # Gradio / Streamlit / FastAPI app
├── report/            # report PDF and slides
└── contributions/     # individual logs
```
