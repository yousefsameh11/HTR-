# PROJECT_STATUS

Last updated: 2026-11-18 (after takeover audit and repair pass)

Verified facts are marked **[verified]**; assumptions/estimates are marked
**[assumption]**.

---

## A. PROJECT OVERVIEW

**Goal:** recognise a complete handwritten English text line from an image.
Input: one line image. Output: the transcribed text.

Two models are compared on the same data:

1. **CRNN + BiLSTM + CTC** — CNN feature extractor (3 stride-2 pool blocks,
   H/8 × W/8 feature map), 2-layer bidirectional LSTM (hidden 256), linear
   classification layer over vocabulary + CTC blank, trained with `CTCLoss`,
   decoded greedily or with prefix beam search.
2. **TrOCR** — pretrained `microsoft/trocr-small-handwritten`
   (`VisionEncoderDecoderModel`: DeiT 384×384 encoder + 6-layer Transformer
   decoder, XLM-R vocabulary), fine-tuned on IAM-line with teacher forcing and
   beam-search generation.

Dataset: `Teklia/IAM-line`, line-level, writer-independent splits.

---

## B. EXISTING IMPLEMENTATION

| Path | Responsibility | State |
|------|----------------|-------|
| `main.py` | CLI dispatcher | working |
| `configs/config.yaml` | global paths/columns | working |
| `configs/crnn_config.yaml` | CRNN + training + augmentation | repaired (`scale` type, LR YAML type) |
| `configs/trocr_config.yaml` | TrOCR training + generation | repaired (LR YAML type) |
| `src/data/dataset.py` | HF split wrapper, TrOCR dataset + collator | repaired (see F) |
| `src/data/preprocessing.py` | CRNN grayscale/resize/normalise/augment; TrOCR processor wrapper | unchanged, verified |
| `src/data/vocabulary.py` | char vocabulary, CTC blank at 0 | repaired (decode bug) |
| `src/data/collate.py` | variable-width batching, CTC targets | repaired (invalid padding) |
| `src/models/crnn.py` | CRNN forward, output lengths | repaired (length formula) |
| `src/models/decoder.py` | greedy + prefix beam CTC decoding | verified working |
| `src/models/trocr.py` | TrOCR loading/processor/generate/save | repaired (tokenizer, weights, start token) |
| `src/training/trainer.py` | loop, scheduler, early stop, checkpoints | repaired (verbose arg, extra state) |
| `src/training/losses.py` | shared CTC loss + alignment validation | new |
| `src/training/checkpoint.py` | save/load helpers | working |
| `src/evaluation/metrics.py` | CER/WER + jiwer alignment | rewritten for jiwer 4 |
| `src/evaluation/evaluate.py` | CRNN/TrOCR evaluation loops | rewritten (was truncated) |
| `src/evaluation/visualization.py` | comparison charts | rewritten (crashed without both models) |
| `src/inference/predict.py` | checkpoint loading + prediction | repaired (TrOCR path, vocab from ckpt) |
| `scripts/inspect_dataset.py` | dataset audit | working (pre-existing) |
| `scripts/train_crnn.py` | CRNN training CLI | rewritten (file was syntactically broken) |
| `scripts/train_trocr.py` | TrOCR fine-tuning CLI | rewritten for transformers 5 |
| `scripts/evaluate.py` | evaluation/comparison CLI | rewritten (was syntactically broken) |
| `scripts/predict.py` | single-image inference CLI | updated |
| `app.py` | Streamlit UI, 6 pages | **new** |
| `tests/` | pytest suite (15 tests) | **new** |
| `README.md`, this file, `DEVELOPMENT_LOG.md`, `NEXT_STEPS.md` | documentation | **new** |

---

## C. COMPLETED WORK (verified in this session)

1. **Dataset pipeline** **[verified]**
   `load_dataset("Teklia/IAM-line")` loads from the local HF cache in ~7 s:
   train 6,482 / validation 976 / test 2,915, columns `image` (grayscale,
   128 px high, variable width) and `text` (1–93 chars, none empty).
2. **Character vocabulary + CTC encoding** **[verified]** — 68 classes for the
   full train split (blank=0, unk=1, 66 characters); encode/decode round-trips;
   CTC repeat collapse and blank-separated repeats behave correctly.
3. **Variable-width batching** **[verified]** — pad to batch max width, exact
   `floor(W/8)` input lengths, padded values excluded via CTC lengths.
4. **CRNN forward + CTC loss + backward** **[verified]** by tests and by a real
   1-epoch training run (train loss 4.22 → 3.86, val loss 3.33 on 256/64
   samples, 49 s on CPU); checkpoints, history JSON and summary are written.
5. **CRNN training script** end-to-end, including vocabulary saved inside the
   checkpoint (`config.vocabulary`) so inference works without a separate file.
6. **TrOCR loading** **[verified]** — processor rebuilt, safetensors conversion,
   forward/backward, generation, save/reload of model + processor.
7. **TrOCR fine-tuning script** end-to-end on 4 train / 2 val samples
   (trainer, collator, `-100` masking, validation CER/WER, final save).
8. **Pretrained TrOCR is functional zero-shot** **[verified]**:
   `example_00 → "put down a resolution on the subject"`,
   `example_01 → "and he is to be backed by Mr. Will"`,
   `example_02 → "nominating any more Labour life . Peers"`.
9. **Evaluation** — CRNN CER/WER/inference-time and TrOCR CER/WER/inference-time
   computed on identical test samples; metrics, TSV predictions and figures are
   written to `experiments/`. Missing checkpoints are reported as unavailable.
10. **Inference CLI** for both models, measuring model-only inference time.
11. **Streamlit app** with Home / Recognition / Comparison / Dataset Explorer /
    Model Info / About; it reads checkpoints and result files from disk and
    explains what is missing instead of fabricating output.
12. **Test suite** — `python -m pytest tests -q` → 15 passed (11 core + 4 TrOCR).

---

## D. PARTIALLY COMPLETED WORK

1. **Real model training.** Both training pipelines are verified on tiny
   subsets only. No full training has been run (blocked on the GPU decision in
   `NEXT_STEPS.md`); therefore **no scientifically meaningful CRNN or TrOCR
   metrics exist yet**.
2. **`experiments/`** is currently empty (the smoke-evaluation artifacts were
   removed so untrained outputs are never presented as real results). The app
   shows its "run evaluate.py" notice until real evaluation is run.
3. **Writer independence** cannot be analysed beyond the dataset's own splits:
   the HF dataset exposes only `image` and `text` (no writer id column).
4. **Beam-search decoding for CRNN** is implemented and unit-tested, but no
   accuracy comparison greedy-vs-beam has been run (needs a trained model).
5. **Mixed precision** is enabled for TrOCR only on CUDA (`fp16=use_cuda`);
   CRNN training is currently full precision (fast enough on GPU, safe on CPU).

---

## E. MISSING COMPONENTS (before takeover → now)

Missing at takeover and now implemented: `app.py`, `README.md`, project status
documents, test suite, CRNN loss module, working train/evaluate scripts.

Still missing intentionally (out of scope until training):

- Full trained checkpoints (`checkpoints/crnn/crnn_best.pt`,
  `checkpoints/trocr/final/`).
- Real evaluation artifacts in `experiments/`.
- Comparative conclusions ("which model is better") — cannot be stated without
  real test metrics.
- Optional: sentence-level normalisation study, writer-level error analysis,
  ONNX export.

---

## F. EXISTING ERRORS (found during the audit, and their fixes)

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | `scripts/train_crnn.py` would not import (`SyntaxError`) | second half of `build_compute_loss` was appended after `main()`, file truncated | script rewritten; loss moved to `src/training/losses.py` |
| 2 | `scripts/evaluate.py` would not import | `load_crnn`/`crnn_compute_loss` spliced into `main` and end of file | script rewritten |
| 3 | `src/evaluation/evaluate.py` broken | `references.append` was outside the batch loop, `return` placed inside `save_predictions` | rewritten |
| 4 | `crnn_collate_fn` crashed on variable widths | `pad_sequence` on `(C,H,W)` tensors pads the channel dim and requires identical trailing dims | per-sample `F.pad` on width, then `stack` |
| 5 | CTC input lengths too large | `ceil(W/8)` instead of `floor(W/8)`; three stride-2 pools floor the size | floor formula everywhere; loss raises if `input_length > T` |
| 6 | Repeated characters never collapsed in decoding | compared a character against an integer index (`ch != prev_idx`) | compare indices; blanks reset the previous label |
| 7 | `scripts/evaluate.py` reported WER on empty references | references were never collected | references taken from `batch["texts"]` |
| 8 | `make_all_figures` crashed with one model | hard-coded `crnn_metrics[...]` / `trocr_metrics[...]` | charts include only available models; "no results" placeholder |
| 9 | TrOCR processor could not load | checkpoint has no fast tokenizer file and `tokenizer_config.json` omits `vocab_file`; transformers 5 conversion path fails | rebuilt XLM-R tokenizer from `sentencepiece.bpe.model` via `XLMRobertaConverter` (verified ids 0/1/2/3) |
| 10 | `sentencepiece` import failed | protobuf runtime 5.29.6 < gencode 7.36.1 | `pip install "protobuf>=6.36"` (now 7.36.2) |
| 11 | TrOCR weights would not load | checkpoint ships legacy `pytorch_model.bin`; transformers 5 blocks `torch.load` on torch < 2.6 | one-off conversion to safetensors through transformers' loader (preserves DeiT key renaming) |
| 12 | First conversion silently dropped all 12 encoder layers | manual `load_state_dict` cannot map `encoder.encoder.layer.*` → `encoder.layers.*` | conversion now uses `from_pretrained` (with the version gate bypassed for this trusted public checkpoint) |
| 13 | TrOCR produced repeated garbage text | config override set `decoder_start_token_id` to 0 instead of the published 2 | preserve `</s>` (2); regression test added |
| 14 | TrOCR fine-tuning crashed in evaluation | collator added a `texts` key that `model.generate()` rejects | collator returns model inputs only |
| 15 | TrOCR training crashed on generation-config save | transformers 5 rejects `early_stopping=True` with `num_beams=1` | generation config uses `early_stopping=False`; beam early stopping passed at `generate()` |
| 16 | `TypeError: 'float' object is not iterable` in training | `augmentation.scale` was a scalar YAML float | config uses `[0.95, 1.05]`; script accepts a scalar too |
| 17 | `TypeError: '<=' not supported between float and str` | YAML 1.1 parses `1e-3` as a string | configs use `0.001` / `0.00005`; scripts cast to `float` |
| 18 | Dataset construction decoded all 6.5k images up front | `_validate` fetched every sample | validation reads only the text column |
| 19 | transformers 5 API drift | `evaluation_strategy` / `tokenizer=` removed | `eval_strategy`, `processing_class` |

---

## G. TRAINING STATUS

Both models are genuinely trained on the full 6,482-sample train split on the
RTX 3050 (CUDA 12.4 build of torch 2.5.1).

| Model | Checkpoint | Run | Wall clock | Validation | Test CER | Test WER | Speed |
|---|---|---|---|---|---|---|---|
| CRNN + BiLSTM + CTC | `checkpoints/crnn/crnn_best.pt` | 49 epochs (early stop; best epoch 39) | 15 min (epochs 1-7) + 38 min (8-49) | val loss 0.2787 | **14.06 %** | **43.51 %** | 0.6 ms/line (greedy) |
| TrOCR-small fine-tuned | `checkpoints/trocr/final` | 5 epochs | 857 s | val CER 1.85 % / WER 4.73 % (300 lines, 4 beams) | **5.34 %** | **10.88 %** | 363 ms/line (4 beams) |

Test numbers are corpus-level CER/WER on the full 2,915-line IAM-line test
split (`experiments/metrics/*.json`). Fine-tuned TrOCR is clearly more accurate;
CRNN is ~600x faster per line and needs no tokenizer.

Training setup actually used:

- **CRNN**: batch 16, AdamW LR 1e-3, ReduceLROnPlateau (factor 0.5,
  patience 5), early stop 10, FP16 convolution path (LSTM/CTC in FP32),
  width-bucketed batches (2 % padding waste), 2 DataLoader workers,
  peak 2.76 GB VRAM.
- **TrOCR**: batch 32, grad accumulation 1, gradient checkpointing, FP16,
  LR 5e-5, warmup 100, peak 1.92 GB VRAM. Batch sizes were picked by
  measurement (batch 64: 2.2 GB and no faster; batch 1: 0.64 GB and 3x slower).

Artefacts: `experiments/metrics/{crnn,trocr,comparison}.json`,
`experiments/predictions/*.txt`, and 8 figures in `experiments/figures/`.

Training-speed work (measured; details in `DEVELOPMENT_LOG.md`):

- Width bucketing + 2 loader workers cut the CRNN epoch from 137-203 s to
  **54 s** (padding waste 40 % -> 2 %).
- FP16 on the conv stack only: full-model FP16 aborted the process inside
  cuDNN's LSTM path on this Windows/CUDA stack (native `AbortHandler` crash).

---

## H. ENVIRONMENT STATUS

| Item | Value |
|------|-------|
| OS | Windows 10 (10.0.26200), Git Bash |
| Python | 3.10.11 |
| PyTorch | 2.5.1 **+cpu** (`torch.cuda.is_available() == False`) |
| torchvision | 0.20.1+cpu |
| CUDA in torch | not available (CPU wheel) |
| GPU present | NVIDIA GeForce RTX 3050 Laptop, **4,096 MiB VRAM**, driver CUDA 13.3 |
| transformers | 5.14.1 |
| datasets | 5.2.0 |
| huggingface-hub | 1.33.0 |
| tokenizers | 0.22.2 |
| accelerate | 1.15.0 |
| sentencepiece | 0.2.3 |
| protobuf | 7.36.2 (upgraded from 5.29.6 — required) |
| jiwer | 4.0.0 |
| streamlit | 1.57.0 |
| opencv-python | 4.13.0.92 |
| Pillow | 12.1.1 |
| numpy / pandas | 1.26.4 / 2.3.3 |
| matplotlib | 3.10.8 |
| scikit-learn | 1.7.2 |
| pytest | installed in this session (was absent) |

Dataset cache: `~/.cache/huggingface/hub/datasets--Teklia--IAM-line` ≈ 434 MB
(parquet files for the three splits). Model cache:
`models--microsoft--trocr-small-handwritten`.

---

## I. NEXT DEVELOPMENT STEPS (priority order)

Done in this session: CUDA torch install, CRNN training, TrOCR fine-tuning, full
test evaluation, figure generation, Streamlit restart, documentation.

Remaining, all optional:

1. **Zero-shot TrOCR baseline on the test split** — the app can show the
   pretrained model, but no test-set metrics were computed for it; running it
   costs ~18 min of GPU time at 4 beams.
2. **CRNN beam search** (`--beam-width 5`) — greedy CER is 14.06 %; beam search
   typically gains 1–2 points and is slow on CPU/GPU (`evaluate.py --beam-width`).
3. **Error analysis** by character class / line length, and a confusion matrix
   from `experiments/predictions/*.txt`.
4. **Writer-level analysis is impossible** with Teklia/IAM-line: it ships no
   writer id column.
