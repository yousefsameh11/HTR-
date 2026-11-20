# DEVELOPMENT_LOG

Append-only record of work, bugs, tests and decisions.

---

## 2026-11-18 — Takeover audit and full pipeline repair

### Starting point (what the previous agent left)

- `HTR_Project/` contained a package skeleton (`src/data`, `src/models`,
  `src/training`, `src/evaluation`, `src/inference`), three YAML configs, five
  `scripts/`, `main.py`, `requirements.txt`, dataset-inspection examples, and
  install/TroCR test logs.
- **No Git repository**, **no checkpoints**, **no results**, **no `app.py`**,
  **no README**, **no tests**, **no project documentation**.
- Several Python files were structurally broken (functions split across the file
  or truncated), so `scripts/train_crnn.py`, `scripts/evaluate.py` and
  `src/evaluation/evaluate.py` could not even be imported.
- `logs/trocr_test_err.txt` already showed the TrOCR tokenizer failure; the
  previous agent had not resolved it.

### Audit (no code changes)

- Verified dataset: `Teklia/IAM-line`, 6,482 / 976 / 2,915 samples, columns
  `image` + `text`, grayscale lines 128 px high, no writer column, cached
  locally (434 MB parquet).
- Verified environment: Python 3.10.11, Windows, torch 2.5.1 **CPU**, no CUDA in
  torch, but an RTX 3050 Laptop (4 GB) is present; transformers 5.14.1,
  datasets 5.2.0, jiwer 4.0.0, streamlit 1.57.0.
- Inspected `archive/iam_words` — word-level IAM data, deliberately not used.
- Ran the existing code paths: dataset loads, CRNN model instantiates, but
  collation and CTC lengths were provably wrong (see fixes below).

### Files changed

**Rewritten / repaired**

- `src/data/collate.py` — per-sample width padding (`F.pad` + `stack`), exact
  `floor(W/8)` input lengths, padding value documented.
- `src/data/vocabulary.py` — fixed CTC repeat collapse (compared char to index),
  exact `num_classes`, added `keep_repeats` debugging option.
- `src/data/dataset.py` — cheap split validation (text column only); added
  `TrOCRDataset` + `TrOCRCollator` (dynamic label padding, `-100` mask).
- `src/models/crnn.py` — `floor(W/8)` output lengths, `rnn_type` option,
  `config_dict()` for checkpoint metadata.
- `src/models/trocr.py` — robust processor builder (XLM-R converter fallback),
  one-off safetensors conversion of the legacy `.bin`, preserved
  `decoder_start_token_id=2`, `generate()` for PIL/tensor input, save/load
  model+processor, processor cache helper.
- `src/training/trainer.py` — removed deprecated scheduler `verbose`, added
  `extra_state` (model config + vocabulary in checkpoints), non-finite loss
  guard, history + total time persisted.
- `src/training/losses.py` — **new**: shared CTC loss with alignment validation.
- `src/evaluation/metrics.py` — jiwer 4 compatible, corpus-level CER/WER, richer
  per-sample details, word-level measure counts from `process_words`.
- `src/evaluation/evaluate.py` — working CRNN loop (references collected!),
  TrOCR evaluation over a split, common result dictionary, save helpers.
- `src/evaluation/visualization.py` — figures work with one or both models;
  grouped CER/WER chart, time charts, prediction table with line images.
- `src/inference/predict.py` — checkpoint config/vocab loading, correct TrOCR
  generation path, inference timing.
- `scripts/train_crnn.py`, `scripts/train_trocr.py`, `scripts/evaluate.py`,
  `scripts/predict.py` — rewritten; transformers 5 arguments; TrOCR validation
  CER/WER; summaries/history written for the app and plots.
- `configs/crnn_config.yaml` — `augmentation.scale: [0.95, 1.05]`,
  `learning_rate: 0.001` (YAML strings/floats).
- `configs/trocr_config.yaml` — `learning_rate: 0.00005`.

**New**

- `app.py` (Streamlit, 6 pages), `tests/test_core.py`, `tests/test_trocr.py`,
  `tests/conftest.py`, `README.md`, `PROJECT_STATUS.md`, `DEVELOPMENT_LOG.md`,
  `NEXT_STEPS.md`, updated `requirements.txt`.

### Bugs found and fixed

19 distinct defects; the full table is in `PROJECT_STATUS.md` section F. The
most consequential:

1. Spliced/truncated functions in the training and evaluation scripts.
2. `pad_sequence` misuse → variable-width batches crashed.
3. `ceil` instead of `floor` CNN length math → CTC alignment errors.
4. CTC decoding never collapsed repeated characters.
5. TrOCR tokenizer could not be built (no fast file, missing `vocab_file`).
6. TrOCR weights blocked by `torch.load` guard; first conversion dropped all
   encoder layers; the fix converts through transformers' own loader.
7. TrOCR decoded from `<s>` instead of `</s>` → repeated garbage; now preserves
   the published `decoder_start_token_id=2` and is regression-tested.
8. TrOCR collator leaked a `texts` key into `generate()` during evaluation.
9. YAML `1e-3` parsed as a string by PyYAML 1.1 → optimizer `TypeError`.
10. `augmentation.scale` scalar vs tuple `TypeError`.

### Environment changes (each technically required)

- `protobuf` 5.29.6 → 7.36.2 (sentencepiece bindings refuse to import otherwise).
- `pytest` installed (test suite). No other package was upgraded or replaced.

### Verification performed

| Check | Result |
|-------|--------|
| `python -m pytest tests/test_core.py -q` | 11 passed |
| `python -m pytest tests/test_trocr.py -q` | 4 passed |
| Dataset load + split sizes | 6,482 / 976 / 2,915 verified |
| CRNN smoke training (256/64, 1 epoch, CPU) | loss 4.22→3.86, val 3.33, 48.9 s, checkpoints + history written |
| CRNN inference CLI (smoke checkpoint) | ran, 70 ms/line |
| `scripts/evaluate.py` without checkpoints | clean "SKIPPED" messages, exit 0 |
| `scripts/evaluate.py` on smoke checkpoint (16 test lines) | metrics/TSV/figures written (artifacts deleted afterwards) |
| TrOCR smoke fine-tune (4/2, 1 epoch, CPU) | eval_loss 11.52, eval_CER 0.029, model+processor saved and reloaded |
| TrOCR inference CLI (smoke model) | `"put down a resolution on the subject"`, 432 ms/line |
| Pretrained TrOCR zero-shot on 3 example lines | all three transcribed correctly |
| Streamlit app | started and served (see NEXT_STEPS for how to relaunch) |

### Decisions

- **Continue, don't restart:** the existing architecture, module layout,
  configuration files, dataset cache and pretrained checkpoint were kept; only
  broken code was repaired and missing pieces added.
- **No fabricated results:** all smoke artifacts were deleted; `experiments/` stays
  empty until real training and evaluation are run.
- **Full training not started:** the prompt requires explicit approval. CPU-only
  torch makes full CRNN/TrOCR training impractical; a CUDA build decision is
  proposed in `NEXT_STEPS.md`.
- **Word-level `archive/` data not used** — the task is line-level.

### Open items

- Install CUDA torch (user decision) and run full training twice (CRNN, TrOCR).
- Run evaluation on the full test split.
- Review the Streamlit app against real metrics after training.

---

## 2026-11-18 — Streamlit launch verification

- `streamlit run app.py --server.headless true --server.port 8501` starts and the
  health endpoint returns `ok`; the app is registered in the preview panel.
- No checkpoints/results exist yet, so the Recognition and Comparison pages show
  their "train/evaluate first" notices — that is the intended behaviour, not a
  failure.

---

## 2026-11-20 — GPU training, evaluation and app verification (Option A)

### Environment

- Installed the CUDA 12.4 build of torch via direct wheel download
  (`wheels/`, `wheels/install_cu124.log`, `install_exit=0`):
  `torch 2.5.1+cu124`, `torchvision 0.20.1+cu124`, cuDNN 90100,
  `NVIDIA GeForce RTX 3050 Laptop GPU`, 4.29 GB, compute capability 8.6.
- Protobuf was upgraded 5.29.6 -> 7.36.2 earlier because `sentencepiece` could
  not be imported; TrOCR's processor is rebuilt from `sentencepiece.bpe.model`.

### Measured failures and fixes

1. **Full-model FP16 aborted the process.** After epoch 1 the run died with
   `Unhandled exception caught in c10/util/AbortHandler.h` (native stack:
   `torch_cuda`/`c10_cuda`). Cause: cuDNN's FP16 LSTM path on this stack.
   Fix: FP16 is applied to the convolutional stack only
   (`CRNN.set_fp16_cnn`), LSTM + CTC run in FP32; `GradScaler` unchanged.
2. **DataLoader workers died mid-run** (`RuntimeError: DataLoader worker (pid ...)
   exited unexpectedly`) and took the training process with them.
   Fix: `--num-workers 0`/2 only; loading is ~2 ms/image so it is not needed for
   throughput.
3. **Padding waste dominated the CRNN step.** Collate pads to the widest line in
   the batch, so random batches ran at ~1.5x the average width.
   Fix: `src/data/sampling.py::WidthBucketBatchSampler` (sort by width, shuffle
   inside buckets, shuffle batch order). Padding waste 40 % -> 2 %.
4. **`plot_loss_curves` crashed** when history lengths differed by one (TrOCR
   appends a final eval): now plots the common prefix.

### Speed optimisation (measured, 6,482-sample CRNN epoch)

| Setting | Epoch time |
|---|---|
| Batch 16, 4 workers, full-model FP16 | 470 s (crashed after epoch 1) |
| Batch 16, 0 workers, FP16 conv only | 137–203 s |
| Batch 16, 2 workers, width buckets (50 batches) | 54–70 s (26 % padding saved) |
| Batch 16, 2 workers, width buckets (10 batches) | **53–54 s** (2 % padding saved) |
| Batch 32, 0 workers | 2x slower (VRAM spill) |

TF32 is enabled for the FP32 matmuls; `cudnn.benchmark` stays off because the
widths are variable.

### Training

- **CRNN**: 49/50 epochs (early stop, patience 10), best val loss **0.2787** at
  epoch 39; 15 min for epochs 1–7 + 38 min for epochs 8–49 (resumed with
  `--resume`), peak VRAM 2.76 GB. Checkpoints `crnn_best.pt`, `crnn_latest.pt`,
  `crnn_history.json`, `training_summary.json`.
- **TrOCR**: 5 epochs, batch 32, gradient checkpointing, FP16, LR 5e-5,
  warmup 100, 857 s, peak VRAM 1.92 GB, best val loss 0.1903, val CER 1.85 % /
  WER 4.73 % on 300 lines. Checkpoint `checkpoints/trocr/final`.
- The pilot confirmed the fine-tuned weights are a faithful copy of the
  pretrained ones (identical key names/count, identical predictions before
  training); the `missing/unexpected keys` log line comes from a secondary
  load path and is harmless.

### Evaluation (full test split, 2,915 lines)

| Model | CER | WER | Speed |
|---|---|---|---|
| CRNN (greedy) | 14.06 % | 43.51 % | 0.6 ms/line |
| TrOCR fine-tuned (4 beams) | **5.34 %** | **10.88 %** | 363 ms/line |

- `scripts/evaluate.py` OOMed at CRNN batch 32 (long lines); batch 16 works.
- All 8 figures were regenerated from the saved metrics after the plot fix.
- `python -m pytest tests -q` -> 15 passed.
- Streamlit restarted (`http://127.0.0.1:8501`, health 200) with the real
  checkpoints and metrics; preview registered in the panel.
