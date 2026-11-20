# NEXT_STEPS

## Current state

Both models are trained on the full 6,482-line train split, evaluated on the full
2,915-line test split, and the Streamlit app reads the resulting metrics.

| Model | Checkpoint | Test CER | Test WER | Speed |
|---|---|---|---|---|
| CRNN + BiLSTM + CTC | `checkpoints/crnn/crnn_best.pt` | 14.06 % | 43.51 % | 0.6 ms/line (greedy) |
| TrOCR-small fine-tuned (5 epochs) | `checkpoints/trocr/final` | **5.34 %** | **10.88 %** | 363 ms/line (4 beams) |

Validation: CRNN best val loss 0.2787 (epoch 39 of 49); TrOCR val CER 1.85 % /
WER 4.73 % on 300 lines. Figures and predictions are in `experiments/`.

Reproduce end to end (RTX 3050, 4 GB):

```bash
python scripts/train_crnn.py --epochs 50 --device cuda --batch-size 16 \
    --bucket-batches 10 --num-workers 2
python scripts/train_trocr.py --epochs 5 --batch-size 32 --grad-checkpointing \
    --eval-subset 300 --device cuda
python scripts/evaluate.py --batch-size 16 --trocr-batch-size 16 --trocr-beams 4
python -m streamlit run app.py
```

## Remaining, optional work

1. **Zero-shot TrOCR baseline on the test split.** The pretrained model is
   available and works, but it has no test-set metrics; adding them would make
   the "fine-tuning gain" explicit (~18 min of GPU time at 4 beams).
2. **CRNN beam search** (`--beam-width 5`): greedy CER is 14.06 %; beam search
   usually gains 1–2 points at a large speed cost.
3. **Error analysis** by character class and line length from
experiments/predictions/*.txt.
4. **Writer-level analysis is impossible** with Teklia/IAM-line: it ships only
   `image` and `text`.

## Performance notes (measured on this machine)

- CRNN epoch: 54 s with width-bucketed batches (2 % padding) and 2 loader
  workers; 137–203 s before, mostly padding plus serial data loading.
- Full-model FP16 aborts the process inside cuDNN's LSTM path on this
  Windows/CUDA stack; FP16 is therefore applied to the conv stack only and the
  LSTM/CTC run in FP32.
- DataLoader workers above 2 added no throughput and previously died mid-run;
  0 works too (about 10 % slower per epoch).
- TrOCR: batch 32 is the sweet spot (1.92 GB peak, ~22 samples/s); batch 64
  needs 2.2 GB and is no faster, batch 1/8 are 3–5x slower.

## Useful commands

```bash
# tests
python -m pytest tests -q

# single-image inference
python scripts/predict.py --model crnn --checkpoint checkpoints/crnn/crnn_best.pt \
    --vocab checkpoints/crnn/vocab.json --image path/to/line.png

# app
python -m streamlit run app.py
```
