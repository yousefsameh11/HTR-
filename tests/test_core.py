"""Unit and smoke tests for the CRNN pipeline (no dataset download needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.collate import compute_cnn_output_lengths, crnn_collate_fn
from src.data.preprocessing import CRNNPreprocessConfig, CRNNPreprocessor
from src.data.vocabulary import Vocabulary
from src.evaluation.metrics import (
    cer,
    compute_cer_wer,
    corpus_cer,
    jiwer_measures,
    normalize_text,
    wer,
)
from src.inference.predict import CRNNInference
from src.models.crnn import CRNN
from src.models.decoder import beam_search_decode, greedy_decode
from src.training.losses import make_ctc_loss


# --------------------------------------------------------------------- vocab
def test_vocabulary_encoding_and_ctc_collapse():
    vocab = Vocabulary().build_from_texts(["hello world"])

    assert vocab.blank_index == 0
    assert vocab.num_classes == 1 + 1 + len(vocab.chars)

    indices = vocab.text_to_indices("hello")
    assert len(indices) == 5

    # CTC: identical consecutive predictions collapse to one character.
    assert vocab.indices_to_text([indices[0], indices[1], indices[2], indices[2], 0]) == "hel"

    # A blank between repeated characters keeps both characters.
    h, e, l, o = (vocab.text_to_indices(ch)[0] for ch in "helo")
    assert vocab.indices_to_text([h, e, l, 0, l, o]) == "hello"

    # Round trip is exact once the CTC rule is satisfied.
    round_trip = vocab.text_to_indices("hello")
    assert vocab.indices_to_text(round_trip, keep_repeats=True) == "hello"


def test_vocabulary_save_load(tmp_path):
    vocab = Vocabulary().build_from_texts(["abc 123"])
    path = tmp_path / "vocab.json"
    vocab.save(str(path))

    loaded = Vocabulary.load(str(path))
    assert loaded.num_classes == vocab.num_classes
    assert loaded.text_to_indices("abc") == vocab.text_to_indices("abc")


def test_unknown_policies():
    vocab = Vocabulary().build_from_texts(["abc"])
    assert vocab.text_to_indices("azc", unknown_policy="skip") == vocab.text_to_indices("ac")
    with pytest.raises(ValueError):
        vocab.text_to_indices("z", unknown_policy="strict")
    assert vocab.text_to_indices("z", unknown_policy="map") == [vocab.unk_index]


# ---------------------------------------------------------------- collation
def _sample(width: int, text: str, vocab: Vocabulary, index: int) -> dict:
    image = np.random.randn(1, 64, width).astype(np.float32)
    target = vocab.text_to_indices(text)
    return {
        "image": image,
        "text": text,
        "target": target,
        "target_length": len(target),
        "index": index,
    }


def test_collate_pads_variable_widths():
    vocab = Vocabulary().build_from_texts(["aaa bbb ccc"])
    batch = [
        _sample(40, "aaa", vocab, 0),
        _sample(27, "bbb", vocab, 1),
        _sample(35, "ccc", vocab, 2),
    ]
    out = crnn_collate_fn(batch)

    assert out["images"].shape == (3, 1, 64, 40)
    assert out["widths"] == [40, 27, 35]
    assert out["targets"].tolist() == [t for s in batch for t in s["target"]]
    assert out["target_lengths"].tolist() == [3, 3, 3]
    # padded region uses the padding value, real data is untouched
    assert torch.allclose(out["images"][1, :, :, 27:], torch.full((1, 64, 13), -1.0))
    assert torch.allclose(
        out["images"][1, :, :, :27], torch.from_numpy(batch[1]["image"]).float()
    )
    assert out["input_lengths"].tolist() == [5, 3, 4]


def test_cnn_output_lengths_uses_floor():
    # exact floors of the three stride-2 pools
    assert compute_cnn_output_lengths([64, 65, 71, 8, 7]) == [8, 8, 8, 1, 1]


# -------------------------------------------------------------------- CRNN
def test_crnn_forward_shapes_and_ctc():
    vocab = Vocabulary().build_from_texts(["abcdefghij"])
    model = CRNN(img_height=64, num_classes=vocab.num_classes, rnn_hidden_size=32, rnn_num_layers=2)
    model.eval()

    images = torch.randn(2, 1, 64, 96)
    logits = model(images)
    assert logits.dim() == 3
    assert logits.shape[1] == 2
    assert logits.shape[2] == vocab.num_classes
    assert logits.shape[0] == 96 // 8

    # CTC target shorter than the frame count -> finite loss
    batch = {
        "images": images,
        "widths": [96, 88],
        "input_lengths": torch.tensor([12, 11]),
        "target_lengths": torch.tensor([3, 4]),
        "targets": torch.tensor([2, 3, 4, 5, 6, 7, 8]),
        "texts": ["abc", "defg"],
    }
    loss = make_ctc_loss()(model, batch, "cpu", False)
    assert torch.isfinite(loss) and loss.item() > 0

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ctc_rejects_inconsistent_lengths():
    vocab = Vocabulary().build_from_texts(["ab"])
    model = CRNN(img_height=64, num_classes=vocab.num_classes, rnn_hidden_size=16)
    batch = {
        "images": torch.randn(1, 1, 64, 32),
        "widths": [32],
        "input_lengths": torch.tensor([999]),  # impossible
        "target_lengths": torch.tensor([2]),
        "targets": torch.tensor([2, 3]),
        "texts": ["ab"],
    }
    with pytest.raises(ValueError):
        make_ctc_loss()(model, batch, "cpu", False)


# ---------------------------------------------------------------- decoders
def test_greedy_and_beam_decoding():
    # T=4 frames, 3 classes: 0 = blank, 1 = 'a', 2 = 'b'
    logits = torch.full((4, 1, 3), -10.0)
    for t, cls in enumerate([1, 1, 0, 2]):
        logits[t, 0, cls] = 10.0

    greedy = greedy_decode(torch.log_softmax(logits, dim=-1), blank_index=0)
    assert greedy == [[1, 2]]

    beam = beam_search_decode(torch.log_softmax(logits, dim=-1), beam_width=3, blank_index=0)
    assert beam == [[1, 2]]


# ------------------------------------------------------------ preprocessing
def test_crnn_preprocessor_preserves_aspect_ratio():
    prep = CRNNPreprocessor(CRNNPreprocessConfig(img_height=64, enabled=False))
    image = Image.new("L", (320, 128), color=255)
    arr = prep(image)
    assert arr.shape == (1, 64, 160)
    assert arr.dtype == np.float32
    assert -1.5 < float(arr.mean()) < 1.5


# ------------------------------------------------------------------ metrics
def test_normalisation_and_cer_wer():
    assert normalize_text("Hello,   World!") == "hello world"
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    assert cer("abc", "") == pytest.approx(1.0)
    assert wer("one two three", "one two tree") == pytest.approx(1 / 3)
    assert corpus_cer(["abc", "abcd"], ["abc", "abce"]) == pytest.approx(1 / 7)

    cer_score, wer_score, details = compute_cer_wer(["hello world"], ["hello word"])
    assert cer_score > 0 and wer_score == pytest.approx(0.5)
    assert details[0]["reference"] == "hello world"

    measures = jiwer_measures(["a b c"], ["a b d"])
    assert measures["substitutions"] == 1
    assert measures["hits"] == 2


# --------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_and_inference(tmp_path):
    vocab = Vocabulary().build_from_texts(["abc"])
    model = CRNN(img_height=64, num_classes=vocab.num_classes, rnn_hidden_size=16)
    ckpt = tmp_path / "crnn_test.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "img_height": 64,
                    "num_classes": vocab.num_classes,
                    "rnn_hidden_size": 16,
                    "rnn_num_layers": 2,
                },
                "vocabulary": {
                    "blank_token": vocab.blank_token,
                    "unk_token": vocab.unk_token,
                    "use_unk": vocab.use_unk,
                    "chars": vocab.chars,
                },
            },
        },
        ckpt,
    )

    predictor = CRNNInference(str(ckpt), device="cpu")
    result = predictor.predict(Image.new("L", (200, 128), color=255))

    assert isinstance(result.text, str)
    assert result.inference_time >= 0
    assert result.model_type == "crnn"
