"""Tests for the CNN + BiLSTM + Bahdanau attention + LSTM decoder model.

No dataset download is needed: every test builds synthetic tensors. The suite
covers the checks required for the third architecture: CNN geometry, the
feature-map-to-sequence step, the BiLSTM encoder, attention dimensions and
masking, decoder dimensions, special tokens, teacher forcing, the padded
cross-entropy mask, forward/backward passes, greedy decoding, checkpoint reload
and variable-width batching.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.datasets.data.attention import AttentionCollator, encoder_lengths_from_widths
from src.datasets.data.attention_vocab import (
    EOS_INDEX,
    PAD_INDEX,
    SOS_INDEX,
    UNK_INDEX,
    AttentionVocabulary,
)
from src.inference.attention_inference import AttentionInference
from src.models.attention_htr import (
    AttentionDecoder,
    AttentionHTR,
    BahdanauAttention,
    BiLSTMEncoder,
    CNNEncoder,
    attention_geometry,
)
from src.training.losses import make_attention_loss, masked_cross_entropy

IMG_HEIGHT = 64
VOCAB_TEXT = "abcdefghijklmnopqrstuvwxyz ."


# --------------------------------------------------------------------- helpers
def make_vocab() -> AttentionVocabulary:
    return AttentionVocabulary.build_from_texts([VOCAB_TEXT])


def make_model(vocab: AttentionVocabulary | None = None, **overrides) -> AttentionHTR:
    vocab = vocab or make_vocab()
    kwargs = dict(
        vocab_size=vocab.num_classes,
        img_height=IMG_HEIGHT,
        encoder_hidden=32,
        encoder_layers=1,
        decoder_hidden=32,
        decoder_layers=1,
        embed_dim=16,
        attn_dim=16,
        dropout=0.0,
        cnn_dropout=0.0,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return AttentionHTR(**kwargs)


def make_batch(batch_size: int = 2, width: int = 160, length: int = 6):
    vocab = make_vocab()
    widths = [width] * batch_size
    images = torch.randn(batch_size, 1, IMG_HEIGHT, width)
    encoder_lengths = torch.tensor([width // 4] * batch_size)
    decoder_input = torch.randint(4, vocab.num_classes, (batch_size, length))
    decoder_input[:, 0] = vocab.sos_index
    targets = torch.randint(4, vocab.num_classes, (batch_size, length))
    targets[:, -1] = vocab.eos_index
    return vocab, images, encoder_lengths, decoder_input, targets


# ------------------------------------------------------------------- 1. CNN
def test_cnn_output_dimensions():
    cnn = CNNEncoder(in_channels=1, img_height=IMG_HEIGHT, dropout=0.0)
    geometry = attention_geometry(IMG_HEIGHT)
    assert geometry["out_height"] == IMG_HEIGHT // 16
    assert cnn.feature_dim == 256 * geometry["out_height"]

    images = torch.randn(3, 1, IMG_HEIGHT, 100)
    sequence = cnn(images)

    assert sequence.dim() == 3
    assert sequence.shape[0] == 3
    assert sequence.shape[1] == 100 // 4          # T = floor(W / 4)
    assert sequence.shape[2] == cnn.feature_dim   # C * H'


def test_cnn_output_length_formula_matches_tensor():
    cnn = CNNEncoder(in_channels=1, img_height=IMG_HEIGHT, dropout=0.0)
    for width in (92, 97, 100, 253, 2514):
        produced = cnn(torch.randn(1, 1, IMG_HEIGHT, width)).shape[1]
        assert produced == cnn.output_length(width)
    assert cnn.output_lengths([92, 1000]) == [23, 250]


def test_cnn_to_sequence_transformation_is_per_position():
    """A change in one image column must stay inside the expected window."""
    cnn = CNNEncoder(in_channels=1, img_height=IMG_HEIGHT, dropout=0.0).eval()
    base = torch.zeros(1, 1, IMG_HEIGHT, 200)
    changed = base.clone()
    changed[:, :, :, 100:] = 1.0  # change the right half only

    with torch.no_grad():
        out_base = cnn(base)[0]     # (T, F)
        out_changed = cnn(changed)[0]

    # The first quarter of the sequence (far away, beyond the receptive field)
    # must be unaffected; the last quarter must change.
    assert torch.allclose(out_base[:10], out_changed[:10], atol=1e-5)
    assert not torch.allclose(out_base[-10:], out_changed[-10:], atol=1e-3)


# --------------------------------------------------------------- 2. BiLSTM
def test_bilstm_encoder_output_and_padding():
    encoder = BiLSTMEncoder(input_size=32, hidden_size=16, num_layers=1, dropout=0.0).eval()
    sequence = torch.randn(2, 10, 32)
    lengths = torch.tensor([10, 4])

    with torch.no_grad():
        outputs = encoder(sequence, lengths)

    assert outputs.shape == (2, 10, 32)  # 2 * hidden
    # Packed sequence => padded time steps are exactly zero.
    assert torch.allclose(outputs[1, 4:], torch.zeros_like(outputs[1, 4:]))
    assert outputs[1, :4].abs().sum() > 0

    full = encoder(sequence)
    assert full.shape == outputs.shape


# -------------------------------------------------------------- 3. attention
def test_attention_score_dimensions():
    attention = BahdanauAttention(encoder_dim=8, decoder_dim=6, attn_dim=5)
    encoder_outputs = torch.randn(4, 9, 8)
    decoder_hidden = torch.randn(4, 6)

    scores = attention.scores(encoder_outputs, decoder_hidden)
    context, weights = attention(encoder_outputs, decoder_hidden)

    assert scores.shape == (4, 9)
    assert weights.shape == (4, 9)
    assert context.shape == (4, 8)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)
    assert (weights >= 0).all()


def test_attention_masks_padded_positions():
    attention = BahdanauAttention(encoder_dim=8, decoder_dim=6, attn_dim=5).eval()
    encoder_outputs = torch.randn(2, 7, 8)
    decoder_hidden = torch.randn(2, 6)
    mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4], dtype=torch.bool)

    with torch.no_grad():
        context, weights = attention(encoder_outputs, decoder_hidden, mask)

    # Padded positions receive exactly zero attention weight ...
    assert weights[1, 3:].abs().max().item() < 1e-7
    assert weights[1, :3].sum().item() == pytest.approx(1.0, abs=1e-5)
    # ... and the context vector is the weighted sum of the *valid* outputs.
    manual = (weights.unsqueeze(-1) * encoder_outputs).sum(dim=1)
    assert torch.allclose(context, manual, atol=1e-6)

    # A mask of the wrong shape is rejected instead of silently broadcasting.
    with pytest.raises(ValueError):
        attention(encoder_outputs, decoder_hidden, mask[:, :5])


# --------------------------------------------------------------- 4. decoder
def test_decoder_output_dimensions():
    vocab = make_vocab()
    decoder = AttentionDecoder(
        vocab_size=vocab.num_classes,
        embed_dim=16,
        hidden_size=32,
        encoder_dim=64,
        attn_dim=16,
        num_layers=1,
        dropout=0.0,
        pad_index=vocab.pad_index,
    )
    encoder_outputs = torch.randn(3, 12, 64)
    mask = torch.ones(3, 12, dtype=torch.bool)
    state = decoder.init_state(3, encoder_outputs.device, encoder_outputs.dtype)
    tokens = torch.randint(4, vocab.num_classes, (3,))

    logits, (hidden, cell), weights = decoder.forward_step(tokens, state, encoder_outputs, mask)

    assert logits.shape == (3, vocab.num_classes)
    assert hidden.shape == (1, 3, 32)
    assert weights.shape == (3, 12)


# --------------------------------------------------------- 5. special tokens
def test_special_token_indices_and_decoding():
    vocab = make_vocab()
    assert (PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX) == (0, 1, 2, 3)
    assert vocab.pad_index == PAD_INDEX
    assert vocab.sos_index == SOS_INDEX
    assert vocab.eos_index == EOS_INDEX
    assert vocab.unk_index == UNK_INDEX
    assert vocab.num_classes == 4 + len(vocab.chars)
    assert all(index >= 4 for index in vocab.encode("abc"))

    ids = vocab.encode("abc")
    assert vocab.decode([SOS_INDEX] + ids + [EOS_INDEX, 0]) == "abc"
    # Unknown characters map to <UNK> and decode as nothing.
    assert vocab.encode("\u00a7") == [UNK_INDEX]
    assert vocab.decode([UNK_INDEX]) == ""
    with pytest.raises(ValueError):
        vocab.encode("\u00a7", unknown_policy="strict")


def test_special_token_save_load(tmp_path):
    vocab = make_vocab()
    path = tmp_path / "attention_vocab.json"
    vocab.save(str(path))
    loaded = AttentionVocabulary.load(str(path))
    assert loaded.num_classes == vocab.num_classes
    assert loaded.encode("abc") == vocab.encode("abc")
    assert loaded.to_dict()["sos_index"] == SOS_INDEX


# -------------------------------------------------------- 6. teacher forcing
def test_teacher_forcing_controls_dependency_on_targets():
    """TF=1: outputs depend on the gold tokens. TF=0: they cannot."""
    vocab, images, lengths, decoder_input, _ = make_batch()
    model = make_model(vocab).eval()

    other = decoder_input.clone()
    other[0, 2] = (other[0, 2] + 1 - 4) % (vocab.num_classes - 4) + 4
    assert not torch.equal(other[0, 2], decoder_input[0, 2])

    with torch.no_grad():
        gold = model(images, decoder_input, lengths, teacher_forcing_ratio=1.0)
        changed = model(images, other, lengths, teacher_forcing_ratio=1.0)
        free_gold = model(images, decoder_input, lengths, teacher_forcing_ratio=0.0)
        free_changed = model(images, other, lengths, teacher_forcing_ratio=0.0)

    # Teacher forcing propagates the changed token to later steps.
    assert not torch.allclose(gold, changed, atol=1e-5)
    # Free running ignores the targets entirely: the model feeds its own
    # predictions, so both runs are identical.
    assert torch.allclose(free_gold, free_changed, atol=1e-6)
    # The first step always consumes <SOS>, so it cannot depend on targets.
    assert torch.allclose(gold[:, 0], changed[:, 0], atol=1e-6)


def test_teacher_forcing_ratio_validation():
    model = make_model()
    model.set_teacher_forcing_ratio(0.3)
    assert model.teacher_forcing_ratio == pytest.approx(0.3)
    with pytest.raises(ValueError):
        model.set_teacher_forcing_ratio(1.5)


# ------------------------------------------------- 7. cross-entropy padding
def test_cross_entropy_ignores_padded_targets():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7)
    targets = torch.tensor([[4, 5, 2, PAD_INDEX, PAD_INDEX]])

    masked = masked_cross_entropy(logits, targets, ignore_index=PAD_INDEX)
    manual = F.cross_entropy(logits[:, :3].reshape(-1, 7), targets[:, :3].reshape(-1))
    naive = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))

    assert torch.allclose(masked, manual, atol=1e-6)
    # A loss that counted the padding would be a different (here smaller) value.
    assert not torch.allclose(masked, naive, atol=1e-6)

    with pytest.raises(ValueError):
        masked_cross_entropy(logits, targets[:, :3])


def test_attention_loss_end_to_end_matches_manual():
    vocab, images, lengths, decoder_input, targets = make_batch()
    model = make_model(vocab).eval()
    batch = {
        "images": images,
        "encoder_lengths": lengths,
        "decoder_input": decoder_input,
        "targets": targets,
        "texts": ["abc"] * 2,
    }

    loss = make_attention_loss(ignore_index=vocab.pad_index)(model, batch, "cpu", False)
    with torch.no_grad():
        logits = model(images, decoder_input, lengths, teacher_forcing_ratio=1.0)
    manual = masked_cross_entropy(logits, targets, ignore_index=vocab.pad_index)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(manual.item(), rel=1e-5)


# ------------------------------------------------------- 8. forward/backward
def test_forward_pass_shapes():
    vocab, images, lengths, decoder_input, _ = make_batch(batch_size=3, length=7)
    model = make_model(vocab).eval()

    with torch.no_grad():
        logits = model(images, decoder_input, lengths)
    assert logits.shape == (3, 7, vocab.num_classes)

    # Omitting the decoder input decodes greedily.
    predicted = model(images, None, lengths, max_length=5)
    assert isinstance(predicted, list) and len(predicted) == 3
    assert all(len(seq) <= 5 for seq in predicted)


def test_backward_pass_produces_finite_gradients():
    vocab, images, lengths, decoder_input, targets = make_batch()
    model = make_model(vocab)
    model.train()

    logits = model(images, decoder_input, lengths, teacher_forcing_ratio=1.0)
    loss = masked_cross_entropy(logits, targets, ignore_index=vocab.pad_index)
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 10
    assert all(torch.isfinite(g).all() for g in grads)
    # The attention projection must actually receive gradient signal.
    assert model.decoder.attention.proj_encoder.weight.grad.abs().sum() > 0


def test_encoder_length_is_validated():
    vocab, images, _, _, _ = make_batch(batch_size=1, width=64)
    model = make_model(vocab).eval()
    too_long = torch.tensor([10_000])
    with pytest.raises(ValueError):
        model.encode(images, too_long)
    with pytest.raises(ValueError):
        model.encode(images, torch.tensor([0]))


# ------------------------------------------------------- 9. greedy inference
def test_greedy_autoregressive_inference():
    vocab, images, lengths, _, _ = make_batch(batch_size=4, width=120)
    model = make_model(vocab).eval()

    with torch.no_grad():
        sequences = model.greedy_decode(images, lengths, max_length=12)

    assert len(sequences) == 4
    for seq in sequences:
        assert isinstance(seq, list) and len(seq) <= 12
        assert all(0 <= token < vocab.num_classes for token in seq)
        if EOS_INDEX in seq:
            assert seq[-1] == EOS_INDEX, "decoding must stop once <EOS> is produced"
        # <SOS> is consumed, never emitted.
        assert SOS_INDEX not in seq
    assert vocab.decode_batch(sequences) is not None


def test_greedy_decode_respects_max_length():
    vocab, images, lengths, _, _ = make_batch(batch_size=2)
    model = make_model(vocab).eval()
    # Force the model to never emit <EOS>: a 1-class vocabulary is impossible,
    # so instead check the hard cap with a model whose eos_index is out of reach.
    model.eos_index = vocab.num_classes + 5
    sequences = model.greedy_decode(images, lengths, max_length=3)
    assert all(len(seq) == 3 for seq in sequences)


# -------------------------------------------------- 10. checkpoint roundtrip
def test_checkpoint_save_and_reload(tmp_path):
    vocab = make_vocab()
    model = make_model(vocab)
    model.eval()

    config = {"model": model.config_dict(), "vocabulary": vocab.to_dict()}
    checkpoint = tmp_path / "attention_smoke.pt"
    torch.save(
        {"epoch": 0, "model_state_dict": model.state_dict(), "config": config},
        checkpoint,
    )

    predictor = AttentionInference(str(checkpoint), device="cpu")
    result = predictor.predict(Image.new("L", (240, 128), color=255))

    assert isinstance(result.text, str)
    assert result.inference_time >= 0
    assert result.model_type == "attention"
    assert predictor.vocabulary.num_classes == vocab.num_classes

    # The reloaded model must reproduce the original weights exactly.
    reloaded = predictor.model.state_dict()
    for key, value in model.state_dict().items():
        assert torch.allclose(reloaded[key].cpu(), value.cpu())


def test_inference_rejects_invalid_images(tmp_path):
    vocab = make_vocab()
    model = make_model(vocab)
    checkpoint = tmp_path / "attention_smoke.pt"
    torch.save(
        {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "config": {"model": model.config_dict(), "vocabulary": vocab.to_dict()},
        },
        checkpoint,
    )
    predictor = AttentionInference(str(checkpoint), device="cpu")

    with pytest.raises(ValueError):
        predictor.predict(Image.new("L", (1, 1)))
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((3, 64, 200), dtype=np.float32))
    with pytest.raises(FileNotFoundError):
        AttentionInference(str(tmp_path / "missing.pt"), device="cpu")


# ------------------------------------------------ 11. variable-width batches
def _sample(width: int, text: str, index: int) -> dict:
    return {
        "image": np.random.RandomState(index).randn(1, IMG_HEIGHT, width).astype(np.float32),
        "text": text,
        "index": index,
    }


def test_collator_pads_variable_widths():
    vocab = make_vocab()
    collator = AttentionCollator(vocab, max_length=16)
    batch = [
        _sample(40, "abc", 0),
        _sample(27, "de", 1),
        _sample(35, "fghij", 2),
    ]

    out = collator(batch)

    assert out["images"].shape == (3, 1, IMG_HEIGHT, 40)
    assert out["widths"] == [40, 27, 35]
    assert out["encoder_lengths"].tolist() == [10, 6, 8]
    # Right padding equals paper white (+1.0), real pixels are untouched.
    assert torch.allclose(out["images"][1, :, :, 27:], torch.ones(1, IMG_HEIGHT, 13))
    assert torch.allclose(
        out["images"][1, :, :, :27], torch.from_numpy(batch[1]["image"]).float()
    )

    # decoder_input = [SOS] + chars, targets = chars + [EOS], padded with PAD.
    assert out["decoder_input"][0, 0].item() == SOS_INDEX
    assert out["targets"][0, 3].item() == EOS_INDEX
    assert out["decoder_input"].shape == out["targets"].shape
    assert out["decoder_input"][1, 3:].eq(PAD_INDEX).all()
    assert out["target_lengths"].tolist() == [4, 3, 6]
    assert out["texts"] == ["abc", "de", "fghij"]


def test_encoder_lengths_helper():
    assert encoder_lengths_from_widths([92, 100, 4]) == [23, 25, 1]
    assert encoder_lengths_from_widths([92], width_downsample=8) == [11]


def test_collator_truncates_long_transcriptions():
    vocab = make_vocab()
    collator = AttentionCollator(vocab, max_length=5)
    out = collator([_sample(40, "abcdefghij", 0)])
    assert out["targets"].shape == (1, 5)
    assert out["targets"][0, -1].item() == EOS_INDEX
    assert out["target_lengths"].tolist() == [5]
