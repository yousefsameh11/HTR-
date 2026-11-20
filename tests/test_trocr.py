"""TrOCR tests: processor integrity, label preparation, forward pass, generation.

These load the real pretrained checkpoint (cached by Hugging Face) and are
therefore slower than the CRNN tests. Set HTR_SKIP_TROCR=1 to skip them.
"""

from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("HTR_SKIP_TROCR") == "1",
    reason="HTR_SKIP_TROCR=1",
)

CHECKPOINT = "microsoft/trocr-small-handwritten"


@pytest.fixture(scope="module")
def wrapper():
    from src.models.trocr import TrOCRWrapper

    return TrOCRWrapper(CHECKPOINT, device="cpu")


def test_processor_special_tokens(wrapper):
    tokenizer = wrapper.tokenizer
    ids = tokenizer("put down a resolution", add_special_tokens=True)["input_ids"]

    # XLM-R convention required by the pretrained TrOCR checkpoint
    assert tokenizer.cls_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert ids[0] == 0 and ids[-1] == 2

    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert text.strip().lower().startswith("put down")


def test_decoder_start_token_is_published_value(wrapper):
    # TrOCR decodes from </s> (2). Overriding it with <s> (0) yields repeated
    # garbage, so this must stay at the checkpoint's value.
    assert wrapper.tokenizer.eos_token_id == 2
    assert wrapper.model.generation_config.decoder_start_token_id == 2


def test_label_preparation_masks_padding(wrapper):
    labels = wrapper.encode_texts(["hello", "a much longer transcription"])

    assert labels.shape[0] == 2
    assert labels.shape[1] >= labels.shape[0]
    # every sample keeps its own EOS token, padding is -100
    for row in labels:
        eos_positions = (row == wrapper.tokenizer.eos_token_id).nonzero().flatten()
        assert eos_positions.numel() == 1
        pad_positions = (row == -100).nonzero().flatten()
        if pad_positions.numel():
            assert pad_positions.min() > eos_positions.max()


def test_forward_backward_and_generation(wrapper):
    images = [
        Image.new("RGB", (300, 100), color="white"),
        Image.new("RGB", (220, 90), color="black"),
    ]
    pixel_values = wrapper.processor(images=images, return_tensors="pt")["pixel_values"]
    labels = wrapper.encode_texts(["hello world", "second line"])

    wrapper.model.train()
    out = wrapper(pixel_values=pixel_values, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in wrapper.model.parameters())

    wrapper.model.zero_grad(set_to_none=True)
    wrapper.tokenizer = wrapper.processor.tokenizer
    predictions = wrapper.generate(images=images, max_length=12, num_beams=1)
    assert isinstance(predictions, list) and len(predictions) == 2
    assert all(isinstance(p, str) for p in predictions)
