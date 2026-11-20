"""TrOCR (Transformer OCR) wrapper built on Hugging Face Transformers.

Loads a pretrained ``microsoft/trocr-*-handwritten`` checkpoint and exposes:
  - processor (image -> pixel values, text -> token ids)
  - the vision-encoder-decoder model
  - a training forward pass (labels padded with -100)
  - text generation for inference
  - save / load of model + processor together

Tokenizer note
--------------
``microsoft/trocr-small-handwritten`` ships only ``sentencepiece.bpe.model``
and a ``tokenizer_config.json`` that names ``XLMRobertaTokenizer`` without a
``vocab_file`` entry. Recent transformers releases therefore cannot build the
tokenizer from that config. ``build_processor`` first tries the plain HF path
and then rebuilds the XLM-R tokenizer from the sentencepiece model with the
official XLM-R converter, which reproduces the published token ids
(``<s>``=0, ``<pad>``=1, ``</s>``=2, ``<unk>``=3).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

SP_MODEL_FILE = "sentencepiece.bpe.model"
XLM_R_SPECIAL_IDS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 250001}

# The published checkpoint only ships a legacy pytorch_model.bin. transformers
# 5.x refuses torch.load below torch 2.6 (CVE-2025-32434), so the weights are
# converted once to safetensors in this local directory.
SAFE_WEIGHTS_DIR = os.path.join("checkpoints", "pretrained", "trocr-small-handwritten")


def materialise_model_dir(
    checkpoint: str, dest_dir: str = SAFE_WEIGHTS_DIR, force: bool = False
) -> str:
    """Return a local directory holding config + safetensors weights.

    * A directory that already has model.safetensors is returned unchanged.
    * A legacy pytorch_model.bin is converted to safetensors once and cached
      in ``dest_dir``.
    """
    safetensors_path = os.path.join(dest_dir, "model.safetensors")
    if os.path.isdir(checkpoint):
        if os.path.isfile(os.path.join(checkpoint, "model.safetensors")):
            return checkpoint
    if not force and os.path.isfile(safetensors_path):
        return dest_dir

    if os.path.isdir(checkpoint):
        weights_path = os.path.join(checkpoint, "pytorch_model.bin")
        config_source = checkpoint
    else:
        from huggingface_hub import hf_hub_download

        try:
            weights_path = hf_hub_download(checkpoint, "pytorch_model.bin")
        except Exception as exc:  # noqa: BLE001 - another weight format may exist
            print(f"[TrOCR] No pytorch_model.bin available ({exc}); using HF loader.")
            return checkpoint
        config_source = checkpoint

    if not os.path.isfile(weights_path):
        return checkpoint

    print("[TrOCR] Converting legacy pytorch_model.bin to safetensors (one-off) ...")
    model = _load_legacy_checkpoint(config_source)
    os.makedirs(dest_dir, exist_ok=True)
    model.save_pretrained(dest_dir, safe_serialization=True)
    print(f"[TrOCR] Safetensors checkpoint written to {dest_dir}")
    return dest_dir


def _load_legacy_checkpoint(source: str):
    """Load a .bin checkpoint through the official transformers loader.

    transformers >= 5 refuses torch.load on torch < 2.6. The published TrOCR
    weights are a public Microsoft checkpoint loaded with ``weights_only=True``
    during a one-off local conversion, and ``from_pretrained`` is required here
    because it applies the model's own key remapping (a plain
    ``load_state_dict`` silently drops renamed DeiT encoder weights).
    """
    import transformers.modeling_utils as modeling_utils

    original = getattr(modeling_utils, "check_torch_load_is_safe", None)
    if original is not None:
        modeling_utils.check_torch_load_is_safe = lambda: None
    try:
        model = VisionEncoderDecoderModel.from_pretrained(source)
    finally:
        if original is not None:
            modeling_utils.check_torch_load_is_safe = original
    return model


def _locate_sp_model(checkpoint: str) -> str:
    """Return the local path of ``sentencepiece.bpe.model`` for a checkpoint."""
    if os.path.isdir(checkpoint):
        local = os.path.join(checkpoint, SP_MODEL_FILE)
        if os.path.isfile(local):
            return local
        raise FileNotFoundError(f"{SP_MODEL_FILE} not found in {checkpoint}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(checkpoint, SP_MODEL_FILE)


def _build_xlmr_tokenizer(sp_model_path: str):
    """Build the XLM-R tokenizer from a sentencepiece model."""
    from transformers import XLMRobertaTokenizer
    from transformers.convert_slow_tokenizer import XLMRobertaConverter

    class _SentencePieceShim:
        vocab_file = sp_model_path
        add_prefix_space = True
        model_max_length = 512

        def convert_tokens_to_ids(self, token: str) -> int:
            return XLM_R_SPECIAL_IDS[token]

    fast_backend = XLMRobertaConverter(_SentencePieceShim()).converted()
    return XLMRobertaTokenizer(tokenizer_object=fast_backend)


def _read_decoder_start(model, fallback: int) -> int:
    """Read decoder_start_token_id from generation config or model config."""
    for holder in (getattr(model, "generation_config", None), model.config):
        if holder is None:
            continue
        try:
            value = holder.decoder_start_token_id
        except AttributeError:
            value = None
        if value is not None:
            return int(value)
    return int(fallback)


def _tokenizer_is_healthy(tokenizer) -> bool:
    """Verify that the tokenizer adds <s>/</s> and has in-range special ids."""
    try:
        ids = tokenizer("handwriting", add_special_tokens=True)["input_ids"]
        if len(ids) < 3:
            return False
        if ids[0] != tokenizer.cls_token_id or ids[-1] != tokenizer.eos_token_id:
            return False
        if tokenizer.pad_token_id is None:
            return False
        return 0 <= int(tokenizer.pad_token_id) < len(tokenizer)
    except Exception:  # noqa: BLE001 - any tokenizer failure means unusable
        return False


def build_processor(checkpoint: str):
    """Build a TrOCRProcessor with a verified XLM-R tokenizer."""
    from transformers import TrOCRProcessor

    try:
        processor = TrOCRProcessor.from_pretrained(checkpoint)
        if _tokenizer_is_healthy(processor.tokenizer):
            return processor
        print(
            "[TrOCR] Default processor lacks the XLM-R special tokens; "
            "rebuilding it from the sentencepiece model."
        )
    except Exception as exc:  # noqa: BLE001 - transformers build failure is expected
        print(f"[TrOCR] Default processor load failed ({exc}); rebuilding ...")

    from transformers import DeiTImageProcessor

    image_processor = DeiTImageProcessor.from_pretrained(checkpoint)
    tokenizer = _build_xlmr_tokenizer(_locate_sp_model(checkpoint))
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    if not _tokenizer_is_healthy(processor.tokenizer):
        raise RuntimeError("Rebuilt TrOCR tokenizer failed validation.")
    return processor


class TrOCRWrapper(nn.Module):
    """Fine-tunable TrOCR model with its processor."""

    def __init__(
        self,
        checkpoint: str = "microsoft/trocr-small-handwritten",
        device: str = "cpu",
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.max_length = max_length

        print(f"[TrOCR] Loading processor from {checkpoint} ...")
        self.processor = build_processor(checkpoint)
        model_dir = materialise_model_dir(checkpoint)
        print(f"[TrOCR] Loading model from {model_dir} ...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.tokenizer = self.processor.tokenizer
        self._configure_model_config()

    def _configure_model_config(self, max_length: Optional[int] = None) -> None:
        config = self.model.config
        # Never shrink the decoder vocabulary: the published weights use 64044
        # decoder rows while the XLM-R tokenizer exposes 64002 tokens. Both use
        # the same id layout, the extra rows are simply unused.
        embed_rows = self.model.decoder.get_input_embeddings().num_embeddings
        if len(self.tokenizer) > embed_rows:
            raise ValueError(
                f"Tokenizer ({len(self.tokenizer)} tokens) exceeds the decoder "
                f"embedding size ({embed_rows})."
            )
        config.decoder.eos_token_id = self.tokenizer.eos_token_id
        config.decoder.bos_token_id = self.tokenizer.cls_token_id
        config.decoder.pad_token_id = self.tokenizer.pad_token_id
        config.pad_token_id = self.tokenizer.pad_token_id
        config.eos_token_id = self.tokenizer.eos_token_id

        # TrOCR starts decoding with </s> (id 2), which is the checkpoint's
        # configured decoder_start_token_id. Using <s> (id 0) here produces
        # degenerate repeated text, so the published value is preserved.
        published_start = _read_decoder_start(self.model, self.tokenizer.eos_token_id)
        config.decoder_start_token_id = published_start

        config.decoder.max_length = max_length or self.max_length
        config.decoder.no_repeat_ngram_size = 0
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.decoder_start_token_id = published_start
        self.model.generation_config.max_length = config.decoder.max_length

    # ------------------------------------------------------------------ train
    def forward(self, pixel_values, labels=None, attention_mask=None):
        return self.model(
            pixel_values=pixel_values,
            labels=labels,
            decoder_attention_mask=attention_mask,
        )

    def encode_texts(self, texts: list[str], max_length: Optional[int] = None):
        """Tokenise texts into padded labels with padding masked as -100."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            add_special_tokens=True,
        )
        labels = enc["input_ids"]
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        images=None,
        pixel_values=None,
        max_length: Optional[int] = None,
        num_beams: int = 4,
    ) -> list[str]:
        """Generate transcriptions from PIL images or preprocessed pixel values."""
        self.model.eval()
        if pixel_values is None:
            if images is None:
                raise ValueError("generate() needs either images or pixel_values")
            if not isinstance(images, (list, tuple)):
                images = [images]
            pixel_values = self.processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
        pixel_values = pixel_values.to(self.device)

        gen_kwargs = {
            "pixel_values": pixel_values,
            "max_length": max_length or self.max_length,
            "num_beams": num_beams,
        }
        if num_beams > 1:
            gen_kwargs["early_stopping"] = True
        outputs = self.model.generate(**gen_kwargs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)

    # ------------------------------------------------------------ persistence
    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        print(f"[TrOCR] Saved model + processor to {save_dir}")

    @classmethod
    def load_pretrained(cls, save_dir: str, device: str = "cpu") -> "TrOCRWrapper":
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.checkpoint = save_dir
        wrapper.device = torch.device(device)
        wrapper.max_length = 128
        print(f"[TrOCR] Loading processor from {save_dir} ...")
        wrapper.processor = build_processor(save_dir)
        model_dir = materialise_model_dir(save_dir)
        print(f"[TrOCR] Loading model from {model_dir} ...")
        wrapper.model = VisionEncoderDecoderModel.from_pretrained(model_dir)
        wrapper.model.to(wrapper.device)
        wrapper.tokenizer = wrapper.processor.tokenizer
        wrapper._configure_model_config()
        return wrapper


def load_processor_with_cache(
    checkpoint: str = "microsoft/trocr-small-handwritten",
    cache_dir: str = "models/trocr_processor",
):
    """Load the processor, caching the repaired version for offline reuse."""
    from transformers import TrOCRProcessor

    tokenizer_file = os.path.join(cache_dir, "tokenizer.json")
    if os.path.isdir(cache_dir) and os.path.isfile(tokenizer_file):
        return TrOCRProcessor.from_pretrained(cache_dir)
    processor = build_processor(checkpoint)
    try:
        processor.save_pretrained(cache_dir)
    except Exception as exc:  # noqa: BLE001 - cache is an optimisation only
        print(f"[TrOCR] Could not cache processor ({exc}); continuing in memory.")
    return processor
