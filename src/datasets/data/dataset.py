"""PyTorch Dataset wrapping the IAM line-level Hugging Face dataset.

Responsibilities:
  - Load / cache the Teklia/IAM-line dataset.
  - Validate every sample (non-empty transcription).
  - Expose train / validation / test splits.
  - Apply CRNN preprocessing per sample.
  - Optionally limit samples for fast CPU smoke tests.
  - TrOCR dataset + collator (dynamic label padding, pad masked with -100).
"""

from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

from .preprocessing import CRNNPreprocessor

DATASET_NAME = "Teklia/IAM-line"


class IAMLineDataset(Dataset):
    """Map-style PyTorch dataset over one HF split of IAM-line."""

    def __init__(
        self,
        hf_dataset,
        preprocessor: CRNNPreprocessor,
        vocabulary=None,
        text_column: str = "text",
        image_column: str = "image",
        unknown_policy: str = "skip",
    ) -> None:
        self.hf_dataset = hf_dataset
        self.preprocessor = preprocessor
        self.vocabulary = vocabulary
        self.text_column = text_column
        self.image_column = image_column
        self.unknown_policy = unknown_policy
        self._training = False
        self._validate()

    def _validate(self) -> None:
        """Validate the split cheaply (text column only).

        Decoding all 6.5k line images up front costs minutes and gives nothing:
        images are decoded lazily in ``__getitem__``.
        """
        text_column = self.hf_dataset[self.text_column]
        missing = sum(
            1 for value in text_column if value is None or str(value).strip() == ""
        )
        if missing:
            raise ValueError(f"{missing} sample(s) in this split have no transcription.")
        if self.image_column not in self.hf_dataset.column_names:
            raise KeyError(
                f"Image column {self.image_column!r} not in {self.hf_dataset.column_names}"
            )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.hf_dataset[int(idx)]
        text = str(sample[self.text_column])
        image = sample[self.image_column]

        img_arr = self.preprocessor(
            image, training=self.preprocessor.cfg.enabled and self._training
        )

        target = None
        target_length = 0
        if self.vocabulary is not None:
            target = self.vocabulary.text_to_indices(
                text, unknown_policy=self.unknown_policy
            )
            target_length = len(target)

        return {
            "image": img_arr,
            "text": text,
            "target": target,
            "target_length": target_length,
            "index": int(idx),
        }

    def set_training(self, training: bool) -> None:
        self._training = training

    @property
    def training(self) -> bool:
        return getattr(self, "_training", False)


def load_iam_splits(cache_dir: Optional[str] = None):
    """Load Teklia/IAM-line splits: train 6482 / validation 976 / test 2915."""
    return load_dataset(DATASET_NAME, cache_dir=cache_dir)


def build_iam_datasets(
    preprocessor: CRNNPreprocessor,
    vocabulary=None,
    text_column: str = "text",
    image_column: str = "image",
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Build train/val/test IAMLineDataset instances with optional sample caps.

    The dataset's own writer-independent splits are preserved: no sample is ever
    moved between train/validation/test.
    """
    hf = load_iam_splits(cache_dir=cache_dir)

    train_split, val_split, test_split = hf["train"], hf["validation"], hf["test"]
    if max_train is not None:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    if max_val is not None:
        val_split = val_split.select(range(min(max_val, len(val_split))))
    if max_test is not None:
        test_split = test_split.select(range(min(max_test, len(test_split))))

    train_ds = IAMLineDataset(
        train_split, preprocessor, vocabulary, text_column, image_column
    )
    val_ds = IAMLineDataset(
        val_split, preprocessor, vocabulary, text_column, image_column
    )
    test_ds = IAMLineDataset(
        test_split, preprocessor, vocabulary, text_column, image_column
    )

    train_ds.set_training(True)
    val_ds.set_training(False)
    test_ds.set_training(False)
    return train_ds, val_ds, test_ds


class TrOCRDataset(Dataset):
    """TrOCR dataset: processor-ready pixel values + unpadded token labels.

    Padding is done per batch by ``TrOCRCollator`` with -100, which is both
    faster and matches the loss mask expected by VisionEncoderDecoderModel.
    """

    def __init__(
        self,
        hf_split,
        processor,
        max_length: int = 128,
        image_column: str = "image",
        text_column: str = "text",
    ) -> None:
        self.hf_split = hf_split
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, idx: int) -> dict:
        sample = self.hf_split[int(idx)]
        image = sample[self.image_column]
        text = str(sample[self.text_column])

        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ][0]
        labels = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )["input_ids"][0]

        return {"pixel_values": pixel_values, "labels": labels, "text": text}


class TrOCRCollator:
    """Pad a batch of TrOCR labels to the longest sequence, masking with -100."""

    def __init__(self, tokenizer, pad_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(int(f["labels"].shape[0]) for f in features)
        input_ids = torch.full(
            (len(features), max_len), self.pad_token_id, dtype=torch.long
        )
        for i, feature in enumerate(features):
            seq = feature["labels"]
            input_ids[i, : seq.shape[0]] = seq

        labels = input_ids.masked_fill(input_ids == self.pad_token_id, -100)

        # Only model inputs are returned: extra keys (e.g. raw text) would be
        # forwarded to model.generate() during evaluation and rejected there.
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features], dim=0),
            "labels": labels,
        }
