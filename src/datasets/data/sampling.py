"""Width-bucketed batching for variable-width line images.

``crnn_collate_fn`` pads every batch to the widest line in it, so with fully
random shuffling each batch is computed at the width of its widest (often 1.5x
average) sample. Sorting into wide buckets and then shuffling inside the bucket
keeps batches width-homogeneous while still mixing content every epoch.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class WidthBucketBatchSampler(Sampler):
    """Yield batches whose members have similar widths.

    Args:
        widths: per-sample image width in pixels (already resized to model height).
        batch_size: samples per batch.
        bucket_size_batches: how many consecutive batches form one sorted bucket.
            Larger buckets -> tighter widths (faster) but less content mixing.
        shuffle: shuffle bucket order, samples inside a bucket and batch order.
        drop_last: drop an incomplete final batch of a bucket.
        seed: RNG seed; a fresh epoch index is mixed in automatically.
    """

    def __init__(
        self,
        widths: Sequence[int],
        batch_size: int,
        bucket_size_batches: int = 50,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.widths = [int(w) for w in widths]
        self.batch_size = int(batch_size)
        self.bucket_size = max(1, int(bucket_size_batches)) * self.batch_size
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        order = sorted(range(len(self.widths)), key=lambda i: self.widths[i])
        buckets = [
            order[i : i + self.bucket_size]
            for i in range(0, len(order), self.bucket_size)
        ]
        if self.shuffle:
            rng.shuffle(buckets)

        batches: list[list[int]] = []
        for bucket in buckets:
            bucket = list(bucket)
            if self.shuffle:
                rng.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()
        self.epoch += 1

    def __len__(self) -> int:
        n = len(self.widths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
