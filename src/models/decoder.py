"""CTC decoding utilities: greedy decoding and (optional) beam search.

Greedy decoding:
    - Take argmax over classes at each time step.
    - Remove blanks and collapse repeated characters (CTC rules).

Beam search decoding:
    - Prefix beam search over the log-probability sequence.
    - Simplified (no language model) but correct for the CTC blank/repeat rules.
"""

from __future__ import annotations

import math
from typing import List

import torch


def greedy_decode(
    log_probs: torch.Tensor,
    blank_index: int = 0,
    repeat_collapse: bool = True,
) -> List[List[int]]:
    """Greedy (best-path) CTC decoding.

    Parameters
    ----------
    log_probs : FloatTensor (T, B, num_classes) log-probabilities.
    blank_index : int index of the CTC blank token.
    repeat_collapse : whether to collapse consecutive repeated non-blank labels.

    Returns
    -------
    list of list of int : decoded label indices per batch element (blanks removed).
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be (T, B, C), got {log_probs.shape}")
    T, B, _ = log_probs.shape
    max_indices = log_probs.argmax(dim=-1)

    decoded: List[List[int]] = []
    for b in range(B):
        labels: List[int] = []
        prev: int | None = None
        for t in range(T):
            idx = int(max_indices[t, b])
            if idx == blank_index:
                prev = None
                continue
            if repeat_collapse and prev is not None and idx == prev:
                continue
            labels.append(idx)
            prev = idx
        decoded.append(labels)
    return decoded


def decode_logits_to_indices(
    logits: torch.Tensor,
    blank_index: int = 0,
) -> List[List[int]]:
    """Convenience: log_softmax then greedy decode."""
    log_probs = torch.log_softmax(logits, dim=-1)
    return greedy_decode(log_probs, blank_index=blank_index)

# ---------------------------------------------------------------------------
# Beam search (prefix search, no external LM)
# ---------------------------------------------------------------------------


class _BeamEntry:
    __slots__ = ("prefix", "score_blank", "score_nonblank", "total_score")

    def __init__(self) -> None:
        self.prefix: tuple[int, ...] = ()
        self.score_blank = 0.0  # log prob ending in blank
        self.score_nonblank = -math.inf  # log prob ending in non-blank
        self.total_score = -math.inf

    @property
    def score(self) -> float:
        return self.total_score


def beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 5,
    blank_index: int = 0,
) -> List[List[int]]:
    """Prefix beam search CTC decoding (no language model).

    Parameters
    ----------
    log_probs : FloatTensor (T, B, num_classes)
    beam_width : number of beams to keep.
    blank_index : index of CTC blank.

    Returns
    -------
    list of list of int : best decoded label indices per batch element.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must be (T, B, C), got {log_probs.shape}")
    T, B, C = log_probs.shape
    log_probs = log_probs.detach().cpu()

    results: List[List[int]] = []
    for b in range(B):
        best = _beam_search_single(log_probs[:, b, :], beam_width, blank_index)
        results.append(list(best.prefix))
    return results


def _beam_search_single(
    seq_log_probs: torch.Tensor,  # (T, C)
    beam_width: int,
    blank_index: int,
) -> _BeamEntry:
    T, C = seq_log_probs.shape

    blank_entry = _BeamEntry()
    blank_entry.prefix = ()
    blank_entry.score_blank = 0.0
    blank_entry.score_nonblank = -math.inf
    blank_entry.total_score = 0.0

    beams: dict[tuple[int, ...], _BeamEntry] = {(): blank_entry}

    for t in range(T):
        next_beams: dict[tuple[int, ...], _BeamEntry] = {}

        def get_entry(prefix: tuple[int, ...]) -> _BeamEntry:
            if prefix not in next_beams:
                next_beams[prefix] = _BeamEntry()
                next_beams[prefix].prefix = prefix
                next_beams[prefix].score_blank = -math.inf
                next_beams[prefix].score_nonblank = -math.inf
            return next_beams[prefix]

        for prefix, entry in beams.items():
            last_char = prefix[-1] if len(prefix) > 0 else None

            e = get_entry(prefix)
            entry_total = _logsumexp(entry.score_blank, entry.score_nonblank)
            e.score_blank = _logsumexp(
                e.score_blank, entry_total + seq_log_probs[t, blank_index]
            )

            for c in range(C):
                if c == blank_index:
                    continue
                new_prefix = prefix + (c,)
                e2 = get_entry(new_prefix)
                if len(prefix) > 0 and last_char == c:
                    e2.score_nonblank = _logsumexp(
                        e2.score_nonblank, entry.score_blank + seq_log_probs[t, c]
                    )
                else:
                    entry_total = _logsumexp(entry.score_blank, entry.score_nonblank)
                    e2.score_nonblank = _logsumexp(
                        e2.score_nonblank, entry_total + seq_log_probs[t, c]
                    )

        scored = [
            (e, _logsumexp(e.score_blank, e.score_nonblank)) for e in next_beams.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {}
        for e, _ in scored[:beam_width]:
            e.total_score = _logsumexp(e.score_blank, e.score_nonblank)
            beams[e.prefix] = e

    best = max(
        beams.values(), key=lambda e: _logsumexp(e.score_blank, e.score_nonblank)
    )
    return best


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf and b == -math.inf:
        return -math.inf
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))