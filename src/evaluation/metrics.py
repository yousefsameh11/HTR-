"""Evaluation metrics for handwritten text recognition: CER and WER.

Both models are scored with the same text normalisation so the comparison is
fair. CER and WER are reported at corpus level (total edits / total reference
units), which is the standard HTR reporting convention; per-sample values are
also returned for error analysis.

jiwer 4.x is used for word-level alignment (``jiwer.wer`` / ``jiwer.process_words``).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import jiwer


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans({c: " " for c in "!\"#$%&()*+,./:;<=>?@[\\]^_`{|}~"})


def normalize_text(text: str) -> str:
    """Normalise a transcription consistently for both models.

    Lowercase, drop punctuation, collapse whitespace. Letters, digits and the
    apostrophe are kept because IAM contains contractions.
    """
    text = str(text).strip().lower()
    text = text.replace("-", " ")
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_cer(text: str) -> str:
    """Character sequence used for CER (spaces removed)."""
    return normalize_text(text).replace(" ", "")


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Classic Levenshtein distance over any sequences of hashable items."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                current[j - 1] + 1,          # insertion
                previous[j] + 1,             # deletion
                previous[j - 1] + cost,      # substitution
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Per-sample Character Error Rate = edit_distance / len(reference chars)."""
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Per-sample Word Error Rate."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level CER: sum of character edits / sum of reference characters."""
    total_edits = 0
    total_chars = 0
    for ref, hyp in zip(references, hypotheses):
        ref_c, hyp_c = normalize_for_cer(ref), normalize_for_cer(hyp)
        total_edits += _levenshtein(ref_c, hyp_c)
        total_chars += len(ref_c)
    return total_edits / total_chars if total_chars else 0.0


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level WER via jiwer (total word edits / total reference words)."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    if not refs:
        return 0.0
    return float(jiwer.wer(refs, hyps))


# ---------------------------------------------------------------------------
# Aggregated API used by evaluation scripts and the Streamlit app
# ---------------------------------------------------------------------------

def compute_cer_wer(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> Tuple[float, float, List[dict]]:
    """Compute corpus CER/WER plus per-sample details.

    Returns
    -------
    (cer_score, wer_score, details)
    where details is a list of dicts with reference, hypothesis, cer and wer.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references ({len(references)}) and hypotheses ({len(hypotheses)}) differ in length"
        )
    details: List[dict] = []
    for ref, hyp in zip(references, hypotheses):
        details.append(
            {
                "reference": ref,
                "hypothesis": hyp,
                "cer": cer(ref, hyp),
                "wer": wer(ref, hyp),
            }
        )
    return corpus_cer(references, hypotheses), corpus_wer(references, hypotheses), details


def jiwer_measures(references: Sequence[str], hypotheses: Sequence[str]) -> dict:
    """Word-level substitution/deletion/insertion/hit counts (corpus level)."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    if not refs:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    output = jiwer.process_words(refs, hyps)
    return {
        "wer": float(output.wer),
        "substitutions": int(output.substitutions),
        "deletions": int(output.deletions),
        "insertions": int(output.insertions),
        "hits": int(output.hits),
    }
