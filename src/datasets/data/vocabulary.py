"""Character vocabulary for CTC-based CRNN.

Responsibilities:
  - Build a sorted character vocabulary from training transcriptions.
  - Reserve index 0 for the CTC blank token.
  - Convert text <-> index sequences.
  - Save / load the vocabulary to JSON for inference.

Index layout
------------
    0        : <blank>   (CTC blank)
    1        : " unk"    (out-of-vocabulary character, optional)
    2 .. V+1 : sorted characters observed in the training text
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Sequence

DEFAULT_BLANK_TOKEN = "<blank>"
DEFAULT_UNK_TOKEN = " unk"


class Vocabulary:
    def __init__(
        self,
        blank_token: str = DEFAULT_BLANK_TOKEN,
        unk_token: str = DEFAULT_UNK_TOKEN,
        use_unk: bool = True,
    ) -> None:
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.use_unk = use_unk

        self._char_to_idx: dict[str, int] = {}
        self._idx_to_char: dict[int, str] = {}
        self._chars: List[str] = []
        self.blank_idx = 0
        self.unk_idx = 1 if use_unk else None
        self._built = False

    # ------------------------------------------------------------------ build
    def build_from_texts(self, texts: Iterable[str]) -> "Vocabulary":
        chars: set[str] = set()
        for text in texts:
            chars.update(str(text))
        return self.build_from_chars(chars)

    def build_from_chars(self, chars: Iterable[str]) -> "Vocabulary":
        self._chars = sorted(set(chars))
        self._build_mappings()
        return self

    def _build_mappings(self) -> None:
        self._char_to_idx = {self.blank_token: 0}
        self._idx_to_char = {0: self.blank_token}
        idx = 1
        if self.use_unk and self.unk_token:
            self._char_to_idx[self.unk_token] = idx
            self._idx_to_char[idx] = self.unk_token
            self.unk_idx = idx
            idx += 1
        else:
            self.unk_idx = None
        for ch in self._chars:
            # Never let a real character shadow the reserved blank/unk slots.
            if ch in self._char_to_idx:
                continue
            self._char_to_idx[ch] = idx
            self._idx_to_char[idx] = ch
            idx += 1
        self._built = True

    # ------------------------------------------------------------- properties
    @property
    def blank_index(self) -> int:
        return 0

    @property
    def unk_index(self) -> int | None:
        return self.unk_idx

    @property
    def num_classes(self) -> int:
        """Number of output classes, including the CTC blank."""
        return len(self._char_to_idx)

    @property
    def chars(self) -> List[str]:
        return list(self._chars)

    def __len__(self) -> int:
        return self.num_classes

    # ------------------------------------------------------------- encoding
    def text_to_indices(self, text: str, unknown_policy: str = "skip") -> List[int]:
        """Convert a string to a list of character indices.

        unknown_policy:
            "skip"   - drop out-of-vocabulary characters (default, safe for CTC)
            "map"    - map to the ``unk`` index (requires use_unk=True)
            "strict" - raise ValueError on unknown character
        """
        if not self._built:
            raise RuntimeError("Vocabulary has not been built yet.")

        indices: List[int] = []
        for ch in text:
            if ch in self._char_to_idx:
                indices.append(self._char_to_idx[ch])
            elif unknown_policy == "map":
                if self.unk_idx is None:
                    raise ValueError(
                        f"Character {ch!r} is out of vocabulary and use_unk=False."
                    )
                indices.append(self.unk_idx)
            elif unknown_policy == "strict":
                raise ValueError(
                    f"Character {ch!r} is out of vocabulary (vocab size={self.num_classes})."
                )
            else:  # "skip"
                continue
        return indices

    def encode_batch(
        self, texts: Sequence[str], unknown_policy: str = "skip"
    ) -> tuple[List[int], List[int]]:
        """Encode a batch of texts into CTCLoss target format.

        Returns
        -------
        flat_targets : list[int]
            Concatenated indices of all texts (CTCLoss target format).
        target_lengths : list[int]
            Length of each text in indices.
        """
        flat: List[int] = []
        lengths: List[int] = []
        for text in texts:
            indices = self.text_to_indices(text, unknown_policy=unknown_policy)
            lengths.append(len(indices))
            flat.extend(indices)
        return flat, lengths

    # ------------------------------------------------------------- decoding
    def indices_to_text(self, indices: Sequence[int], keep_repeats: bool = False) -> str:
        """Convert a frame-level index sequence back to text.

        Implements the CTC collapse rule: blanks separate repeated characters,
        and consecutive identical predictions that are not separated by a blank
        are collapsed into a single character.

        Parameters
        ----------
        keep_repeats : bool
            If True, do not collapse consecutive identical predictions. This is
            only useful for debugging; normal decoding uses False.
        """
        chars: List[str] = []
        prev_idx: int | None = None
        for idx in indices:
            idx = int(idx)
            if idx == self.blank_index:
                prev_idx = None
                continue
            ch = self._idx_to_char.get(idx)
            if ch is None or ch == self.blank_token:
                prev_idx = None
                continue
            if keep_repeats or idx != prev_idx:
                chars.append(ch)
            prev_idx = idx
        return "".join(chars)

    def decode_batch(self, batches: Sequence[Sequence[int]]) -> List[str]:
        return [self.indices_to_text(seq) for seq in batches]

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        data = {
            "blank_token": self.blank_token,
            "unk_token": self.unk_token,
            "use_unk": self.use_unk,
            "chars": self._chars,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            blank_token=data.get("blank_token", DEFAULT_BLANK_TOKEN),
            unk_token=data.get("unk_token", DEFAULT_UNK_TOKEN),
            use_unk=data.get("use_unk", True),
        )
        vocab.build_from_chars(data["chars"])
        return vocab
