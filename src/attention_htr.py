"""CNN + BiLSTM + Bahdanau attention + autoregressive LSTM decoder for HTR.

This is the project's third architecture. It is deliberately independent of
``crnn.py`` (CTC, no attention, frame-synchronous) and ``trocr.py`` (pretrained
Transformer encoder-decoder), while consuming exactly the same preprocessed
line images.

Data flow
---------
    image (B, 1, H, W)
      -> CNNEncoder        feature map (B, C, H/16, W/4) -> sequence (B, T, C*H/16)
      -> BiLSTMEncoder     (B, T, 2 * encoder_hidden), padding ignored via packing
      -> BahdanauAttention additive scores at every decoder step, padded frames
                           masked out with -inf before the softmax
      -> AttentionDecoder  LSTM cell-style step: [embedding(y_{t-1}); context_t]
      -> Linear output     character distribution over the vocabulary
      -> masked CrossEntropyLoss (PAD ignored) during training,
         greedy autoregressive decoding at inference time

Why the dimensions are what they are
------------------------------------
* Height: two stride-2 height pools (H -> H/4) followed by two height-only
  pools (H/4 -> H/16). With the project's 64 px lines the feature map is
  4 rows high, so 256 channels x 4 rows = 1024 features per time step.
* Width: two stride-2 width pools give T = floor(W / 4). Keeping W/4 (instead
  of the CRNN's W/8) preserves more horizontal detail, which the attention
  decoder can exploit: IAM lines are up to 2514 px wide at 64 px height, i.e.
  T = 628 at most.
* The CNN output length is ``floor(W / 4)`` exactly, because every convolution
  uses ``padding=1`` (size preserving) and every pooling uses
  ``ceil_mode=False``. ``CNNEncoder.output_length()`` is the single source of
  truth for that formula; ``AttentionCollator`` uses the same helper.

Residual blocks were left out on purpose: at 1.5 M parameters the model is
already small enough that the extra layers would only cost VRAM on the 4 GB
laptop GPU this project targets.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CNN encoder
# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    """Lightweight convolutional front-end producing a sequence of features.

    ``(B, 1, H, W) -> (B, T, C * H')`` with ``T = floor(W / 4)`` and
    ``H' = H // 16`` for the default 64 px line height.
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_height: int = 64,
        dropout: float = 0.1,
        width_downsample: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.img_height = int(img_height)
        self.width_downsample = int(width_downsample)
        if img_height // 16 < 1:
            raise ValueError(
                f"img_height {img_height} is too small: the CNN reduces the height by 16."
            )

        self.features = nn.Sequential(
            # H/2, W/2
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # H/4, W/4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # H/4, W/4 (wider receptive field, no extra pooling in width)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8
            # H/16, W/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16
            nn.Dropout2d(dropout),
        )

        self.out_channels = 256
        self.out_height = max(1, self.img_height // 16)
        self.feature_dim = self.out_channels * self.out_height

    # -- geometry ---------------------------------------------------------
    def output_length(self, width: int) -> int:
        """Number of time steps produced for an image of ``width`` pixels."""
        return max(1, int(width) // self.width_downsample)

    def output_lengths(self, widths: Sequence[int]) -> List[int]:
        return [self.output_length(w) for w in widths]

    # -- forward ----------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)  # (B, C, H', W')
        b, c, h, w = feature_map.shape
        # (B, C, H', W') -> (B, W', C, H') -> (B, W', C*H')
        sequence = feature_map.permute(0, 3, 1, 2).contiguous()
        return sequence.view(b, w, c * h)


# ---------------------------------------------------------------------------
# BiLSTM encoder
# ---------------------------------------------------------------------------
class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM over the visual sequence.

    When ``lengths`` is given the sequence is packed, so the recurrent layers
    never consume padded time steps and the outputs at padded positions are
    exactly zero.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_size = self.hidden_size * 2

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )

    def forward(
        self, sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            outputs, _ = self.rnn(sequence)
            return outputs

        lengths = lengths.to(device="cpu", dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=sequence.size(1)
        )
        return outputs


# ---------------------------------------------------------------------------
# Bahdanau (additive) attention
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """Additive attention over encoder outputs, with padding masked out.

    ``score_t = v^T tanh(W_e h_t + W_d s)``  --  Bahdanau et al. (2015)
    ``alpha   = softmax(score over t, padded t masked to -inf)``
    ``context = sum_t alpha_t h_t``
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        attn_dim = int(attn_dim or decoder_dim)
        self.encoder_dim = int(encoder_dim)
        self.decoder_dim = int(decoder_dim)
        self.attn_dim = attn_dim

        self.proj_encoder = nn.Linear(self.encoder_dim, attn_dim, bias=False)
        self.proj_decoder = nn.Linear(self.decoder_dim, attn_dim, bias=True)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def scores(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Raw (un-normalised) energies, shape ``(B, T)``."""
        enc_proj = self.proj_encoder(encoder_outputs)                 # (B, T, A)
        dec_proj = self.proj_decoder(decoder_hidden).unsqueeze(1)     # (B, 1, A)
        energies = torch.tanh(enc_proj + dec_proj)
        return self.score(energies).squeeze(-1)                       # (B, T)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context (B, E), weights (B, T))``.

        ``mask`` is boolean with True for valid encoder positions. Masked
        positions receive exactly zero attention weight because their energy is
        replaced by the smallest representable float before the softmax.
        """
        energies = self.scores(encoder_outputs, decoder_hidden)
        if mask is not None:
            if mask.shape != energies.shape:
                raise ValueError(
                    f"attention mask {tuple(mask.shape)} does not match "
                    f"energies {tuple(energies.shape)}"
                )
            energies = energies.masked_fill(~mask.to(torch.bool), torch.finfo(energies.dtype).min)

        weights = torch.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


# ---------------------------------------------------------------------------
# LSTM decoder with attention context
# ---------------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    """Single-step LSTM decoder: embedding + attention + output projection."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        encoder_dim: int,
        attn_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_index: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.encoder_dim = int(encoder_dim)
        self.num_layers = int(num_layers)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=pad_index)
        self.attention = BahdanauAttention(self.encoder_dim, self.hidden_size, attn_dim)
        self.lstm = nn.LSTM(
            input_size=self.embed_dim + self.encoder_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.hidden_size + self.encoder_dim, self.vocab_size)

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward_step(
        self,
        tokens: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Decode one time step.

        Parameters
        ----------
        tokens : LongTensor (B,) previously generated / teacher-forced token.
        state  : (h, c) with shape (num_layers, B, hidden).

        Returns
        -------
        logits : (B, vocab_size)
        state  : updated (h, c)
        weights: attention weights (B, T) for inspection / visualisation
        """
        hidden, cell = state
        # Attention is computed from the top-layer decoder state of the
        # previous step, before consuming the current token (Bahdanau order).
        context, weights = self.attention(encoder_outputs, hidden[-1], mask)

        embedded = self.dropout(self.embedding(tokens))                    # (B, E)
        step_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)   # (B, 1, E+C)
        out, (hidden, cell) = self.lstm(step_input, (hidden, cell))
        out = out.squeeze(1)                                               # (B, hidden)

        logits = self.output(self.dropout(torch.cat([out, context], dim=-1)))
        return logits, (hidden, cell), weights


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class AttentionHTR(nn.Module):
    """CNN + BiLSTM + Bahdanau attention + autoregressive character decoder.

    No CTC is used anywhere: supervision is a masked token-level
    cross-entropy on the decoder outputs, with ``<SOS>``/``<EOS>`` delimiters
    and ``<PAD>`` ignored by the loss.
    """

    def __init__(
        self,
        vocab_size: int,
        img_height: int = 64,
        in_channels: int = 1,
        cnn_dropout: float = 0.1,
        encoder_hidden: int = 128,
        encoder_layers: int = 1,
        decoder_hidden: int = 128,
        decoder_layers: int = 1,
        embed_dim: int = 128,
        attn_dim: Optional[int] = 128,
        dropout: float = 0.2,
        pad_index: int = 0,
        sos_index: int = 1,
        eos_index: int = 2,
        unk_index: int = 3,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.img_height = int(img_height)
        self.pad_index = int(pad_index)
        self.sos_index = int(sos_index)
        self.eos_index = int(eos_index)
        self.unk_index = int(unk_index)
        self.encoder_hidden = int(encoder_hidden)
        self.encoder_layers = int(encoder_layers)
        self.decoder_hidden = int(decoder_hidden)
        self.decoder_layers = int(decoder_layers)
        self.embed_dim = int(embed_dim)
        self.attn_dim = int(attn_dim) if attn_dim is not None else None
        self.dropout_rate = float(dropout)
        self.cnn_dropout = float(cnn_dropout)
        self.in_channels = int(in_channels)

        self.cnn = CNNEncoder(
            in_channels=self.in_channels, img_height=self.img_height, dropout=self.cnn_dropout
        )
        self.encoder = BiLSTMEncoder(
            input_size=self.cnn.feature_dim,
            hidden_size=self.encoder_hidden,
            num_layers=self.encoder_layers,
            dropout=dropout,
        )
        self.decoder = AttentionDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.decoder_hidden,
            encoder_dim=self.encoder.output_size,
            attn_dim=self.attn_dim,
            num_layers=self.decoder_layers,
            dropout=dropout,
            pad_index=self.pad_index,
        )

        # Teacher-forcing probability for the *next* training pass. It is a plain
        # attribute (not a buffer) so it never enters the state dict; training
        # code sets it once per epoch, mirroring CRNN.set_fp16_cnn().
        self.teacher_forcing_ratio = 1.0
        # Mixed precision is applied to the convolutional stack only. Full-model
        # FP16 aborted the process inside cuDNN's LSTM kernels on this Windows /
        # CUDA stack (measured for the CRNN run, see DEVELOPMENT_LOG.md), and the
        # attention model has two LSTMs, so the same restriction applies.
        self.use_fp16_cnn = False

    # ------------------------------------------------------------ utilities
    def set_teacher_forcing_ratio(self, ratio: float) -> None:
        if not 0.0 <= float(ratio) <= 1.0:
            raise ValueError(f"teacher forcing ratio must be in [0, 1], got {ratio}")
        self.teacher_forcing_ratio = float(ratio)

    def set_fp16_cnn(self, enabled: bool) -> None:
        """Enable/disable the FP16 fast path for the convolutional stack."""
        self.use_fp16_cnn = bool(enabled)

    def encoder_lengths_from_widths(self, widths: Sequence[int]) -> torch.Tensor:
        """CNN output length for each image width (``floor(W / 4)``)."""
        return torch.tensor(self.cnn.output_lengths(widths), dtype=torch.long)

    def config_dict(self) -> dict:
        """Constructor arguments, stored in checkpoints for exact reloads."""
        return {
            "vocab_size": self.vocab_size,
            "img_height": self.img_height,
            "in_channels": self.in_channels,
            "cnn_dropout": self.cnn_dropout,
            "encoder_hidden": self.encoder_hidden,
            "encoder_layers": self.encoder_layers,
            "decoder_hidden": self.decoder_hidden,
            "decoder_layers": self.decoder_layers,
            "embed_dim": self.embed_dim,
            "attn_dim": self.attn_dim,
            "dropout": self.dropout_rate,
            "pad_index": self.pad_index,
            "sos_index": self.sos_index,
            "eos_index": self.eos_index,
            "unk_index": self.unk_index,
        }

    # ------------------------------------------------------------- encoding
    def encode(
        self,
        images: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Run the CNN + BiLSTM stack.

        Returns ``(encoder_outputs (B, T, D), mask (B, T) bool, T)``.
        """
        if self.use_fp16_cnn and images.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sequence = self.cnn(images)
            sequence = sequence.float()
        else:
            sequence = self.cnn(images)
        total_steps = int(sequence.size(1))

        if encoder_lengths is None:
            mask = torch.ones(
                sequence.size(0), total_steps, dtype=torch.bool, device=sequence.device
            )
            outputs = self.encoder(sequence)
            return outputs, mask, total_steps

        lengths = encoder_lengths.to(dtype=torch.long)
        if lengths.dim() != 1 or lengths.numel() != sequence.size(0):
            raise ValueError(
                f"encoder_lengths must be (B,) with B={sequence.size(0)}, got {tuple(lengths.shape)}"
            )
        if int(lengths.min()) < 1:
            raise ValueError("encoder_lengths must be >= 1 for every sample")
        if int(lengths.max()) > total_steps:
            raise ValueError(
                f"encoder length {int(lengths.max())} exceeds the {total_steps} frames the "
                "CNN produced for this batch; check the width formula."
            )

        outputs = self.encoder(sequence, lengths)
        steps = torch.arange(total_steps, device=sequence.device).unsqueeze(0)
        mask = steps < lengths.to(device=sequence.device).unsqueeze(1)
        return outputs, mask, total_steps

    # ------------------------------------------------------------ training
    def forward(
        self,
        images: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
        encoder_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: Optional[float] = None,
        max_length: int = 128,
    ):
        """Teacher-forced (or free-running) pass.

        ``decoder_input`` is ``(B, L)`` and already starts with ``<SOS>``; the
        returned logits are ``(B, L, vocab_size)`` and position ``t`` predicts
        ``targets[:, t]`` (a character, or ``<EOS>`` on the last step).

        If ``decoder_input`` is None the model decodes greedily and returns the
        list of predicted id sequences instead (convenience for inference and
        for tests).
        """
        encoder_outputs, mask, _ = self.encode(images, encoder_lengths)

        if decoder_input is None:
            return self._greedy_from_encoder(encoder_outputs, mask, max_length=max_length)

        ratio = self.teacher_forcing_ratio if teacher_forcing_ratio is None else float(
            teacher_forcing_ratio
        )
        batch_size, steps = decoder_input.shape
        state = self.decoder.init_state(batch_size, images.device, encoder_outputs.dtype)

        tokens = decoder_input[:, 0]
        step_logits: List[torch.Tensor] = []
        for step in range(steps):
            logits, state, _ = self.decoder.forward_step(tokens, state, encoder_outputs, mask)
            step_logits.append(logits)
            if step + 1 < steps:
                if ratio >= 1.0:
                    tokens = decoder_input[:, step + 1]
                elif ratio <= 0.0:
                    tokens = logits.argmax(dim=-1)
                else:
                    # Per-sample Bernoulli choice: teacher forcing where the
                    # random draw succeeds, the model's own prediction otherwise.
                    use_gold = torch.rand(batch_size, device=logits.device) < ratio
                    tokens = torch.where(use_gold, decoder_input[:, step + 1], logits.argmax(dim=-1))

        return torch.stack(step_logits, dim=1)

    # ------------------------------------------------------------- decoding
    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        encoder_lengths: Optional[torch.Tensor] = None,
        max_length: int = 128,
    ) -> List[List[int]]:
        """Autoregressive greedy decoding, stopping at ``<EOS>`` per sample."""
        encoder_outputs, mask, _ = self.encode(images, encoder_lengths)
        return self._greedy_from_encoder(encoder_outputs, mask, max_length=max_length)

    @torch.no_grad()
    def _greedy_from_encoder(
        self,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
        max_length: int = 128,
    ) -> List[List[int]]:
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        state = self.decoder.init_state(batch_size, device, encoder_outputs.dtype)

        tokens = torch.full((batch_size,), self.sos_index, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        outputs: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(int(max_length)):
            logits, state, _ = self.decoder.forward_step(tokens, state, encoder_outputs, mask)
            next_tokens = logits.argmax(dim=-1)

            for i in range(batch_size):
                if not bool(finished[i]):
                    token = int(next_tokens[i])
                    outputs[i].append(token)
                    if token == self.eos_index:
                        finished[i] = True

            if bool(finished.all()):
                break
            # Keep stepping for unfinished samples; finished ones get <PAD>.
            tokens = next_tokens.masked_fill(finished, self.pad_index)

        return outputs

    # --------------------------------------------------------------- misc
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def attention_geometry(img_height: int = 64, width_downsample: int = 4) -> dict:
    """Feature-map geometry, used by tests and documentation."""
    out_height = max(1, img_height // 16)
    return {
        "img_height": img_height,
        "out_height": out_height,
        "out_channels": 256,
        "feature_dim": 256 * out_height,
        "width_downsample": width_downsample,
    }
