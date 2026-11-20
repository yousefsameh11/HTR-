"""CRNN + BiLSTM + CTC model for handwritten text recognition.

Architecture:
  Input image (1, H, W)
    -> CNN feature extractor (conv+bn+relu+pool x3)  => (C, H/8, W/8)
    -> reshape to sequence (B, W/8, C*H/8)
    -> Bidirectional LSTM(s)
    -> Fully connected layer to num_classes (incl. blank)
    -> logits, decoded with CTC

Handles variable-width inputs. Returns logits of shape (T, B, num_classes)
(time-major) as expected by torch.nn.CTCLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(
        self,
        img_height: int = 64,
        num_classes: int = 80,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        dropout: float = 0.3,
        in_channels: int = 1,
        cnn_downsample_factor: int = 8,
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.cnn_downsample_factor = cnn_downsample_factor
        self.in_channels = in_channels
        self.rnn_type = rnn_type.lower()
        # FP16 convolution path (optional). The recurrent layers stay in FP32:
        # cuDNN's FP16 LSTM path aborted the process on this Windows/CUDA stack.
        self.use_fp16_cnn = False

        cnn_out_height = img_height // cnn_downsample_factor
        if cnn_out_height < 1:
            raise ValueError(
                f"img_height {img_height} too small for downsample factor {cnn_downsample_factor}"
            )
        self.cnn_out_height = cnn_out_height

        # ------------------------------------------------------------------ CNN
        self.cnn = nn.Sequential(
            # block 1 -> /2
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2 -> /4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 3 -> /8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

        # ------------------------------------------------------------------ RNN
        # Feature map: (B, 256, H/8, W/8) -> treat width as time.
        rnn_input_size = 256 * cnn_out_height
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )

        # ------------------------------------------------------------------ FC
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        images : FloatTensor (B, C, H, W)

        Returns
        -------
        logits : FloatTensor (T, B, num_classes) -- time-major for CTCLoss
        """
        if self.use_fp16_cnn and images.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = self.cnn(images)
            features = features.float()
        else:
            features = self.cnn(images)  # (B, 256, H/8, W/8)
        b, c, h, w = features.shape

        # (B, W/8, H/8, C) -> (B, T, C*H/8)
        features = features.permute(0, 3, 2, 1).contiguous()
        features = features.view(b, w, c * h)

        rnn_out, _ = self.rnn(features)  # (B, T, hidden*2)
        logits = self.fc(rnn_out)  # (B, T, num_classes)

        return logits.permute(1, 0, 2)  # (T, B, num_classes)

    def set_fp16_cnn(self, enabled: bool) -> None:
        """Enable/disable the FP16 fast path for the convolutional stack."""
        self.use_fp16_cnn = bool(enabled)

    def get_output_lengths(self, input_lengths) -> list[int]:
        """Compute CNN output sequence lengths from image widths.

        Three stride-2 pools => ``floor(W / 8)`` valid time steps per sample.
        """
        if isinstance(input_lengths, torch.Tensor):
            input_lengths = input_lengths.tolist()
        factor = max(1, int(self.cnn_downsample_factor))
        return [max(1, int(w) // factor) for w in input_lengths]

    def config_dict(self) -> dict:
        """Constructor arguments, stored inside checkpoints for exact reloads."""
        return {
            "img_height": self.img_height,
            "num_classes": self.num_classes,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "dropout": 0.0,
            "in_channels": self.in_channels,
            "cnn_downsample_factor": self.cnn_downsample_factor,
            "rnn_type": self.rnn_type,
        }
