"""Reusable image preprocessing pipelines for CRNN and TrOCR.

Design principles:
  - Configurable via simple dataclasses / kwargs.
  - CRNN: grayscale -> aspect-ratio-preserving resize to fixed height ->
    normalization -> variable width (padding handled in collate).
  - TrOCR: defer to the HF processor (size/normalization) to avoid
    double-preprocessing.
  - Augmentations are mild and handwriting-safe (rotation, translation,
    affine, brightness/contrast). No cropping that could cut off characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class CRNNPreprocessConfig:
    img_height: int = 64
    img_width: Optional[int] = None
    grayscale: bool = True
    mean: float = 0.5
    std: float = 0.5
    enabled: bool = True
    rotation_degrees: float = 2.0
    translate: float = 0.02
    scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = 0.2
    contrast: float = 0.2


class CRNNPreprocessor:
    def __init__(self, config: CRNNPreprocessConfig) -> None:
        self.cfg = config

    def __call__(self, image: Image.Image, training: bool = False) -> np.ndarray:
        img = self._ensure_pil(image)
        if self.cfg.grayscale and img.mode != "L":
            img = img.convert("L")
        elif not self.cfg.grayscale and img.mode != "RGB":
            img = img.convert("RGB")

        if training and self.cfg.enabled:
            img = self._augment(img)

        img = self._resize_preserve_aspect(img)
        arr = self._to_normalized_array(img)
        return arr

    @staticmethod
    def _ensure_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _augment(self, img: Image.Image) -> Image.Image:
        angle = np.random.uniform(-self.cfg.rotation_degrees, self.cfg.rotation_degrees)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

        arr = np.array(img)
        h, w = arr.shape[:2]
        scale = np.random.uniform(self.cfg.scale[0], self.cfg.scale[1])
        tx = np.random.uniform(-self.cfg.translate, self.cfg.translate) * w
        ty = np.random.uniform(-self.cfg.translate, self.cfg.translate) * h
        M = np.float32([[scale, 0, tx], [0, scale, ty]])
        arr = cv2.warpAffine(
            arr, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255,
        )
        img = Image.fromarray(arr)

        if self.cfg.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.cfg.brightness, self.cfg.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.cfg.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.cfg.contrast, self.cfg.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        return img

    def _resize_preserve_aspect(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == self.cfg.img_height:
            return img
        new_w = max(1, int(round(w * self.cfg.img_height / h)))
        return img.resize((new_w, self.cfg.img_height), Image.BICUBIC)

    def _to_normalized_array(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        arr = (arr / 255.0 - self.cfg.mean) / self.cfg.std
        return arr


@dataclass
class TrOCRPreprocessConfig:
    force_rgb: bool = True
    pre_resize: Optional[int] = None


class TrOCRPreprocessor:
    def __init__(self, processor, config: TrOCRPreprocessConfig) -> None:
        self.processor = processor
        self.cfg = config

    def __call__(self, image: Image.Image) -> np.ndarray:
        img = self._ensure_pil(image)
        if self.cfg.force_rgb and img.mode != "RGB":
            img = img.convert("RGB")
        if self.cfg.pre_resize is not None:
            img = self._resize_longest_edge(img, self.cfg.pre_resize)
        inputs = self.processor(images=img, return_tensors="np")
        pixel_values = inputs["pixel_values"]
        return pixel_values[0].astype(np.float32)

    @staticmethod
    def _ensure_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _resize_longest_edge(img: Image.Image, max_edge: int) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_edge:
            return img
        scale = max_edge / max(w, h)
        return img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC
        )


def pil_to_np_gray(image: Image.Image) -> np.ndarray:
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image, dtype=np.float32) / 255.0