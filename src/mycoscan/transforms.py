"""Image preprocessing.

Both splits: optional autocontrast (brightness/contrast normalization across
microscope cameras and smartphones, one luminance stretch so hue is kept),
resize, ImageNet normalization.
Training only: geometric and photometric augmentation. Hue is never jittered
because pigment colour is diagnostic (e.g. the red diffusible pigment of
Talaromyces marneffei on the reverse of the colony).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb(img: Image.Image) -> Image.Image:
    """8-bit RGB from any camera output: 16-bit and float frames are min-max scaled,
    transparency is composited over white."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(a)
        img = Image.fromarray(a.astype(np.uint8))
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


class AutoContrast:
    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(img, cutoff=1, preserve_tone=True)


class RandomRightAngleRotation:
    def __call__(self, img: Image.Image) -> Image.Image:
        return img.rotate(90 * random.randint(0, 3))


def build_transform(image_size: int, autocontrast: bool, train: bool) -> T.Compose:
    steps: list = [AutoContrast()] if autocontrast else []
    if train:
        steps += [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        ]
    else:
        steps += [T.Resize(image_size), T.CenterCrop(image_size)]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)
