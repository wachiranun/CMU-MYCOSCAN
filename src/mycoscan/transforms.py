"""Image preprocessing.

Colony images, when configured: plate-circle crop, before anything else.
Both splits: optional autocontrast (brightness/contrast normalization across
microscope cameras and smartphones, one luminance stretch so hue is kept),
resize, ImageNet normalization.
Training only: the augmentation recipe (none | standard | trivial_wide). Hue is
never changed because pigment colour is diagnostic (e.g. the red diffusible
pigment of Talaromyces marneffei on the reverse of the colony).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
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


class HueSafeTrivialAugmentWide(T.TrivialAugmentWide):
    """TrivialAugment-Wide without the ops that shift hue: Solarize inverts channels,
    Posterize quantises them, and Equalize and AutoContrast stretch each channel separately.
    Brightness, Color (saturation) and Contrast blend towards black or grey, which keeps hue."""

    HUE_SHIFTING = ("Solarize", "Posterize", "Equalize", "AutoContrast")

    def _augmentation_space(self, num_bins: int):
        space = super()._augmentation_space(num_bins)
        for op in self.HUE_SHIFTING:
            space.pop(op)
        return space


def _otsu(gray: np.ndarray) -> float:
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(float)
    levels = np.arange(256)
    below = np.cumsum(hist)
    above = below[-1] - below
    mean_below = np.cumsum(hist * levels) / np.maximum(below, 1)
    mean_above = ((hist * levels).sum() - np.cumsum(hist * levels)) / np.maximum(above, 1)
    return float(np.argmax(below * above * (mean_below - mean_above) ** 2))


def plate_circle_crop(img: Image.Image, min_radius: float = 0.25) -> Image.Image:
    """Crop a colony photo to the Petri plate and black out everything outside its circle.

    The plate is the largest region on the other side of an Otsu threshold from the image border,
    with holes (the colony) filled; its area gives the radius. An image with no such region of
    radius >= min_radius * the shorter side (no plate in frame) is returned unchanged.
    """
    gray = np.asarray(img.convert("L"), dtype=float)
    fg = gray > _otsu(gray)
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    if border.mean() > 0.5:
        fg = ~fg
    fg = ndimage.binary_fill_holes(fg)
    labels, n = ndimage.label(fg)
    if n == 0:
        return img
    sizes = ndimage.sum(fg, labels, range(1, n + 1))
    plate = labels == int(np.argmax(sizes)) + 1
    radius = float(np.sqrt(plate.sum() / np.pi))
    if radius < min_radius * min(gray.shape):
        return img
    cy, cx = ndimage.center_of_mass(plate)
    box = tuple(round(v) for v in (cx - radius, cy - radius, cx + radius, cy + radius))
    crop = img.crop(box)
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, crop.size[0] - 1, crop.size[1] - 1], fill=255)
    return Image.composite(crop, Image.new("RGB", crop.size), mask)


def load_image(source: Image.Image | str | Path, plate_crop: bool, modality: str) -> Image.Image:
    """8-bit RGB, cropped to the plate when plate_crop is on and the image is a colony photo.
    Training, evaluation, Predictor and explain all load images here, so they crop alike."""
    if isinstance(source, Image.Image):
        rgb = to_rgb(source)
    else:
        with Image.open(source) as img:
            rgb = to_rgb(img)
    return plate_circle_crop(rgb) if plate_crop and modality == "colony" else rgb


def build_transform(image_size: int, autocontrast: bool, train: bool, augmentation: str = "standard") -> T.Compose:
    """augmentation applies to training only: none (the validation transform), standard, or trivial_wide."""
    steps: list = [AutoContrast()] if autocontrast else []
    if train and augmentation == "standard":
        steps += [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            RandomRightAngleRotation(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.0),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        ]
    elif train and augmentation == "trivial_wide":
        steps += [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            HueSafeTrivialAugmentWide(),
        ]
    else:
        steps += [T.Resize(image_size), T.CenterCrop(image_size)]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)
