"""Datasets and loaders. Augmentation is attached only to the training loader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .transforms import build_transform, to_rgb


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            return self.transform(to_rgb(img)), self.labels[i]


def balanced_sample_weights(species: pd.Series, group: pd.Series) -> np.ndarray:
    """Per-image weights so every class, and every group within a class, is drawn equally often.

    A group is an isolate (CMU) or a plate's pseudo-group (OpenFungi). One photographed
    30 times must not outweigh one photographed 5 times, and a class with 3 groups
    must not outweigh a class with 2.
    """
    groups_per_class = species.map(group.groupby(species).nunique())
    images_per_group = group.map(group.value_counts())
    return np.array(1.0 / (groups_per_class * images_per_group), dtype=float)


def class_loss_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(df: pd.DataFrame, class_to_idx: dict[str, int], image_size: int, autocontrast: bool,
                train: bool, batch_size: int, num_workers: int, imbalance: str = "none", seed: int = 0) -> DataLoader:
    labels = [class_to_idx[s] for s in df["species"]]
    ds = ImageDataset(df["image_path"].tolist(), labels, build_transform(image_size, autocontrast, train))
    generator = torch.Generator().manual_seed(seed)
    if train and imbalance == "sampler":
        weights = balanced_sample_weights(df["species"], df["group"])
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=generator)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=len(ds) > batch_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=generator, num_workers=num_workers,
                      drop_last=train and len(ds) > batch_size)
