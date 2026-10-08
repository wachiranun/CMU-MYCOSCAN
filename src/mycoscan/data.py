"""Datasets and loaders. Augmentation is attached only to the training loader."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .splits import held_out_mask
from .transforms import TileSpec, build_transform, load_image, tile_crop


class ImageDataset(Dataset):
    """One item per row: the image, or with `tiles` the row's tile of it."""

    def __init__(self, paths: list[str], labels: list[int], transform, modalities: list[str], plate_crop: bool = False,
                 tiles: TileSpec | None = None, tile_index: list[int] | None = None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.modalities = modalities
        self.plate_crop = plate_crop
        self.tiles = tiles
        self.tile_index = tile_index

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_image(self.paths[i], self.plate_crop, self.modalities[i])
        if self.tiles is not None and self.tile_index is not None:
            img = tile_crop(img, self.tiles, self.tile_index[i])
        return self.transform(img), self.labels[i]


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


def refuse_held_out(df: pd.DataFrame) -> None:
    """A training loader never sees an external-test image."""
    held_out = held_out_mask(df)
    if held_out.any():
        paths = df.loc[held_out, "image_path"].tolist()
        raise ValueError(f"training loader refused {len(paths)} held-out rows (Pool B or test): {', '.join(paths[:10])}")


def make_loader(df: pd.DataFrame, class_to_idx: dict[str, int], image_size: int, autocontrast: bool,
                train: bool, batch_size: int, num_workers: int, imbalance: str = "none", seed: int = 0,
                augmentation: str = "standard", plate_crop: bool = False, tiles: TileSpec | None = None) -> DataLoader:
    """With `tiles`, each row is the tile of its image given by its `tile` column."""
    if train:
        refuse_held_out(df)
    # Prediction ignores labels, and an external set's reference labels need not be model classes.
    labels = [class_to_idx[s] if train else class_to_idx.get(s, -1) for s in df["species"]]
    ds = ImageDataset(df["image_path"].tolist(), labels, build_transform(image_size, autocontrast, train, augmentation),
                      df["modality"].tolist(), plate_crop, tiles, df["tile"].astype(int).tolist() if tiles else None)
    generator = torch.Generator().manual_seed(seed)
    if train and imbalance == "sampler":
        weights = balanced_sample_weights(df["species"], df["group"])
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=generator)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=len(ds) > batch_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=generator, num_workers=num_workers,
                      drop_last=train and len(ds) > batch_size)
