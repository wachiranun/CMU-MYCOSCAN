"""Bags: the instances one prediction is pooled from.

A bag is all images of one isolate (`bag = "isolate"`), its images from one device
(`"isolate_device"`), or the tiles of one high-resolution image (`"tiles"`). Bags are
built inside a group, so a bag never mixes groups and never straddles a split.

`pooling` (`mean` or `max`) turns instance predictions into the bag prediction. Both
are parameter-free, so training stays single-instance (single images, or single tiles)
and pooling is applied when predicting, at validation and at evaluation alike; `mean`
over an isolate's images is the mean-of-images vote the pipeline always used.

A bag batch is a padded tensor (bags, instances, ...) with a mask marking the real
instances. Pooling reads the mask, so padding contributes nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import TileSpec, load_image, tile_crop

BAG_MODES = ("none", "isolate", "isolate_device", "tiles")
POOLINGS = ("mean", "max")


def pool(x: torch.Tensor, mask: torch.Tensor, how: str) -> tuple[torch.Tensor, torch.Tensor]:
    """x (bags, instances, features) and mask (bags, instances) -> pooled (bags, features) and each
    instance's weight (bags, instances): 1/n for mean, for max the share of features the instance supplied.
    Masked instances get weight 0."""
    weights_mask = mask.to(x.dtype)
    if how == "mean":
        weights = weights_mask / weights_mask.sum(dim=1, keepdim=True)
        return (weights.unsqueeze(-1) * torch.where(mask.unsqueeze(-1), x, 0)).sum(dim=1), weights
    if how == "max":
        pooled, which = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1)
        supplied = torch.zeros_like(weights_mask).scatter_add_(1, which, torch.ones_like(which, dtype=x.dtype))
        return pooled, supplied / x.shape[-1]
    raise ValueError(f"pooling {how!r}; expected one of {list(POOLINGS)}")


def pool_probabilities(probs: torch.Tensor, mask: torch.Tensor, how: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled class probabilities of each bag, renormalised to sum to 1 (a max over instances does not)."""
    pooled, weights = pool(probs, mask, how)
    return pooled / pooled.sum(dim=-1, keepdim=True), weights


def collate_bags(items: list[tuple[torch.Tensor, int, dict]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """(instances, label, meta) per bag -> padded instances, mask, labels and the metas, in order."""
    largest = max(len(x) for x, _, _ in items)
    first = items[0][0]
    x = first.new_zeros((len(items), largest, *first.shape[1:]))
    mask = torch.zeros(len(items), largest, dtype=torch.bool)
    for b, (bag, _, _) in enumerate(items):
        x[b, :len(bag)] = bag
        mask[b, :len(bag)] = True
    return x, mask, torch.tensor([label for _, label, _ in items]), [meta for _, _, meta in items]


def with_bag_ids(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """df with a `bag` column: its group (isolate), its group and device, or its image (tiles).
    Every bag id starts from something inside one group, so no bag mixes groups."""
    if mode == "isolate":
        return df.assign(bag=df["group"].astype(str))
    if mode == "isolate_device":
        return df.assign(bag=df["group"].astype(str) + "|" + df["device"].astype(str))
    if mode == "tiles":
        return df.assign(bag=df["image_path"].astype(str))
    raise ValueError(f"bag {mode!r}; expected one of {list(BAG_MODES[1:])}")


def instances(df: pd.DataFrame, mode: str, tiles: TileSpec | None) -> pd.DataFrame:
    """One row per instance with its `bag`: the image rows, or for tile bags each image's rows repeated once
    per tile with a `tile` index."""
    rows = with_bag_ids(df.reset_index(drop=True), mode)
    if mode != "tiles":
        return rows.reset_index(drop=True)
    if tiles is None:
        raise ValueError("bag='tiles' needs a tile grid")
    return rows.loc[rows.index.repeat(tiles.count)].assign(
        tile=np.tile(np.arange(tiles.count), len(rows))).reset_index(drop=True)


def assert_no_bag_leakage(df: pd.DataFrame, fold) -> None:
    """Every bag inside one group, and all of a bag's rows on one side of the fold."""
    groups_per_bag = df.groupby("bag")["group"].nunique()
    mixed = groups_per_bag[groups_per_bag > 1].index.tolist()
    if mixed:
        raise AssertionError(f"bags mixing groups: {mixed[:5]}")
    shared = set(df["bag"].iloc[fold.train_idx]) & set(df["bag"].iloc[fold.val_idx])
    if shared:
        raise AssertionError(f"{fold.name}: bags in both train and val: {sorted(shared)[:5]}")


class BagDataset(Dataset):
    """One item per bag of `rows` (from `instances`): its instances as one tensor, the class index
    (-1 for a label outside `class_to_idx`) and {"bag", "rows"}, the positions of its rows in `rows`."""

    def __init__(self, rows: pd.DataFrame, class_to_idx: dict[str, int], transform, plate_crop: bool = False,
                 tiles: TileSpec | None = None):
        self.rows = rows.reset_index(drop=True)
        self.bags = list(self.rows.groupby("bag", sort=False).indices.items())
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.plate_crop = plate_crop
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, i: int):
        name, positions = self.bags[i]
        loaded: dict[str, Image.Image] = {}  # the tiles of a bag share one image; load it once
        xs = []
        for p in positions:
            row = self.rows.iloc[p]
            if row["image_path"] not in loaded:
                loaded[row["image_path"]] = load_image(row["image_path"], self.plate_crop, row["modality"])
            img = loaded[row["image_path"]]
            xs.append(self.transform(tile_crop(img, self.tiles, int(row["tile"])) if self.tiles else img))
        label = self.class_to_idx.get(self.rows["species"].iat[positions[0]], -1)
        return torch.stack(xs), label, {"bag": name, "rows": positions}


@torch.no_grad()
def predict_bags(model: torch.nn.Module, rows: pd.DataFrame, transform, pooling: str, device: str, batch_size: int,
                 plate_crop: bool = False, tiles: TileSpec | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Each instance's class probabilities (in the order of `rows`), and each bag's name and pooled
    probabilities. Instances are classified one by one; only the pooling sees the bag."""
    dataset = BagDataset(rows, {}, transform, plate_crop, tiles)
    largest = max(len(positions) for _, positions in dataset.bags)
    loader = DataLoader(dataset, batch_size=max(1, batch_size // largest), collate_fn=collate_bags)
    model.eval()
    instance_probs: np.ndarray | None = None
    names, pooled_probs = [], []
    for x, mask, _, metas in loader:
        flat = x[mask]
        probs = torch.cat([model(flat[i:i + batch_size].to(device)).softmax(dim=1).cpu()
                           for i in range(0, len(flat), batch_size)])
        padded = probs.new_zeros((*mask.shape, probs.shape[1]))
        padded[mask] = probs
        pooled, _ = pool_probabilities(padded, mask, pooling)
        if instance_probs is None:
            instance_probs = np.zeros((len(rows), probs.shape[1]), dtype=np.float32)
        for b, meta in enumerate(metas):
            instance_probs[meta["rows"]] = padded[b, :len(meta["rows"])].numpy()
            names.append(meta["bag"])
        pooled_probs.append(pooled.numpy())
    assert instance_probs is not None
    return instance_probs, names, np.concatenate(pooled_probs)


def bag_frame(rows: pd.DataFrame, names: list[str], columns: list[str]) -> pd.DataFrame:
    """One row per bag, in the order of `names`: each column's value where the bag's rows agree, else empty,
    and the bag's instance count."""
    by_bag = rows.groupby("bag", sort=False)
    out = by_bag[columns].agg(lambda col: col.iat[0] if col.nunique(dropna=False) == 1 else None)
    out["n_instances"] = by_bag.size()
    return out.loc[names].reset_index()
