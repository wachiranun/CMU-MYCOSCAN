"""Frozen backbone features, and a cache so a backbone embeds each image once.

Features are the backbone's pooled output just before its classification head,
from the evaluation transform. The model comes from `models.build_model`, the
same registry training uses, so any timm arch, its pretrained weights, or an
earlier-stage checkpoint can be embedded.

The cache holds one feature vector per image path, in a file named by a key of
everything that changes a feature: the manifest's hash, the arch, its weights,
the image size, autocontrast and the plate crop. A changed manifest is a new
key, so old features are never reused on new data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import make_loader
from .models import PRETRAINED
from .provenance import sha256_file


@torch.no_grad()
def extract_features(model: nn.Module, df: pd.DataFrame, image_size: int, autocontrast: bool, plate_crop: bool,
                     device: str, batch_size: int = 32) -> np.ndarray:
    """One row per image of `df`. `model` is a timm model; its head is skipped."""
    loader = make_loader(df, {}, image_size, autocontrast, train=False, batch_size=batch_size, num_workers=0,
                         plate_crop=plate_crop)
    model.eval().to(device)
    forward_features, forward_head = getattr(model, "forward_features"), getattr(model, "forward_head")
    return np.concatenate([forward_head(forward_features(x.to(device)), pre_logits=True).cpu().numpy()
                           for x, _ in loader]).astype(np.float32)


def cache_key(manifest_sha256: str, arch: str, weights: str, image_size: int, autocontrast: bool,
              plate_crop: bool) -> str:
    weights_id = weights if weights in (*PRETRAINED, "none") else sha256_file(weights)
    fields = {"manifest_sha256": manifest_sha256, "arch": arch, "weights": weights_id, "image_size": image_size,
              "autocontrast": autocontrast, "plate_crop": plate_crop}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def cached_features(df: pd.DataFrame, cache_path: Path, model: nn.Module, image_size: int, autocontrast: bool,
                    plate_crop: bool, device: str, batch_size: int = 32) -> tuple[np.ndarray, dict]:
    """Features of every image of `df`, extracting only those the cache file lacks and adding them to it."""
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as stored:
            cached = dict(zip(stored["paths"].tolist(), stored["features"]))
    missing = df[~df["image_path"].isin(cached)]
    if len(missing):
        new = extract_features(model, missing, image_size, autocontrast, plate_crop, device, batch_size)
        cached.update(zip(missing["image_path"], new))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, paths=np.array(list(cached)), features=np.stack(list(cached.values())))
    info = {"path": str(cache_path), "extracted": len(missing), "reused": len(df) - len(missing)}
    return np.stack([cached[p] for p in df["image_path"]]), info
