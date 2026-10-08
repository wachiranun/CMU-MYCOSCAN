"""Inference entry point for the web app (component 3).

    from mycoscan.predict import Predictor
    predictor = Predictor("runs/cmu_microscopic/model.pt")
    predictor.predict(pil_image_or_path)   # {"Talaromyces_marneffei": 0.91, ...}, highest first
    predictor.predict_bag([fov1, fov2, ...])
    # {"probabilities": {...}, highest first, "top2": [first, second], "no_call": False}

An image is taken to be of the checkpoint's modality unless `modality` says otherwise
(needed only for a `modality = "all"` model trained with `plate_crop`).

`predict_bag` takes the images of one isolate and pools them the way the checkpoint was trained to:
an attention-MIL model with its learned attention, any other model by its `pooling` (mean or max) of the
per-image probabilities. `no_call` is true when the top pooled probability is below the reject threshold
tau stored in the checkpoint; without a stored tau every call is made. The Top-2 is given either way.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from .bags import checkpoint_tiles, pool_probabilities
from .mil import MILModel
from .models import load_checkpoint
from .transforms import build_transform, load_image, tile_crop

Images = list[Image.Image | str | Path]


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = device
        self.model, meta = load_checkpoint(checkpoint, device)
        self.classes: list[str] = meta["classes"]
        self.modality: str = meta.get("modality", "")
        self.transform = build_transform(meta["image_size"], meta["autocontrast"], train=False)
        self.plate_crop: bool = meta.get("plate_crop", False)
        self.pooling: str = meta.get("pooling", "mean")
        self.tau: float | None = meta.get("tau")
        self.tiles = checkpoint_tiles(meta)

    def _ranked(self, probs: list[float]) -> dict[str, float]:
        return dict(sorted(zip(self.classes, probs, strict=True), key=lambda kv: kv[1], reverse=True))

    def _batch(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in images]).to(self.device)

    def _load(self, image: Image.Image | str | Path, modality: str | None) -> Image.Image:
        return load_image(image, self.plate_crop, modality or self.modality)

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path, modality: str | None = None) -> dict[str, float]:
        return self._ranked(self.model(self._batch([self._load(image, modality)])).softmax(dim=1)[0].cpu().tolist())

    @torch.no_grad()
    def predict_bag(self, images: Images, modality: str | None = None, devices: list[str] | None = None) -> dict:
        """The pooled probabilities of one isolate's images, highest first, its Top-2 and whether it is a no-call.
        devices: each image's device, for a device-then-isolate model; by default all images share one device.
        A tile checkpoint cuts every image into its tiles, as in training, and pools all of them."""
        if not images:
            raise ValueError("predict_bag needs at least one image")
        loaded = [self._load(img, modality) for img in images]
        per_image = self.tiles.count if self.tiles else 1
        if self.tiles:
            loaded = [tile_crop(img, self.tiles, k) for img in loaded for k in range(per_image)]
        x = self._batch(loaded)
        mask = torch.ones(1, len(loaded), dtype=torch.bool, device=self.device)
        if isinstance(self.model, MILModel):
            codes = pd.factorize(pd.Series(devices or ["device"] * len(images)).repeat(per_image))[0]
            index = torch.from_numpy(codes).unsqueeze(0).to(self.device)
            probs = self.model(x.unsqueeze(0), mask, index if self.model.hierarchical else None).softmax(dim=1)[0]
        else:
            probs = pool_probabilities(self.model(x).softmax(dim=1).unsqueeze(0), mask, self.pooling)[0][0]
        ranked = self._ranked(probs.cpu().tolist())
        top = list(ranked)
        return {"probabilities": ranked, "top2": top[:2],
                "no_call": self.tau is not None and ranked[top[0]] < self.tau}
