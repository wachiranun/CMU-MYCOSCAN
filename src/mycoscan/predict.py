"""Inference entry point for the web app (component 3).

    from mycoscan.predict import Predictor
    predictor = Predictor("runs/cmu_microscopic/model.pt")
    predictor.predict(pil_image_or_path)   # {"Talaromyces_marneffei": 0.91, ...}, highest first

An image is taken to be of the checkpoint's modality unless `modality` says otherwise
(needed only for a `modality = "all"` model trained with `plate_crop`).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .models import load_checkpoint
from .transforms import build_transform, load_image


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = device
        self.model, meta = load_checkpoint(checkpoint, device)
        self.classes: list[str] = meta["classes"]
        self.modality: str = meta.get("modality", "")
        self.transform = build_transform(meta["image_size"], meta["autocontrast"], train=False)
        self.plate_crop: bool = meta.get("plate_crop", False)

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path, modality: str | None = None) -> dict[str, float]:
        img = load_image(image, self.plate_crop, modality or self.modality)
        x = self.transform(img).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0].cpu().tolist()
        return dict(sorted(zip(self.classes, probs, strict=True), key=lambda kv: kv[1], reverse=True))
