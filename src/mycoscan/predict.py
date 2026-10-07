"""Inference entry point for the web app (component 3).

    from mycoscan.predict import Predictor
    predictor = Predictor("runs/cmu_microscopic/model.pt")
    predictor.predict(pil_image_or_path)   # {"Talaromyces_marneffei": 0.91, ...}, highest first
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .models import load_checkpoint
from .transforms import build_transform, plate_circle_crop, to_rgb


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = device
        self.model, meta = load_checkpoint(checkpoint, device)
        self.classes: list[str] = meta["classes"]
        self.modality: str = meta.get("modality", "")
        self.transform = build_transform(meta["image_size"], meta["autocontrast"], train=False)
        self.plate_crop: bool = meta.get("plate_crop", False) and self.modality == "colony"

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path) -> dict[str, float]:
        if not isinstance(image, Image.Image):
            with Image.open(image) as img:
                image = to_rgb(img)
        image = to_rgb(image)
        x = self.transform(plate_circle_crop(image) if self.plate_crop else image).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0].cpu().tolist()
        return dict(sorted(zip(self.classes, probs, strict=True), key=lambda kv: kv[1], reverse=True))
