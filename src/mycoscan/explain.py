"""Explainability for review by clinical mycologists (spec 2.4).

Grad-CAM: class-discriminative heatmap from the last convolutional block.
SmoothGrad: pixel-level saliency, the mean absolute input gradient over noisy copies.
Each image gets a panel (original | Grad-CAM overlay | saliency) and a row in
review_sheet.csv with blank columns for the expert's concordance judgement.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T

from .models import adapter, load_checkpoint
from .transforms import build_transform, plate_circle_crop, to_rgb


def gradcam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target: int) -> np.ndarray:
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(hook)
    try:
        x = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x)[0, target].backward()
    finally:
        handle.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _unit(cam.detach().cpu().numpy())


def smoothgrad(model: nn.Module, x: torch.Tensor, target: int, n: int = 25, sigma: float = 0.15, seed: int = 0) -> np.ndarray:
    gen = torch.Generator(device=x.device).manual_seed(seed)
    scale = sigma * (x.max() - x.min())
    total = torch.zeros_like(x)
    for _ in range(n):
        noisy = (x + scale * torch.randn(x.shape, generator=gen, device=x.device)).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(noisy)[0, target].backward()
        total += noisy.grad.abs()
    return _unit(total[0].amax(dim=0).cpu().numpy())


def _unit(a: np.ndarray) -> np.ndarray:
    span = a.max() - a.min()
    return (a - a.min()) / span if span > 0 else np.zeros_like(a)


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = plt.get_cmap("jet")(heat)[..., :3]
    return ((1 - alpha) * rgb / 255.0 + alpha * colored).clip(0, 1)


def explain_images(checkpoint: str | Path, images: list[str], out_dir: str | Path, device: str = "cpu",
                   true_labels: list[str] | None = None) -> pd.DataFrame:
    model, meta = load_checkpoint(checkpoint, device)
    classes = meta["classes"]
    layer = model.get_submodule(adapter(model).cam_layer)
    tf = build_transform(meta["image_size"], meta["autocontrast"], train=False)
    size = meta["image_size"]
    view = T.Compose([T.Resize(size), T.CenterCrop(size)])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(images):
        with Image.open(path) as raw:
            img = to_rgb(raw)
        if meta.get("plate_crop") and meta.get("modality") == "colony":
            img = plate_circle_crop(img)
        x = tf(img).unsqueeze(0).to(device)
        rgb = np.asarray(view(img), dtype=float)
        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0].cpu().numpy()
        target = int(probs.argmax())
        cam = gradcam(model, layer, x, target)
        sal = smoothgrad(model, x, target)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        axes[0].imshow(rgb.astype(np.uint8))
        axes[0].set_title("original" + (f" (true: {true_labels[i]})" if true_labels else ""), fontsize=9)
        axes[1].imshow(overlay(rgb, cam))
        axes[1].set_title(f"Grad-CAM: {classes[target]} p={probs[target]:.2f}", fontsize=9)
        axes[2].imshow(sal, cmap="inferno")
        axes[2].set_title("SmoothGrad saliency", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        panel = out_dir / f"{i:04d}_{Path(path).stem}.png"
        fig.savefig(panel, dpi=110)
        plt.close(fig)
        rows.append({"image_path": path, "true_species": true_labels[i] if true_labels else "",
                     "predicted": classes[target], "probability": round(float(probs[target]), 4),
                     "panel": panel.name, "reviewer": "", "concordant_with_morphology": "",
                     "highlighted_structure": "", "artifact_suspected": "", "notes": ""})
    sheet = pd.DataFrame(rows)
    sheet.to_csv(out_dir / "review_sheet.csv", index=False)
    return sheet
