"""Explainability for review by clinical mycologists (spec 2.4).

The map is picked from the checkpoint's architecture:
Grad-CAM for CNNs: class-discriminative heatmap from the last convolutional block.
Attention rollout for ViTs: attention averaged over heads, with the residual added, multiplied
through the layers, read from the class token to the patches (Abnar and Zuidema, 2020).
SmoothGrad for both: pixel-level saliency, the mean absolute input gradient over noisy copies.

Each image gets a panel (original | map overlay | saliency) and a row in review_sheet.csv
for two raters, each judging where the map points on a 3-point scale: structure (the
diagnostic morphology), partial, or background. The sheet is blinded: it and the panels
show no true label and no image path (OpenFungi paths name the class) unless `reveal`.
review_key.csv maps each panel to its image and true label; keep it from the raters.
`score_review` reads filled sheets and reports percent structure-focused per rater and
Cohen's kappa between the raters.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T

from .metrics import cohen_kappa
from .mil import ATTENTION_POOLINGS
from .models import adapter, load_checkpoint
from .transforms import build_transform, load_image

FOCUS_SCALE = ("structure", "partial", "background")
RATERS = ("rater1", "rater2")
REVIEW_COLUMNS = ("panel", "method", "predicted", "probability", "true_species", "image_path",
                  *(f"{r}_{field}" for r in RATERS for field in ("focus", "notes")))


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


def is_vit(model: nn.Module) -> bool:
    """A timm vision transformer: patch tokens behind class (and register) tokens."""
    return all(hasattr(model, a) for a in ("patch_embed", "blocks", "num_prefix_tokens"))


@torch.no_grad()
def attention_rollout(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Attention rollout of a timm ViT for one image, upsampled to the input size, in [0, 1].

    Fused attention never materialises the attention matrix, so it is switched off for this pass;
    on the unfused path timm passes the softmaxed attention through `attn_drop`, where a hook reads it."""
    attentions: list[torch.Tensor] = []
    modules = [m for m in model.modules()
               if hasattr(m, "fused_attn") and isinstance(getattr(m, "attn_drop", None), nn.Module)]
    fused = [getattr(m, "fused_attn") for m in modules]
    hooks = [getattr(m, "attn_drop").register_forward_hook(lambda _m, _i, out: attentions.append(out)) for m in modules]
    try:
        for m in modules:
            setattr(m, "fused_attn", False)
        model.eval()(x)
    finally:
        for m, f in zip(modules, fused):
            setattr(m, "fused_attn", f)
        for h in hooks:
            h.remove()
    if not attentions:
        raise ValueError(f"{type(model).__name__} exposes no attention to roll out")
    tokens = attentions[0].shape[-1]
    eye = torch.eye(tokens, device=x.device)
    rollout = eye
    for attention in attentions:
        a = attention[0].mean(dim=0) + eye  # heads averaged, residual connection added
        rollout = (a / a.sum(dim=-1, keepdim=True)) @ rollout
    prefix = getattr(model, "num_prefix_tokens")
    patches = rollout[0, prefix:]
    side = int(round(patches.numel() ** 0.5))
    grid = tuple(getattr(getattr(model, "patch_embed"), "grid_size", (side, side)))
    heat = F.interpolate(patches.reshape(1, 1, *grid), size=x.shape[-2:], mode="bilinear", align_corners=False)
    return _unit(heat[0, 0].cpu().numpy())


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


METHOD_TITLES = {"gradcam": "Grad-CAM", "attention_rollout": "Attention rollout"}


def explain_images(checkpoint: str | Path, images: list[str], out_dir: str | Path, device: str = "cpu",
                   true_labels: list[str] | None = None, modalities: list[str] | None = None,
                   reveal: bool = False) -> pd.DataFrame:
    """Panels, review_sheet.csv and review_key.csv for `images`. True labels and image paths reach the sheet
    and the panel titles only with `reveal`. modalities: one per image, for the plate crop; defaults to the
    checkpoint's modality."""
    model, meta = load_checkpoint(checkpoint, device)
    if meta.get("pooling") in ATTENTION_POOLINGS:
        raise ValueError(f"{checkpoint} is an attention-MIL model; its per-instance attention weights are in the "
                         "run's attention.csv, and Grad-CAM explains single-image networks only")
    classes = meta["classes"]
    method = "attention_rollout" if is_vit(model) else "gradcam"
    layer = None if method == "attention_rollout" else model.get_submodule(adapter(model).cam_layer)
    tf = build_transform(meta["image_size"], meta["autocontrast"], train=False)
    size = meta["image_size"]
    view = T.Compose([T.Resize(size), T.CenterCrop(size)])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, key = [], []
    for i, path in enumerate(images):
        img = load_image(path, meta.get("plate_crop", False), modalities[i] if modalities else meta.get("modality", ""))
        x = tf(img).unsqueeze(0).to(device)
        rgb = np.asarray(view(img), dtype=float)
        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0].cpu().numpy()
        target = int(probs.argmax())
        heat = attention_rollout(model, x) if layer is None else gradcam(model, layer, x, target)
        sal = smoothgrad(model, x, target)
        true = true_labels[i] if true_labels else ""
        panel = f"{i:04d}.png"
        titles = [f"image {i:04d}" + (f" (true: {true})" if reveal and true else ""),
                  f"{METHOD_TITLES[method]}: {classes[target]} p={probs[target]:.2f}", "SmoothGrad saliency"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        axes[0].imshow(rgb.astype(np.uint8))
        axes[1].imshow(overlay(rgb, heat))
        axes[2].imshow(sal, cmap="inferno")
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / panel, dpi=110, metadata={"Title": " | ".join(titles)})
        plt.close(fig)
        rows.append({"panel": panel, "method": method, "predicted": classes[target],
                     "probability": round(float(probs[target]), 4), "true_species": true if reveal else "",
                     "image_path": path if reveal else ""})
        key.append({"panel": panel, "image_path": path, "true_species": true})
    sheet = pd.DataFrame(rows).reindex(columns=list(REVIEW_COLUMNS), fill_value="")
    sheet.to_csv(out_dir / "review_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(out_dir / "review_key.csv", index=False)
    return sheet


def sample_for_review(df: pd.DataFrame, per_class: int, seed: int = 0) -> pd.DataFrame:
    """Up to `per_class` images of each class, drawn at random with `seed`; a class with fewer gives all it has."""
    return pd.concat([rows.sample(n=min(per_class, len(rows)), random_state=seed)
                      for _, rows in df.groupby("species", sort=True)], ignore_index=True)


def score_review(sheets: Sequence[str | Path]) -> dict:
    """Percent of panels each rater judged structure, partial or background, and the raters' agreement
    (percent and Cohen's kappa) on the panels both rated. Several sheets (one per rater) are merged by panel,
    each rater column taken from whichever sheet filled it."""
    if not sheets:
        raise ValueError("no review sheets given")
    merged: pd.DataFrame | None = None
    for path in sheets:
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("panel")
        merged = sheet if merged is None else merged.mask(merged == "", sheet.reindex(merged.index).fillna(""))
    assert merged is not None
    result: dict = {"scale": list(FOCUS_SCALE)}
    focus = {}
    for rater in RATERS:
        values = merged[f"{rater}_focus"].str.strip().str.lower()
        bad = values[(values != "") & ~values.isin(FOCUS_SCALE)]
        if len(bad):
            raise ValueError(f"{rater}_focus has values off the scale {list(FOCUS_SCALE)}: "
                             f"{dict(list(bad.items())[:5])}")
        focus[rater] = values
        rated = values[values != ""]
        result[rater] = {"n": len(rated), **{v: float((rated == v).mean()) if len(rated) else float("nan")
                                             for v in FOCUS_SCALE}}
    both = (focus["rater1"] != "") & (focus["rater2"] != "")
    index = {v: k for k, v in enumerate(FOCUS_SCALE)}
    cm = np.zeros((len(FOCUS_SCALE), len(FOCUS_SCALE)), dtype=int)
    np.add.at(cm, (focus["rater1"][both].map(index).to_numpy(), focus["rater2"][both].map(index).to_numpy()), 1)
    result["agreement"] = {"n": int(both.sum()), "percent": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
                           "kappa": cohen_kappa(cm), "table": cm.tolist()}
    return result
