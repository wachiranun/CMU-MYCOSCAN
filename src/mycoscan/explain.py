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

import logging
from collections.abc import Sequence
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

from .bags import checkpoint_tiles, instances
from .metrics import cohen_kappa
from .mil import MILModel, predict_mil
from .models import adapter, load_checkpoint
from .transforms import build_transform, load_image, tile_crop

log = logging.getLogger("mycoscan")

FOCUS_SCALE = ("structure", "partial", "background")
RATERS = ("rater1", "rater2")
REVIEW_COLUMNS = ("panel", "level", "method", "predicted", "probability", "true_species", "image_path",
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
BAG_INSTANCES = 8  # instances drawn in a bag panel, highest weight first
BAG_HEATMAPS = 3  # of those, how many get a heat-map
KEY_COLUMNS = ("panel", "level", "bag", "image_path", "attention", "true_species")


def _heatmapper(net: nn.Module, encoder: nn.Module):
    """The map method for `net` (its encoder for an attention-MIL model) and a function (x, target) -> map."""
    if is_vit(encoder):
        return "attention_rollout", lambda x, _target: attention_rollout(encoder, x)
    layer = encoder.get_submodule(adapter(encoder).cam_layer)
    return "gradcam", lambda x, target: gradcam(net, layer, x, target)


def _save_panel(fig, path: Path, titles: list[str]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, metadata={"Title": " | ".join(titles)})
    plt.close(fig)


def explain_images(checkpoint: str | Path, images: list[str], out_dir: str | Path, device: str = "cpu",
                   true_labels: list[str] | None = None, modalities: list[str] | None = None,
                   reveal: bool = False, bags: list[str] | None = None,
                   devices: list[str] | None = None) -> pd.DataFrame:
    """Panels, review_sheet.csv and review_key.csv for `images`. True labels and image paths reach the sheet
    and the panel titles only with `reveal`. modalities: one per image, for the plate crop; defaults to the
    checkpoint's modality.

    An attention-MIL checkpoint gets one panel per bag (`level = "bag"` rows): its instances ordered by
    attention weight, each with its weight, and a heat-map of the top instances for the bag's call, each
    instance scored alone. bags: each image's bag (by default all images form one bag; a tile checkpoint
    bags each image's tiles); devices: each image's device, for a device-then-isolate model, whose weights
    shown are each instance's share of the isolate (its weight in its device times the device's weight).
    Any other checkpoint gets one panel per image (`level = "image"` rows)."""
    model, meta = load_checkpoint(checkpoint, device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true_labels = true_labels or [""] * len(images)
    modalities = modalities or [meta.get("modality", "")] * len(images)
    if isinstance(model, MILModel):
        rows, key = _bag_panels(model, meta, images, out_dir, device, true_labels, modalities, reveal, bags, devices)
    else:
        if bags is not None or meta.get("bag", "none") != "none":
            log.info("%s pools its bags by a fixed %s of instance predictions, not learned attention, so there are no "
                     "instance weights to show: no bag panel, each image is explained alone",
                     checkpoint, meta.get("pooling", "mean"))
        rows, key = _image_panels(model, meta, images, out_dir, device, true_labels, modalities, reveal)
    sheet = pd.DataFrame(rows).reindex(columns=list(REVIEW_COLUMNS), fill_value="")
    sheet.to_csv(out_dir / "review_sheet.csv", index=False)
    pd.DataFrame(key).reindex(columns=list(KEY_COLUMNS), fill_value="").to_csv(out_dir / "review_key.csv", index=False)
    return sheet


def _image_panels(model: nn.Module, meta: dict, images: list[str], out_dir: Path, device: str, true_labels: list[str],
                  modalities: list[str], reveal: bool) -> tuple[list[dict], list[dict]]:
    classes = meta["classes"]
    method, heatmap = _heatmapper(model, model)
    tf = build_transform(meta["image_size"], meta["autocontrast"], train=False)
    size = meta["image_size"]
    view = T.Compose([T.Resize(size), T.CenterCrop(size)])
    rows, key = [], []
    for i, path in enumerate(images):
        img = load_image(path, meta.get("plate_crop", False), modalities[i])
        x = tf(img).unsqueeze(0).to(device)
        rgb = np.asarray(view(img), dtype=float)
        with torch.no_grad():
            probs = model(x).softmax(dim=1)[0].cpu().numpy()
        target = int(probs.argmax())
        heat = heatmap(x, target)
        sal = smoothgrad(model, x, target)
        true = true_labels[i]
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
        _save_panel(fig, out_dir / panel, titles)
        rows.append({"panel": panel, "level": "image", "method": method, "predicted": classes[target],
                     "probability": round(float(probs[target]), 4), "true_species": true if reveal else "",
                     "image_path": path if reveal else ""})
        key.append({"panel": panel, "level": "image", "image_path": path, "true_species": true})
    return rows, key


def _bag_panels(model: MILModel, meta: dict, images: list[str], out_dir: Path, device: str, true_labels: list[str],
                modalities: list[str], reveal: bool, bags: list[str] | None,
                devices: list[str] | None) -> tuple[list[dict], list[dict]]:
    classes = meta["classes"]
    method, heatmap = _heatmapper(model, model.encoder)
    tiles = checkpoint_tiles(meta)
    plate_crop = meta.get("plate_crop", False)
    tf = build_transform(meta["image_size"], meta["autocontrast"], train=False)
    size = meta["image_size"]
    view = T.Compose([T.Resize(size), T.CenterCrop(size)])
    frame = pd.DataFrame({"image_path": images, "modality": modalities, "species": true_labels,
                          "bag": bags or ["bag"] * len(images), "device": devices or ["device"] * len(images)})
    if tiles:
        frame = instances(frame.drop(columns="bag"), "tiles", tiles)
    preds = predict_mil(model, frame, tf, device, 32, plate_crop, tiles)
    weight = preds.instance_weights.mean(axis=1)  # heads averaged
    if preds.device_names is not None and preds.device_weights is not None:
        of_device = dict(zip(preds.device_names, preds.device_weights))
        weight = weight * (frame["bag"].astype(str) + "|" + frame["device"].astype(str)).map(of_device).to_numpy()
    names = frame["image_path"] + (("#tile" + frame["tile"].astype(str)) if tiles else "")
    rows, key = [], []
    for b, bag in enumerate(preds.names):
        positions = np.flatnonzero(frame["bag"].to_numpy() == bag)
        order = positions[np.argsort(-weight[positions], kind="stable")]
        probs = preds.bag_probs[b]
        target = int(probs.argmax())
        true = frame["species"].iat[order[0]]
        panel = f"bag_{b:04d}.png"
        shown = order[:BAG_INSTANCES]
        more = f" (+{len(order) - len(shown)} more)" if len(order) > len(shown) else ""
        titles = [f"bag {b:04d}" + (f" (true: {true})" if reveal and true else "")
                  + f": {classes[target]} p={probs[target]:.2f}, {len(order)} instances{more}",
                  *(f"#{rank + 1} w={weight[p]:.3f}" for rank, p in enumerate(shown))]

        fig, axes = plt.subplots(2, len(shown), figsize=(2.6 * len(shown) + 0.6, 5.8), squeeze=False)
        loaded: dict[str, Image.Image] = {}
        for rank, p in enumerate(shown):
            row = frame.iloc[p]
            if row["image_path"] not in loaded:
                loaded[row["image_path"]] = load_image(row["image_path"], plate_crop, row["modality"])
            img = loaded[row["image_path"]]
            img = tile_crop(img, tiles, int(row["tile"])) if tiles else img
            rgb = np.asarray(view(img), dtype=float)
            axes[0, rank].imshow(rgb.astype(np.uint8))
            axes[0, rank].set_title(titles[rank + 1], fontsize=9)
            if rank < BAG_HEATMAPS:
                axes[1, rank].imshow(overlay(rgb, heatmap(tf(img).unsqueeze(0).to(device), target)))
                axes[1, rank].set_title(METHOD_TITLES[method], fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(titles[0], fontsize=10)
        _save_panel(fig, out_dir / panel, titles)
        ordered = ";".join(names.iloc[order])
        rows.append({"panel": panel, "level": "bag", "method": method, "predicted": classes[target],
                     "probability": round(float(probs[target]), 4), "true_species": true if reveal else "",
                     "image_path": ordered if reveal else ""})
        key.append({"panel": panel, "level": "bag", "bag": bag, "image_path": ordered,
                    "attention": ";".join(f"{weight[p]:.6f}" for p in order), "true_species": true})
    return rows, key


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
