"""Training with isolate-level validation, and evaluation of a saved checkpoint."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import Config
from .data import make_loader
from .losses import build_loss
from .manifest import load_manifest, select
from .metrics import evaluate_predictions, prob_columns
from .models import apply_finetune, build_model, load_checkpoint, save_checkpoint, train_mode
from .splits import assert_no_group_leakage, describe_fold, make_folds

log = logging.getLogger("mycoscan")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def fit_model(train_df: pd.DataFrame, cfg: Config, classes: list[str], device: str, seed: int) -> tuple[nn.Module, list[dict]]:
    seed_everything(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    model = build_model(cfg.arch, len(classes), cfg.weights, cfg.image_size)
    apply_finetune(model, cfg.finetune)
    model.to(device)
    loader = make_loader(train_df, class_to_idx, cfg.image_size, cfg.autocontrast, train=True,
                         batch_size=cfg.batch_size, num_workers=cfg.num_workers, imbalance=cfg.imbalance, seed=seed,
                         augmentation=cfg.augmentation, plate_crop=cfg.plate_crop)
    labels = torch.tensor(train_df["species"].map(class_to_idx).to_numpy())
    loss_fn = build_loss(cfg.loss, labels, len(classes), cfg.label_smoothing, cfg.focal_gamma).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    device_type = torch.device(device).type
    # float16 needs loss scaling and exists only on GPU; CPU autocast uses bfloat16, which needs none.
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp and amp_dtype == torch.float16)
    history = []
    for epoch in range(cfg.epochs):
        train_mode(model)
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type, dtype=amp_dtype, enabled=cfg.amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), "train_acc": correct / max(seen, 1)})
        log.info("  epoch %d/%d loss=%.4f acc=%.3f", epoch + 1, cfg.epochs, history[-1]["train_loss"], history[-1]["train_acc"])
    return model.eval(), history


@torch.no_grad()
def predict_probs(model: nn.Module, df: pd.DataFrame, classes: list[str], image_size: int, autocontrast: bool,
                  device: str, batch_size: int = 32, plate_crop: bool = False) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    loader = make_loader(df, class_to_idx, image_size, autocontrast, train=False, batch_size=batch_size, num_workers=0,
                         plate_crop=plate_crop)
    model.eval()
    return np.concatenate([model(x.to(device)).softmax(dim=1).cpu().numpy() for x, _ in loader])


def prediction_table(df: pd.DataFrame, probs: np.ndarray, classes: list[str], fold: str) -> pd.DataFrame:
    keep = ["image_path", "species", "isolate_id", "group_id", "group", "modality", "view", "device", "day",
            "source", "genus", "temperature", "phase", "fov_id", "z_index"]
    table = df[keep].reset_index(drop=True).copy()
    table["fold"] = fold
    table["predicted"] = [classes[i] for i in probs.argmax(axis=1)]
    table[prob_columns(classes)] = probs
    return table


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(classes), 0.8 + 0.6 * len(classes)))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(run_dir: Path, preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int, extra: dict) -> dict:
    preds.to_csv(run_dir / "predictions.csv", index=False)
    metrics = {**extra, **evaluate_predictions(preds, classes, n_boot, seed)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for level in ("image_level", "isolate_level"):
        plot_confusion(metrics[level]["confusion_matrix"], classes, run_dir / f"confusion_{level}.png", level.replace("_", " "))
    return metrics


def _resolve_classes(df: pd.DataFrame, cfg_classes: tuple[str, ...]) -> list[str]:
    classes = list(cfg_classes) or sorted(df["species"].unique())
    unknown = sorted(set(df["species"]) - set(classes))
    if unknown:
        raise ValueError(f"manifest species not in config classes: {unknown}")
    return classes


def run_training(cfg: Config) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    df = select(load_manifest(cfg.manifest), cfg.modality, cfg.source)
    if df.empty:
        raise ValueError(f"no images for modality={cfg.modality} source={cfg.source} in {cfg.manifest}")
    classes = _resolve_classes(df, cfg.classes)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    log.info("run %s: %d images, %d groups, %d classes, device=%s", cfg.run_name, len(df),
             df["group"].nunique(), len(classes), device)
    meta = {"modality": cfg.modality, "source": cfg.source, "finetune": cfg.finetune, "weights_init": cfg.weights,
            "plate_crop": cfg.plate_crop}

    folds = make_folds(df, cfg.split, cfg.n_folds, cfg.val_fraction, cfg.seed)
    fold_info, tables, histories = [], [], {}
    started = time.time()
    for k, fold in enumerate(folds):
        assert_no_group_leakage(df, fold)
        info = describe_fold(df, fold, classes)
        fold_info.append(info)
        log.info("%s: %d/%d train/val images, %d/%d groups, val classes without groups: %s", fold.name,
                 info["train_images"], info["val_images"], info["train_groups"], info["val_groups"],
                 info["val_classes_without_groups"])
        model, histories[fold.name] = fit_model(df.iloc[fold.train_idx], cfg, classes, device, cfg.seed + k)
        val_df = df.iloc[fold.val_idx]
        probs = predict_probs(model, val_df, classes, cfg.image_size, cfg.autocontrast, device, cfg.batch_size,
                              cfg.plate_crop)
        tables.append(prediction_table(val_df, probs, classes, fold.name))
        ckpt_path = run_dir / "model.pt" if cfg.split == "holdout" else run_dir / "folds" / f"{fold.name}.pt"
        save_checkpoint(ckpt_path, model, cfg.arch, classes, cfg.image_size, cfg.autocontrast, {**meta, "fold": fold.name})

    metrics = write_report(run_dir, pd.concat(tables, ignore_index=True), classes, cfg.bootstrap, cfg.seed,
                           {"split": cfg.split, "folds": fold_info, "train_history": histories,
                            "note": "validation predictions are out-of-fold, real images only, groups disjoint from training"})
    if cfg.split != "holdout" and cfg.fit_final:
        log.info("final model on all %d groups", df["group"].nunique())
        model, _ = fit_model(df, cfg, classes, device, cfg.seed)
        save_checkpoint(run_dir / "model.pt", model, cfg.arch, classes, cfg.image_size, cfg.autocontrast, {**meta, "fold": "all"})
    log.info("done in %.0fs. image acc=%.3f macro-F1=%.3f | isolate acc=%.3f macro-F1=%.3f -> %s", time.time() - started,
             metrics["image_level"]["accuracy"], metrics["image_level"]["macro"]["f1"],
             metrics["isolate_level"]["accuracy"], metrics["isolate_level"]["macro"]["f1"], run_dir)
    return run_dir


def evaluate_checkpoint(checkpoint: str | Path, manifest: str | Path, out_dir: str | Path, modality: str | None = None,
                        source: str = "all", n_boot: int = 1000, device: str = "auto", seed: int = 0) -> dict:
    """Score a saved model on a manifest it was not trained on (e.g. a later external test set)."""
    device = resolve_device(device)
    model, meta = load_checkpoint(checkpoint, device)
    classes = meta["classes"]
    df = select(load_manifest(manifest), modality or meta.get("modality", "all"), source)
    _resolve_classes(df, tuple(classes))
    probs = predict_probs(model, df, classes, meta["image_size"], meta["autocontrast"], device,
                          plate_crop=meta.get("plate_crop", False))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_report(out_dir, prediction_table(df, probs, classes, "eval"), classes, n_boot, seed,
                        {"checkpoint": str(checkpoint), "manifest": str(manifest)})
