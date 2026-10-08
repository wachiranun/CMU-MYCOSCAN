"""Training with isolate-level validation, and evaluation of a saved checkpoint."""
from __future__ import annotations

import hashlib
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
from torch.optim.swa_utils import AveragedModel

from .config import Config
from .data import held_out_mask, make_loader, refuse_held_out
from .features import cache_key, cached_features
from .losses import build_loss
from .manifest import load_manifest, select
from .metrics import evaluate_predictions, prob_columns
from .models import (add_lora, apply_finetune, build_model, load_checkpoint, lr_scales, merge_lora, require_peft,
                     save_checkpoint, train_mode)
from .probe import Probes, fit_probes, with_linear_head
from .provenance import collect, log_to_mlflow, require_mlflow, sha256_file
from .splits import (Fold, apply_splits, assert_no_group_leakage, describe_fold, frozen_folds, load_splits_file,
                     make_folds, subsample_groups)

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


def prepare_model(cfg: Config, n_classes: int) -> nn.Module:
    """The model a run trains: its backbone and initialisation, with the fine-tuning policy applied."""
    model = build_model(cfg.arch, n_classes, cfg.weights, cfg.image_size)
    if cfg.finetune == "lora":
        add_lora(model, cfg.lora_rank)
    else:
        apply_finetune(model, cfg.finetune, cfg.partial_blocks)
    return model


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW over the trainable parameters, one group per learning rate that layer_decay assigns."""
    scales = lr_scales(model, cfg.layer_decay)
    by_scale: dict[float, list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            by_scale.setdefault(scales[name], []).append(param)
    groups = [{"params": params, "lr": cfg.lr * scale} for scale, params in sorted(by_scale.items(), reverse=True)]
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _ema_with_warmup(decay: float):
    """EMA whose decay ramps up as min(decay, (1 + n) / (10 + n)) after n updates, as timm's ModelEmaV3 does.
    A fixed 0.999 from the first step would keep most of the untrained weights in a run of a few hundred steps."""
    def average(averaged: torch.Tensor, current: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        d = torch.clamp((1 + n) / (10 + n), max=decay)
        return d * averaged + (1 - d) * current
    return average


def fit_model(train_df: pd.DataFrame, cfg: Config, classes: list[str], device: str, seed: int) -> tuple[nn.Module, list[dict]]:
    seed_everything(seed)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    model = prepare_model(cfg, len(classes))
    model.to(device)
    loader = make_loader(train_df, class_to_idx, cfg.image_size, cfg.autocontrast, train=True,
                         batch_size=cfg.batch_size, num_workers=cfg.num_workers, imbalance=cfg.imbalance, seed=seed,
                         augmentation=cfg.augmentation, plate_crop=cfg.plate_crop)
    labels = torch.tensor(train_df["species"].map(class_to_idx).to_numpy())
    loss_fn = build_loss(cfg.loss, labels, len(classes), cfg.label_smoothing, cfg.focal_gamma).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    device_type = torch.device(device).type
    # float16 needs loss scaling and exists only on GPU; CPU autocast uses bfloat16, which needs none.
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.amp and amp_dtype == torch.float16)
    # Buffers are averaged too, so BatchNorm statistics match the averaged weights.
    ema = AveragedModel(model, avg_fn=_ema_with_warmup(cfg.ema_decay), use_buffers=True) if cfg.ema else None
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
            if ema is not None:
                ema.update_parameters(model)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        sched.step()
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), "train_acc": correct / max(seen, 1)})
        log.info("  epoch %d/%d loss=%.4f acc=%.3f", epoch + 1, cfg.epochs, history[-1]["train_loss"], history[-1]["train_acc"])
    if ema is not None:
        model = ema.module
    if cfg.finetune == "lora":
        merge_lora(model)
    return model.eval(), history


@torch.no_grad()
def predict_probs(model: nn.Module, df: pd.DataFrame, classes: list[str], image_size: int, autocontrast: bool,
                  device: str, batch_size: int = 32, plate_crop: bool = False) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    loader = make_loader(df, class_to_idx, image_size, autocontrast, train=False, batch_size=batch_size, num_workers=0,
                         plate_crop=plate_crop)
    model.eval()
    return np.concatenate([model(x.to(device)).softmax(dim=1).cpu().numpy() for x, _ in loader])


def prediction_table(df: pd.DataFrame, probs: np.ndarray, classes: list[str], fold: str,
                     classifier: str = "network") -> pd.DataFrame:
    keep = ["image_path", "species", "isolate_id", "group_id", "group", "modality", "view", "device", "day",
            "source", "genus", "temperature", "phase", "fov_id", "z_index"]
    table = df[keep].reset_index(drop=True).copy()
    table["fold"] = fold
    table["classifier"] = classifier
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


def evaluate_by_classifier(preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int, **metric_options) -> dict:
    """The metric block of the table's first classifier, and with several (a linear probe's logreg and knn)
    every classifier's block under `classifiers`. Each classifier is scored on its own rows only."""
    names = list(dict.fromkeys(preds["classifier"]))
    blocks = {name: evaluate_predictions(preds[preds["classifier"] == name].reset_index(drop=True), classes, n_boot,
                                         seed, **metric_options) for name in names}
    return {"classifier": names[0], **blocks[names[0]], **({"classifiers": blocks} if len(names) > 1 else {})}


def write_report(run_dir: Path, preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int, extra: dict,
                 **metric_options) -> dict:
    """metric_options: genus_map, order_map, subgroups and label_map, passed to evaluate_predictions."""
    preds.to_csv(run_dir / "predictions.csv", index=False)
    metrics = {**extra, **evaluate_by_classifier(preds, classes, n_boot, seed, **metric_options)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flag = " (LEAKY, comparison only)" if extra.get("leaky") else ""
    for level in ("isolate_level", "image_level"):
        plot_confusion(metrics[level]["confusion_matrix"], metrics["scored_classes"], run_dir / f"confusion_{level}.png",
                       level.replace("_", " ") + flag)
    return metrics


def _resolve_classes(df: pd.DataFrame, cfg_classes: tuple[str, ...]) -> list[str]:
    classes = list(cfg_classes) or sorted(df["species"].unique())
    unknown = sorted(set(df["species"]) - set(classes))
    if unknown:
        raise ValueError(f"manifest species not in config classes: {unknown}")
    return classes


def genus_map_of(df: pd.DataFrame, overrides: dict) -> dict:
    """species -> genus from the manifest's genus column, with the config's map taking precedence.
    Read from the whole manifest, because a validation fold need not contain every class."""
    named = df[df["genus"] != ""]
    return {**named.groupby("species")["genus"].first().to_dict(), **overrides}


def run_training(cfg: Config) -> Path:
    if cfg.tracking == "mlflow":
        require_mlflow()
    if cfg.finetune == "lora":
        require_peft()
    provenance = collect(cfg.manifest, asdict(cfg), cfg.splits_file or None)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    leaky = cfg.split == "image_random"
    df = select(load_manifest(cfg.manifest, allow_ungrouped=leaky), cfg.modality, cfg.source)
    if cfg.select_classes:
        absent = sorted(set(cfg.select_classes) - set(df["species"]))
        if absent:
            raise ValueError(f"config select_classes {absent} have no images for modality={cfg.modality} "
                             f"source={cfg.source}")
        df = df[df["species"].isin(cfg.select_classes)].reset_index(drop=True)
    if cfg.splits_file:
        df = apply_splits(df, load_splits_file(cfg.splits_file, cfg.manifest))
        held_out = held_out_mask(df)
        log.info("splits file %s: %d Pool B or sealed test images held out of this run", cfg.splits_file,
                 int(held_out.sum()))
        df = df[~held_out].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no images for modality={cfg.modality} source={cfg.source} in {cfg.manifest}")
    frozen = bool(cfg.splits_file) and cfg.split == "kfold"
    df, subsample = subsample_groups(df, cfg.train_fraction, cfg.seed,
                                     strata=("species", "fold") if frozen else ("species",))
    if subsample["removed_groups"]:
        log.info("train_fraction=%g: %d groups removed before fold creation", cfg.train_fraction,
                 len(subsample["removed_groups"]))
    classes = _resolve_classes(df, cfg.classes)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "leaky": leaky}, indent=2), encoding="utf-8")
    log.info("run %s: %d images, %d groups, %d classes, device=%s", cfg.run_name, len(df),
             df["group"].nunique(), len(classes), device)
    if leaky:
        log.warning("split=image_random is leaky, comparison only: images of one group land on both sides of a split")
    meta = {"modality": cfg.modality, "source": cfg.source, "finetune": cfg.finetune, "weights_init": cfg.weights,
            "plate_crop": cfg.plate_crop, "leaky": leaky}

    if frozen:
        folds = frozen_folds(df)
        if len(folds) != cfg.n_folds:
            raise ValueError(f"config n_folds={cfg.n_folds} but {cfg.splits_file} freezes {len(folds)} folds")
    else:
        folds = make_folds(df, cfg.split, cfg.n_folds, cfg.val_fraction, cfg.seed)
    fold_info = []
    for fold in folds:
        if not leaky:
            assert_no_group_leakage(df, fold)
        info = describe_fold(df, fold, classes)
        fold_info.append(info)
        log.info("%s: %d/%d train/val images, %d/%d groups, val classes without groups: %s", fold.name,
                 info["train_images"], info["val_images"], info["train_groups"], info["val_groups"],
                 info["val_classes_without_groups"])
    # Folds come from `seed` alone; each entry of `seeds` changes initialisation and sampling, never the split.
    seeds = list(cfg.seeds) or [cfg.seed]
    provenance |= {"seeds": seeds, "folds": fold_record(df, folds)}
    metric_options = dict(genus_map=genus_map_of(df, cfg.genus_map), order_map=cfg.order_map, subgroups=cfg.subgroups)
    base = {"run_name": cfg.run_name, "cell": cfg.run_name, "split": cfg.split, "leaky": leaky,
            "subsample": subsample, "provenance": provenance}
    note = ("LEAKY, comparison only: groups shared between training and validation" if leaky
            else "validation predictions are out-of-fold, real images only, groups disjoint from training")
    started = time.time()
    probe = cfg.finetune == "linear_probe"
    if probe:
        # Held-out rows were dropped above, so no Pool B or sealed test image is ever embedded.
        backbone = build_model(cfg.arch, len(classes), cfg.weights, cfg.image_size)
        key = cache_key(provenance["manifest_sha256"], cfg.arch, cfg.weights, cfg.image_size, cfg.autocontrast,
                        cfg.plate_crop)
        extraction_started = start_cost(device)
        features, cache_info = cached_features(df, Path(cfg.output_dir) / "feature_cache" / f"{key}.npz", backbone,
                                               cfg.image_size, cfg.autocontrast, cfg.plate_crop, device, cfg.batch_size)
        extraction_cost = cost_since(extraction_started, device)
        base["feature_cache"] = {**cache_info, "wall_seconds": extraction_cost["wall_seconds"]}
        log.info("features: %d extracted, %d from cache", cache_info["extracted"], cache_info["reused"])
    fold_scores, pooled = [], {}
    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}" if cfg.seeds else run_dir
        tables, histories = [], {}
        costs = [extraction_cost] if probe else []  # a probe has one seed, so extraction is counted once
        for k, fold in enumerate(folds):
            fold_started = start_cost(device)
            val_df = df.iloc[fold.val_idx]
            if probe:
                probes = _fit_probes(df.iloc[fold.train_idx], features[fold.train_idx], classes)
                table = pd.concat([prediction_table(val_df, p, classes, fold.name, name)
                                   for name, p in probes.predict(features[fold.val_idx]).items()], ignore_index=True)
                model = with_linear_head(backbone, probes.linear_head())
            else:
                model, histories[fold.name] = fit_model(df.iloc[fold.train_idx], cfg, classes, device, seed + k)
                probs = predict_probs(model, val_df, classes, cfg.image_size, cfg.autocontrast, device, cfg.batch_size,
                                      cfg.plate_crop)
                table = prediction_table(val_df, probs, classes, fold.name)
            table = table.assign(leaky=leaky)
            tables.append(table)
            costs.append(cost_since(fold_started, device))
            fold_meta = {**meta, "fold": fold.name, "seed": seed}
            if cfg.split == "holdout":
                save_checkpoint(seed_dir / "model.pt", model, cfg.arch, classes, cfg.image_size, cfg.autocontrast, fold_meta)
                continue
            fold_dir = seed_dir / "folds" / fold.name
            save_checkpoint(fold_dir / "model.pt", model, cfg.arch, classes, cfg.image_size, cfg.autocontrast, fold_meta)
            fold_metrics = {**base, "seed": seed, "fold": fold.name, "resources": costs[-1],  # this fold's own cost
                            **evaluate_by_classifier(table, classes, 0, seed, **metric_options)}
            (fold_dir / "metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
            fold_scores.append(fold_metrics)

        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics = write_report(seed_dir, pd.concat(tables, ignore_index=True), classes, cfg.bootstrap, seed,
                               {**base, "seed": seed, "fold": "holdout" if cfg.split == "holdout" else "pooled",
                                "resources": total_cost(costs), "folds": fold_info, "train_history": histories, "note": note}, **metric_options)
        if cfg.split == "holdout":
            fold_scores.append(metrics)
        pooled[seed] = seed_dir / "predictions.csv"
        if cfg.tracking == "mlflow":
            log_to_mlflow(seed_dir, cfg.run_name + (f"/seed{seed}" if cfg.seeds else ""), metrics)
        log.info("%sseed %d: image acc=%.3f macro-F1=%.3f | isolate acc=%.3f macro-F1=%.3f",
                 "[leaky, comparison only] " if leaky else "", seed,
                 metrics["image_level"]["accuracy"], metrics["image_level"]["macro"]["f1"],
                 metrics["isolate_level"]["accuracy"], metrics["isolate_level"]["macro"]["f1"])
    write_summary(run_dir, base, fold_scores, pooled)
    if cfg.split != "holdout" and cfg.fit_final:
        log.info("final model on all %d groups", df["group"].nunique())
        if probe:
            model = with_linear_head(backbone, _fit_probes(df, features, classes).linear_head())
        else:
            model, _ = fit_model(df, cfg, classes, device, cfg.seed)
        save_checkpoint(run_dir / "model.pt", model, cfg.arch, classes, cfg.image_size, cfg.autocontrast, {**meta, "fold": "all"})
    log.info("done in %.0fs -> %s", time.time() - started, run_dir)
    return run_dir


def start_cost(device: str) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def cost_since(started: float, device: str) -> dict:
    """What a fold cost: wall time, GPU-minutes (wall minutes on a GPU, 0 on CPU) and peak GPU memory."""
    seconds = time.perf_counter() - started
    cuda = torch.device(device).type == "cuda"
    return {"device": device, "wall_seconds": seconds, "gpu_minutes": seconds / 60 if cuda else 0.0,
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else 0.0}


def total_cost(costs: list[dict]) -> dict:
    return {"device": costs[0]["device"], "wall_seconds": sum(c["wall_seconds"] for c in costs),
            "gpu_minutes": sum(c["gpu_minutes"] for c in costs),
            "peak_memory_mb": max(c["peak_memory_mb"] for c in costs)}


def _fit_probes(train_df: pd.DataFrame, train_x: np.ndarray, classes: list[str]) -> Probes:
    refuse_held_out(train_df)  # the same guard every training loader applies
    class_idx = {c: i for i, c in enumerate(classes)}
    return fit_probes(train_x, train_df["species"].map(class_idx).to_numpy(), len(classes))


def fold_record(df: pd.DataFrame, folds: list[Fold]) -> dict:
    """Fold names, and a hash of which images each fold validates, so two runs' folds can be shown identical."""
    lines = sorted(f"{df['group'].iat[i]}\t{Path(df['image_path'].iat[i]).name}\t{fold.name}"
                   for fold in folds for i in fold.val_idx)
    return {"names": [f.name for f in folds],
            "membership_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()}


def _mean_sd(values: list[float]) -> dict:
    finite = [v for v in values if not np.isnan(v)]
    return {"mean": float(np.mean(finite)) if finite else float("nan"),
            "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"), "n": len(finite)}


def write_summary(run_dir: Path, base: dict, fold_scores: list[dict], pooled: dict[int, Path]) -> dict:
    """Mean and SD of isolate macro-F1 and accuracy over every fold of every seed, and each seed's pooled table."""
    summary: dict = {k: base[k] for k in ("run_name", "cell", "split", "leaky")}
    for key, read in (("isolate_macro_f1", lambda m: m["isolate_level"]["macro"]["f1"]),
                      ("isolate_accuracy", lambda m: m["isolate_level"]["accuracy"])):
        values = [read(m) for m in fold_scores]
        summary[key] = {**_mean_sd(values),
                        "values": [{"seed": m["seed"], "fold": m["fold"], "value": v} for m, v in zip(fold_scores, values)]}
    summary["provenance"] = {**base["provenance"], "pooled_predictions": {
        str(seed): {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path)}
        for seed, path in pooled.items()}}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoint(checkpoint: str | Path, manifest: str | Path, out_dir: str | Path, modality: str | None = None,
                        source: str = "all", n_boot: int = 2000, device: str = "auto", seed: int = 0,
                        **metric_options) -> dict:
    """Score a saved model on a manifest it was not trained on (e.g. a later external test set).

    metric_options as for write_report; with a label_map the manifest's species are reference
    labels mapped onto the model's classes, so they need not be model classes themselves."""
    device = resolve_device(device)
    model, meta = load_checkpoint(checkpoint, device)
    classes = meta["classes"]
    df = select(load_manifest(manifest), modality or meta.get("modality", "all"), source)
    if not metric_options.get("label_map"):
        _resolve_classes(df, tuple(classes))
        metric_options["genus_map"] = genus_map_of(df, metric_options.get("genus_map", {}))
    probs = predict_probs(model, df, classes, meta["image_size"], meta["autocontrast"], device,
                          plate_crop=meta.get("plate_crop", False))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_report(out_dir, prediction_table(df, probs, classes, "eval"), classes, n_boot, seed,
                        {"checkpoint": str(checkpoint), "manifest": str(manifest),
                         "provenance": collect(manifest, checkpoint=checkpoint)}, **metric_options)
