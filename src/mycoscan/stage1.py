"""Stage-1 checkpoint export: the OpenFungi-adapted backbone Plan 2 starts from.

`mycoscan export-stage1 --config <final recipe>.toml` trains the config's backbone on
every Pool A image of one modality (micro: 5 classes, macro: 6) and writes
`<output_dir>/stage1/of_<micro|macro>_<arch>.pt` without its head, plus a JSON with its
provenance beside it. Pool B is held out by the config's splits file; without one the
export refuses to run, and the training loader's guard refuses any Pool B row that
reached it anyway.

A Stage-2 config names the file as `weights`. Loading it checks the arch, leaves only
the head newly initialised, and copies this provenance into the Stage-2 run's own.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .manifest import load_manifest, select
from .models import adapter
from .pipeline import fit_model, resolve_classes, resolve_device, seed_everything
from .provenance import collect
from .splits import apply_splits, held_out_mask, load_splits_file

log = logging.getLogger("mycoscan")
MODALITY_NAMES = {"microscopic": "micro", "colony": "macro"}


def stage1_name(modality: str, arch: str) -> str:
    return f"of_{MODALITY_NAMES[modality]}_{re.sub(r'[^A-Za-z0-9]+', '-', arch)}"


def export_stage1(cfg: Config) -> Path:
    if cfg.source != "openfungi" or cfg.modality not in MODALITY_NAMES:
        raise ValueError(f"export-stage1 trains on OpenFungi Pool A of one modality; config has source={cfg.source} "
                         f"modality={cfg.modality}, expected source='openfungi' and modality in {sorted(MODALITY_NAMES)}")
    if not cfg.splits_file:
        raise ValueError("export-stage1 trains on all of Pool A, so it needs the config's splits_file to hold "
                         "Pool B out; none is set")
    provenance = collect(cfg.manifest, asdict(cfg), cfg.splits_file)
    seed_everything(cfg.seed)
    df = select(load_manifest(cfg.manifest), cfg.modality, cfg.source)
    if cfg.select_classes:
        df = df[df["species"].isin(cfg.select_classes)]
    df = apply_splits(df, load_splits_file(cfg.splits_file, cfg.manifest))
    pool_a = df[~held_out_mask(df)].reset_index(drop=True)
    classes = resolve_classes(pool_a, cfg.classes)
    name = stage1_name(cfg.modality, cfg.arch)
    log.info("%s: %s on %d Pool A images, %d groups, %d classes", name, cfg.arch, len(pool_a),
             pool_a["group"].nunique(), len(classes))
    model, history = fit_model(pool_a, cfg, classes, resolve_device(cfg.device), cfg.seed)
    head = adapter(model).head
    groups = sorted(pool_a["group"].unique().tolist())
    provenance |= {"run_name": cfg.run_name, "stage1_name": name, "classes": classes,
                   "trained_groups": {"n": len(groups), "groups": groups}, "train_history": history}
    out = Path(cfg.output_dir) / "stage1" / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v for k, v in model.state_dict().items() if not k.startswith(head + ".")},
                "arch": cfg.arch, "head_stripped": True, "image_size": cfg.image_size,
                "autocontrast": cfg.autocontrast, "plate_crop": cfg.plate_crop, "modality": cfg.modality,
                "provenance": provenance}, out)
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return out
