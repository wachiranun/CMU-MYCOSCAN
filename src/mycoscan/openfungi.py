"""OpenFungi manifest builder: pseudo-groups of repeated shots of one plate.

OpenFungi has no isolate identifiers, and many of its photos are repeated shots of
one plate. This module walks `<root>/macro/<class>/` (colony photos) and
`<root>/micro/<class>/` (micrographs), and within each modality and class links two
images when both signals say they are the same plate:

- the Hamming distance between their perceptual hashes (pHash) is at most
  `hamming_threshold` (default 8), and
- the cosine distance between their frozen embeddings (DINO by default, from the
  same backbone registry training uses) is below `cosine_threshold` (default 0.15).

Groups are the connected components of those links (single-linkage agglomerative
clustering). Requiring both signals is deliberate: single linkage on either one
alone chains look-alike plates of a class into one group, the embedding especially,
since different plates of one species photographed in one setup sit close together.

A contact sheet is written for every group with more than one image, so a
mycologist can confirm that the grouped photos really are one plate.
"""
from __future__ import annotations

import os
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.sparse.csgraph import connected_components

from .config import parse_override
from .features import extract_features
from .models import build_model
from .provenance import sha256_file
from .transforms import load_image

MODALITIES = {"macro": "colony", "micro": "microscopic"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MIXED = "Mixed"
THUMB = 160


@dataclass(frozen=True)
class GroupingConfig:
    hamming_threshold: int = 8
    cosine_threshold: float = 0.15
    include_mixed: bool = False
    embed_arch: str = "vit_small_patch14_dinov2"
    embed_weights: str = "dino"
    embed_image_size: int = 224
    batch_size: int = 32
    device: str = "auto"
    min_images: int = 20  # classes with fewer images are flagged as under-powered
    contact_sheets: bool = True


def load_grouping_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> GroupingConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    raw.update(dict(parse_override(o) for o in overrides))
    unknown = sorted(set(raw) - {f.name for f in fields(GroupingConfig)})
    if unknown:
        raise ValueError(f"unknown grouping config keys: {unknown}")
    return GroupingConfig(**raw)


def _scan(root: Path, include_mixed: bool) -> pd.DataFrame:
    rows = []
    for folder, modality in MODALITIES.items():
        for class_dir in sorted(p for p in (root / folder).glob("*") if p.is_dir()):
            if class_dir.name == MIXED and not include_mixed:
                continue
            for path in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
                rows.append({"path": path, "species": class_dir.name, "modality": modality, "folder": folder})
    if not rows:
        raise ValueError(f"no images under {root}/macro/<class>/ or {root}/micro/<class>/")
    return pd.DataFrame(rows)


def _groups(hashes: np.ndarray, embeddings: np.ndarray, cfg: GroupingConfig) -> np.ndarray:
    """Component label per image: linked when pHash and embedding both say same plate.
    `hashes` holds each 64-bit pHash as one uint64, so distances cost n x n words, not n x n x 64 bits."""
    hamming = np.bitwise_count(hashes[:, None] ^ hashes[None, :])
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    cosine = 1 - unit @ unit.T
    linked = (hamming <= cfg.hamming_threshold) & (cosine < cfg.cosine_threshold)
    return connected_components(linked, directed=False)[1]


def _relative(path: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:  # another drive on Windows
        return path.resolve().as_posix()


def _contact_sheet(paths: list[Path], out: Path) -> None:
    cols = min(4, len(paths))
    rows = -(-len(paths) // cols)
    sheet = Image.new("RGB", (cols * THUMB, rows * (THUMB + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        thumb = load_image(path, False, "")
        thumb.thumbnail((THUMB, THUMB))
        x, y = (i % cols) * THUMB, (i // cols) * (THUMB + 14)
        sheet.paste(thumb, (x, y))
        draw.text((x + 2, y + THUMB), path.name[:26], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def build_openfungi_manifest(root: str | Path, out: str | Path,
                             cfg: GroupingConfig = GroupingConfig()) -> tuple[Path, dict]:
    """Write the manifest (image paths relative to it), contact sheets beside it, and return a summary."""
    from .pipeline import resolve_device

    root, out = Path(root), Path(out)
    images = _scan(root, cfg.include_mixed)
    images["image_path"] = [str(p.resolve()) for p in images["path"]]
    sizes, hashes = [], []
    for path, modality in zip(images["path"], images["modality"]):
        img = load_image(path, False, modality)
        sizes.append(img.size)
        hashes.append(imagehash.phash(img))
    images["width"], images["height"] = zip(*sizes)
    images["phash"] = [str(h) for h in hashes]
    bits = np.array([int(str(h), 16) for h in hashes], dtype=np.uint64)
    model = build_model(cfg.embed_arch, 1, cfg.embed_weights, cfg.embed_image_size)
    embeddings = extract_features(model, images, cfg.embed_image_size, autocontrast=False, plate_crop=False,
                                  device=resolve_device(cfg.device), batch_size=cfg.batch_size)

    images["group_id"] = ""
    sheets_dir = out.parent / "contact_sheets"
    for (folder, species), members in images.groupby(["folder", "species"], sort=True):
        idx = members.index.to_numpy()
        labels = _groups(bits[idx], embeddings[idx], cfg)
        # Number groups in the order of their first image, so ids are stable across rebuilds.
        order = {label: k for k, label in enumerate(dict.fromkeys(labels))}
        ids = [f"OF_{folder}_{species}_{order[label]:03d}" for label in labels]
        images.loc[idx, "group_id"] = ids
        if cfg.contact_sheets:
            for group_id in dict.fromkeys(ids):
                paths = [p for p, g in zip(members["path"], ids) if g == group_id]
                if len(paths) > 1:
                    _contact_sheet(paths, sheets_dir / f"{group_id}.png")

    manifest = pd.DataFrame({
        "image_path": [_relative(p, out.parent) for p in images["path"]],
        "species": images["species"], "genus": images["species"].str.split("_").str[0],
        "isolate_id": "", "group_id": images["group_id"], "modality": images["modality"], "view": "na",
        "device": "unknown", "source": "openfungi", "sha256": [sha256_file(p) for p in images["path"]],
        "phash": images["phash"], "width": images["width"], "height": images["height"],
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    return out, summarize(manifest, cfg)


def summarize(manifest: pd.DataFrame, cfg: GroupingConfig) -> dict:
    """Images and groups per modality and class; classes under `min_images` are flagged as under-powered."""
    classes = []
    for (modality, species), rows in manifest.groupby(["modality", "species"], sort=True):
        classes.append({"modality": modality, "species": species, "images": len(rows),
                        "groups": rows["group_id"].nunique(), "under_powered": len(rows) < cfg.min_images})
    return {"config": asdict(cfg), "classes": classes,
            "under_powered": [f"{c['modality']}/{c['species']}" for c in classes if c["under_powered"]]}
