"""The image manifest: one CSV row per image, parsed and validated at load time.

Columns:
    image_path  relative to the manifest's folder, or absolute
    species     class label
    isolate_id  sequencing-confirmed isolate; the unit of splitting. Required for CMU
                rows. Blank is allowed only for OpenFungi, where each image becomes its own group.
    modality    colony | microscopic
    view        obverse | reverse | na            (optional, default na)
    device      microscope_camera | smartphone | unknown   (optional, default unknown)
    day         growth day, integer or blank     (optional)
    source      cmu | openfungi                   (optional, default cmu)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("image_path", "species", "isolate_id", "modality")
DEFAULTS = {"view": "na", "device": "unknown", "day": "", "source": "cmu"}
ALLOWED = {
    "modality": {"colony", "microscopic"},
    "view": {"obverse", "reverse", "na"},
    "device": {"microscope_camera", "smartphone", "unknown"},
    "source": {"cmu", "openfungi"},
}


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        elif default:
            df[col] = df[col].replace("", default)
    for col, allowed in ALLOWED.items():
        bad = sorted(set(df[col]) - allowed)
        if bad:
            raise ValueError(f"{path}: column {col} has invalid values {bad}; allowed {sorted(allowed)}")

    root = path.parent
    df["image_path"] = [str((root / p).resolve()) for p in df["image_path"]]
    absent = [p for p in df["image_path"] if not Path(p).is_file()]
    if absent:
        raise FileNotFoundError(f"{path}: {len(absent)} images not found, e.g. {absent[:3]}")

    df["isolate_id"] = df["isolate_id"].str.strip()
    no_isolate = df["isolate_id"] == ""
    blank_cmu = df.loc[no_isolate & (df["source"] == "cmu"), "image_path"].tolist()
    if blank_cmu:
        raise ValueError(f"{path}: {len(blank_cmu)} CMU images have no isolate_id, e.g. {blank_cmu[:3]}")
    df.loc[no_isolate, "isolate_id"] = "img:" + df.loc[no_isolate, "image_path"]
    species_per_isolate = df.groupby("isolate_id")["species"].nunique()
    conflicting = species_per_isolate[species_per_isolate > 1].index.tolist()
    if conflicting:
        raise ValueError(f"{path}: isolates labelled with more than one species: {conflicting[:5]}")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def select(df: pd.DataFrame, modality: str, source: str) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if modality != "all":
        keep &= df["modality"] == modality
    if source != "all":
        keep &= df["source"] == source
    return df[keep].reset_index(drop=True)
