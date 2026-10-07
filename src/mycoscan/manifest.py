"""The image manifest: one CSV row per image, parsed and validated at load time.

Columns:
    image_path  relative to the manifest's folder, or absolute
    species     class label
    isolate_id  sequencing-confirmed isolate. Required for CMU rows; blank for OpenFungi.
    group_id    pseudo-group of repeated shots of one plate. Required for OpenFungi rows
                (built by the OpenFungi manifest builder); ignored for CMU rows.
    modality    colony | microscopic
    view        obverse | reverse | na            (optional, default na)
    device      microscope_camera | smartphone | unknown   (optional, default unknown)
    day         growth day, integer or blank     (optional)
    source      cmu | openfungi                   (optional, default cmu)
    genus       taxonomic rollup of species       (optional)
    temperature incubation temperature            (optional)
    phase       mold | yeast | na                 (optional, default na; dimorphic isolates)
    fov_id      field-of-view identifier          (optional)
    z_index     Z-plane, integer or blank         (optional)
    sha256      hash of the image file            (optional)
    split, fold assigned by the splits file       (optional)

Loading adds one derived column, `group`: the unit of splitting, sampling and
bootstrapping. It is `isolate_id` for CMU rows and `group_id` for OpenFungi rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("image_path", "species", "isolate_id", "modality")
DEFAULTS = {
    "view": "na", "device": "unknown", "day": "", "source": "cmu", "group_id": "",
    "genus": "", "temperature": "", "phase": "na", "fov_id": "", "z_index": "", "sha256": "", "split": "", "fold": "",
}
ALLOWED = {
    "modality": {"colony", "microscopic"},
    "view": {"obverse", "reverse", "na"},
    "device": {"microscope_camera", "smartphone", "unknown"},
    "source": {"cmu", "openfungi"},
    "phase": {"mold", "yeast", "na"},
}
INTEGER = ("day", "z_index")


def load_manifest(path: str | Path, allow_ungrouped: bool = False) -> pd.DataFrame:
    """allow_ungrouped: let OpenFungi rows without a group_id load as one group per image.
    Only for the leaky image-level comparison split; grouped splits must refuse them."""
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
    df["group_id"] = df["group_id"].str.strip()
    ungrouped = (df["source"] == "openfungi") & (df["group_id"] == "")
    if ungrouped.any() and not allow_ungrouped:
        examples = df.loc[ungrouped, "image_path"].tolist()
        raise ValueError(f"{path}: {len(examples)} OpenFungi images have no group_id, e.g. {examples[:3]}; "
                         "build pseudo-groups first, or pass allow_ungrouped=True for a leaky image-level split")
    df.loc[ungrouped, "group_id"] = "img:" + df.loc[ungrouped, "image_path"]
    df["group"] = df["isolate_id"].where(df["source"] == "cmu", df["group_id"])
    species_per_isolate = df.groupby("isolate_id")["species"].nunique()
    conflicting = species_per_isolate[species_per_isolate > 1].index.tolist()
    if conflicting:
        raise ValueError(f"{path}: isolates labelled with more than one species: {conflicting[:5]}")
    for col in INTEGER:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def select(df: pd.DataFrame, modality: str, source: str) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if modality != "all":
        keep &= df["modality"] == modality
    if source != "all":
        keep &= df["source"] == source
    return df[keep].reset_index(drop=True)
