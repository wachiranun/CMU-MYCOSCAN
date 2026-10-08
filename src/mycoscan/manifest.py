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
    temperature incubation temperature, integer or blank   (optional)
    phase       mold | yeast | na                 (optional, default na; dimorphic isolates)
    fov_id      field-of-view identifier          (optional)
    z_index     Z-plane, integer or blank         (optional)
    sha256      hash of the image file            (optional)
    batch       imaging batch                     (optional; sealing mixes batches across folds)
    year        year of isolation or imaging      (optional; sealing mixes years across folds)
    split       reserved for the sealed CMU test set; a training loader refuses `test` rows
    fold        leave blank; a run with a splits file fills it from that file

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
    "batch": "", "year": "",
}
ALLOWED = {
    "modality": {"colony", "microscopic"},
    "view": {"obverse", "reverse", "na"},
    "device": {"microscope_camera", "smartphone", "unknown"},
    "source": {"cmu", "openfungi"},
    "phase": {"mold", "yeast", "na"},
}
INTEGER = ("day", "z_index", "temperature", "year")


def load_manifest(path: str | Path, allow_ungrouped: bool = False) -> pd.DataFrame:
    """allow_ungrouped: let OpenFungi rows without a group_id load as one group per image.
    Only the leaky `image_random` comparison split passes it; grouped splits must refuse them."""
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "level" in df.columns:  # a prediction table loads as a manifest of its images
        df = df[(df["level"] != "tile") & (df["image_path"] != "")].reset_index(drop=True)
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

    _require_id(df, path, "isolate_id", source="cmu", required=True)
    _require_id(df, path, "group_id", source="openfungi", required=not allow_ungrouped,
                hint="build pseudo-groups first, or pass allow_ungrouped=True for a leaky image-level split")
    df["group"] = df["isolate_id"].where(df["source"] == "cmu", df["group_id"])
    species_per_group = df.groupby("group")["species"].nunique()
    conflicting = species_per_group[species_per_group > 1].index.tolist()
    if conflicting:
        raise ValueError(f"{path}: groups labelled with more than one species: {conflicting[:5]}")
    for col in INTEGER:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def _require_id(df: pd.DataFrame, path: Path, col: str, source: str, required: bool, hint: str = "") -> None:
    """Rows of `source` must carry `col`; blanks elsewhere (or when not required) become one id per image."""
    df[col] = df[col].str.strip()
    blank = df[col] == ""
    offending = df.loc[blank & (df["source"] == source), "image_path"].tolist()
    if offending and required:
        raise ValueError(f"{path}: {len(offending)} {source} images have no {col}, e.g. {offending[:3]}"
                         + (f"; {hint}" if hint else ""))
    df.loc[blank, col] = "img:" + df.loc[blank, "image_path"]


def select(df: pd.DataFrame, modality: str, source: str) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    if modality != "all":
        keep &= df["modality"] == modality
    if source != "all":
        keep &= df["source"] == source
    return df[keep].reset_index(drop=True)
