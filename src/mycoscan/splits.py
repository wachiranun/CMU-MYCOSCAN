"""Group-level splitting.

The unit is the manifest's `group` column: the isolate for CMU rows, the
pseudo-group of repeated shots for OpenFungi rows. Folds are computed on the
group table (one row per group) and only then mapped back to image rows, so
every FOV, Z-plane, device, view and timepoint of a group always lands on the
same side of a split.

The frozen splits file (`splits_v1.csv`) is written once: every OpenFungi group
goes to Pool A (development, with a fold) or Pool B (external test, never
trained on). It records the hash of the manifest it was built from, and a
sidecar records its own hash, so a training run can verify both before using it.
The CMU sealed split file has the same shape with a `split` column instead of
`pool`: each isolate is `dev` (with a fold) or `test` (the locked test set).
Sealing is additive: a later run assigns new isolates and never moves a sealed one.

The only split that ignores groups is `image_random`, kept to reproduce the
leaky image-level number of the OpenFungi paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import load_manifest
from .provenance import sha256_file


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("group", sort=True)["species"].first().reset_index()


def _to_image_fold(df: pd.DataFrame, groups: pd.DataFrame, name: str, val_group_idx) -> Fold:
    val_groups = set(groups["group"].iloc[val_group_idx])
    is_val = df["group"].isin(val_groups).to_numpy()
    return Fold(name, np.flatnonzero(~is_val), np.flatnonzero(is_val))


def _stratified_group_folds(groups: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    # Deal groups class by class onto folds like cards, continuing the deal
    # across classes. sklearn's StratifiedKFold refuses n_splits larger than
    # every class count, which small classes can hit.
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(groups), dtype=int)
    position = 0
    for species in rng.permutation(groups["species"].unique()):
        members = rng.permutation(np.flatnonzero(groups["species"].to_numpy() == species))
        fold_of[members] = (position + np.arange(len(members))) % n_splits
        position += len(members)
    return [np.flatnonzero(fold_of == k) for k in range(n_splits)]


def make_folds(df: pd.DataFrame, strategy: str, n_folds: int = 5, val_fraction: float = 0.2, seed: int = 42) -> list[Fold]:
    """holdout: one stratified group split with ~val_fraction of groups in validation.
    kfold:   n_folds stratified group folds (each group validated exactly once).
    loio:    leave-one-group-out (one isolate for CMU data).
    """
    groups = _group_table(df)
    if strategy == "holdout":
        n_splits = max(2, round(1 / val_fraction))
        val = _stratified_group_folds(groups, n_splits, seed)[0]
        return [_to_image_fold(df, groups, "holdout", val)]
    if strategy == "kfold":
        vals = _stratified_group_folds(groups, n_folds, seed)
        return [_to_image_fold(df, groups, f"fold{i}", v) for i, v in enumerate(vals)]
    if strategy == "loio":
        return [_to_image_fold(df, groups, f"loio_{g}", [i]) for i, g in enumerate(groups["group"])]
    if strategy == "image_random":
        images = df[["species"]].reset_index(drop=True)
        vals = _stratified_group_folds(images, n_folds, seed)
        return [Fold(f"fold{i}", np.setdiff1d(np.arange(len(df)), v), v) for i, v in enumerate(vals)]
    raise ValueError(f"unknown split strategy {strategy!r}")


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def subsample_groups(df: pd.DataFrame, fraction: float, seed: int = 42,
                     strata: tuple[str, ...] = ("species",)) -> tuple[pd.DataFrame, dict]:
    """Keep round(fraction * n) whole groups of each stratum (at least one), chosen at random, so a learning
    curve never splits a group. Strata are classes, or class and fold when a splits file has already frozen
    the folds, so no fold loses a class. The record lists what is kept and what was removed."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("group", sort=True)[list(strata)].first().reset_index()
    keep: set[str] = set()
    for _, members in groups.groupby(list(strata), sort=True):
        n = max(1, _round_half_up(fraction * len(members)))
        keep.update(rng.permutation(members["group"].to_numpy())[:n].tolist())
    kept = df[df["group"].isin(keep)].reset_index(drop=True)
    record = {"train_fraction": fraction,
              "images_per_class": {str(k): int(v) for k, v in kept.groupby("species").size().items()},
              "groups_per_class": {str(k): int(v) for k, v in kept.groupby("species")["group"].nunique().items()},
              "removed_groups": sorted(set(groups["group"]) - keep)}
    return kept, record


def frozen_folds(df: pd.DataFrame) -> list[Fold]:
    """One fold per value of the `fold` column a splits file merged in."""
    return [Fold(f"fold{k}", np.flatnonzero(df["fold"] != k), np.flatnonzero(df["fold"] == k))
            for k in sorted(df["fold"].unique(), key=int)]


def _spread(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k positions evenly spaced over 0..n-1, starting from a random offset."""
    return np.floor((np.arange(k) + rng.uniform()) * n / k).astype(int)


def partition_pools(df: pd.DataFrame, b_fraction: float = 0.3, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """One row per group: pool A or B, and a fold for Pool A groups.

    Within each class, groups are ordered by modality and spread evenly over that order,
    so Pool B takes round(b_fraction * n) groups of every class, split across modalities in proportion.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby("group", sort=True).agg(species=("species", "first"), modality=("modality", "first")).reset_index()
    groups["pool"] = "A"
    for species in sorted(groups["species"].unique()):
        members = groups[groups["species"] == species]
        ordered = members.loc[rng.permutation(members.index)].sort_values("modality", kind="stable").index
        k = round(b_fraction * len(ordered))
        if k:
            groups.loc[ordered[_spread(len(ordered), k, rng)], "pool"] = "B"
    groups["fold"] = ""
    pool_a = groups[groups["pool"] == "A"].reset_index()
    for k, members in enumerate(_stratified_group_folds(pool_a, n_folds, seed)):
        groups.loc[pool_a["index"].iloc[members], "fold"] = str(k)
    return groups


MIN_ISOLATES_TO_SEAL = 8
SEALED_COLUMNS = ["group", "species", "split", "fold"]


def _isolate_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per isolate with what sealing mixes across folds: imaging batch, year and devices."""
    return df.groupby("group", sort=True).agg(
        species=("species", "first"), batch=("batch", "first"), year=("year", "first"),
        devices=("device", lambda d: "+".join(sorted(set(d))))).reset_index()


def seal_groups(df: pd.DataFrame, sealed: pd.DataFrame | None, test_fraction: float = 0.15, n_folds: int = 5,
                seed: int = 0) -> pd.DataFrame:
    """One row per isolate: split `test`, or `dev` with a fold. Rows of `sealed` are kept as they are;
    only isolates new to it are assigned.

    Within each class, new isolates are ordered by batch, year and devices, so test picks spaced evenly
    over that order, and dev isolates dealt in that order onto the class's least-filled fold, mix all three.
    A class's test share is topped up to round(test_fraction * all its isolates), counting sealed ones.
    """
    rng = np.random.default_rng(seed)
    sealed = pd.DataFrame(columns=SEALED_COLUMNS) if sealed is None else sealed[SEALED_COLUMNS]
    isolates = _isolate_table(df)
    new = isolates[~isolates["group"].isin(sealed["group"])]
    dev = sealed[sealed["split"] == "dev"]
    fold_total = [int((dev["fold"] == str(f)).sum()) for f in range(n_folds)]
    rows = []
    for species in sorted(new["species"].unique()):
        members = new[new["species"] == species]
        ordered = members.loc[rng.permutation(members.index)].sort_values(
            ["batch", "year", "devices"], key=lambda col: col.astype(str), kind="stable")
        old = sealed[sealed["species"] == species]
        target = _round_half_up(test_fraction * (len(old) + len(ordered)))
        k = min(max(target - int((old["split"] == "test").sum()), 0), len(ordered))
        test_positions = set(_spread(len(ordered), k, rng).tolist()) if k else set()
        class_dev = old[old["split"] == "dev"]
        per_fold = [int((class_dev["fold"] == str(f)).sum()) for f in range(n_folds)]
        for position, group in enumerate(ordered["group"]):
            if position in test_positions:
                rows.append({"group": group, "species": species, "split": "test", "fold": ""})
                continue
            f = min(range(n_folds), key=lambda f: (per_fold[f], fold_total[f], f))
            per_fold[f] += 1
            fold_total[f] += 1
            rows.append({"group": group, "species": species, "split": "dev", "fold": str(f)})
    return pd.concat([sealed, pd.DataFrame(rows, columns=SEALED_COLUMNS)]).sort_values("group").reset_index(drop=True)


def sidecar_path(splits_path: Path) -> Path:
    return splits_path.with_name(splits_path.name + ".sha256")


def _write_with_sidecar(table: pd.DataFrame, out: Path, mode: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, mode, encoding="utf-8", newline="") as f:
        table.to_csv(f, index=False)
    with open(sidecar_path(out), mode, encoding="utf-8") as f:
        f.write(f"{sha256_file(out)}  {out.name}\n")


def write_splits_file(manifest: str | Path, out: str | Path, b_fraction: float = 0.3, n_folds: int = 5,
                      seed: int = 0) -> Path:
    """Partition the manifest's OpenFungi groups into Pool A / Pool B, once. Refuses to overwrite."""
    out = Path(out)
    for path in (out, sidecar_path(out)):
        if path.exists():
            raise FileExistsError(f"{path} exists; a splits file is created once and never overwritten")
    df = load_manifest(manifest)
    openfungi = df[df["source"] == "openfungi"]
    if openfungi.empty:
        raise ValueError(f"{manifest} has no OpenFungi rows to partition")
    splits = partition_pools(openfungi, b_fraction, n_folds, seed)
    splits["manifest_sha256"] = sha256_file(manifest)
    _write_with_sidecar(splits, out, "x")
    return out


def write_sealed_file(manifest: str | Path, out: str | Path, test_fraction: float = 0.15, n_folds: int = 5,
                      seed: int = 0, force: bool = False) -> Path:
    """Seal the manifest's CMU isolates into dev and test, or extend an existing sealed file with new isolates."""
    out = Path(out)
    df = load_manifest(manifest)
    cmu = df[df["source"] == "cmu"]
    if cmu.empty:
        raise ValueError(f"{manifest} has no CMU rows to seal")
    counts = cmu.groupby("species")["group"].nunique()
    short = counts[counts < MIN_ISOLATES_TO_SEAL]
    if len(short) and not force:
        listed = ", ".join(f"{species} ({n})" for species, n in short.items())
        raise ValueError(f"classes with fewer than {MIN_ISOLATES_TO_SEAL} isolates: {listed}; pass --force to seal anyway")
    sealed = None
    if out.exists():
        sealed = _read_verified(out)
        if "split" not in sealed:
            raise ValueError(f"{out} has no split column; it is not a sealed CMU split file")
        sealed_folds = sorted(set(sealed.loc[sealed["split"] == "dev", "fold"]), key=int)
        if any(int(f) >= n_folds for f in sealed_folds) or len(sealed_folds) < min(n_folds, len(sealed)):
            raise ValueError(f"{out} was sealed with folds {sealed_folds}; reseal with the same --n-folds")
    table = seal_groups(cmu, sealed, test_fraction, n_folds, seed)
    table["manifest_sha256"] = sha256_file(manifest)
    _write_with_sidecar(table, out, "w")
    return out


def _read_verified(path: Path) -> pd.DataFrame:
    recorded = sidecar_path(path).read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != recorded:
        raise ValueError(f"{path} does not match the hash in {sidecar_path(path).name}; the splits file was modified")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_splits_file(path: str | Path, manifest: str | Path) -> pd.DataFrame:
    """The splits table, after checking it is unmodified and was built from this exact manifest."""
    path = Path(path)
    splits = _read_verified(path)
    built_from = set(splits["manifest_sha256"])
    actual = sha256_file(manifest)
    if built_from != {actual}:
        raise ValueError(f"manifest {manifest} has hash {actual}, but {path} was built from a manifest with hash "
                         f"{sorted(built_from)}; refusing to reuse a frozen split on a changed manifest")
    return splits


def apply_splits(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Merge each group's pool (OpenFungi) or split (CMU), and fold, into the image rows.
    Every group must be in the splits file."""
    unknown = sorted(set(df["group"]) - set(splits["group"]))
    if unknown:
        raise ValueError(f"{len(unknown)} groups are not in the splits file, e.g. {unknown[:5]}")
    by_group = splits.set_index("group")
    return df.assign(**{col: df["group"].map(by_group[col]) for col in ("pool", "split", "fold") if col in by_group})


def assert_no_group_leakage(df: pd.DataFrame, fold: Fold) -> None:
    train = set(df["group"].iloc[fold.train_idx])
    val = set(df["group"].iloc[fold.val_idx])
    shared = train & val
    if shared:
        raise AssertionError(f"{fold.name}: groups in both train and val: {sorted(shared)[:5]}")
    if len(fold.train_idx) + len(fold.val_idx) != len(df):
        raise AssertionError(f"{fold.name}: fold does not cover every image exactly once")


def describe_fold(df: pd.DataFrame, fold: Fold, classes: list[str]) -> dict:
    val = df.iloc[fold.val_idx]
    train = df.iloc[fold.train_idx]
    val_groups_per_class = val.groupby("species")["group"].nunique()
    return {
        "fold": fold.name,
        "train_images": len(train),
        "val_images": len(val),
        "train_groups": train["group"].nunique(),
        "val_groups": val["group"].nunique(),
        "val_classes_without_groups": [c for c in classes if val_groups_per_class.get(c, 0) == 0],
    }
