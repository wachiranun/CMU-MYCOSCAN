"""Group-level splitting.

The unit is the manifest's `group` column: the isolate for CMU rows, the
pseudo-group of repeated shots for OpenFungi rows. Folds are computed on the
group table (one row per group) and only then mapped back to image rows, so
every FOV, Z-plane, device, view and timepoint of a group always lands on the
same side of a split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    raise ValueError(f"unknown split strategy {strategy!r}")


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
