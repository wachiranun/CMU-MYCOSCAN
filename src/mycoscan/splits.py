"""Isolate-level splitting.

Folds are computed on the isolate table (one row per isolate) and only then
mapped back to image rows, so every FOV, Z-plane, device, view and timepoint of
an isolate always lands on the same side of a split.
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


def _isolate_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("isolate_id", sort=True)["species"].first().reset_index()


def _to_image_fold(df: pd.DataFrame, isolates: pd.DataFrame, name: str, val_iso_idx) -> Fold:
    val_isolates = set(isolates["isolate_id"].iloc[val_iso_idx])
    is_val = df["isolate_id"].isin(val_isolates).to_numpy()
    return Fold(name, np.flatnonzero(~is_val), np.flatnonzero(is_val))


def _stratified_isolate_folds(isolates: pd.DataFrame, n_splits: int, seed: int) -> list[np.ndarray]:
    # Deal isolates class by class onto folds like cards, continuing the deal
    # across classes. sklearn's StratifiedKFold refuses n_splits larger than
    # every class count, which is the normal case here (2-3 isolates per class).
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(isolates), dtype=int)
    position = 0
    for species in rng.permutation(isolates["species"].unique()):
        members = rng.permutation(np.flatnonzero(isolates["species"].to_numpy() == species))
        fold_of[members] = (position + np.arange(len(members))) % n_splits
        position += len(members)
    return [np.flatnonzero(fold_of == k) for k in range(n_splits)]


def make_folds(df: pd.DataFrame, strategy: str, n_folds: int = 5, val_fraction: float = 0.2, seed: int = 42) -> list[Fold]:
    """holdout: one stratified isolate split with ~val_fraction of isolates in validation.
    kfold:   n_folds stratified isolate folds (each isolate validated exactly once).
    loio:    leave-one-isolate-out.
    """
    isolates = _isolate_table(df)
    if strategy == "holdout":
        n_splits = max(2, round(1 / val_fraction))
        val = _stratified_isolate_folds(isolates, n_splits, seed)[0]
        return [_to_image_fold(df, isolates, "holdout", val)]
    if strategy == "kfold":
        vals = _stratified_isolate_folds(isolates, n_folds, seed)
        return [_to_image_fold(df, isolates, f"fold{i}", v) for i, v in enumerate(vals)]
    if strategy == "loio":
        return [_to_image_fold(df, isolates, f"loio_{iso}", [i]) for i, iso in enumerate(isolates["isolate_id"])]
    raise ValueError(f"unknown split strategy {strategy!r}")


def assert_no_isolate_leakage(df: pd.DataFrame, fold: Fold) -> None:
    train = set(df["isolate_id"].iloc[fold.train_idx])
    val = set(df["isolate_id"].iloc[fold.val_idx])
    shared = train & val
    if shared:
        raise AssertionError(f"{fold.name}: isolates in both train and val: {sorted(shared)[:5]}")
    if len(fold.train_idx) + len(fold.val_idx) != len(df):
        raise AssertionError(f"{fold.name}: fold does not cover every image exactly once")


def describe_fold(df: pd.DataFrame, fold: Fold, classes: list[str]) -> dict:
    val = df.iloc[fold.val_idx]
    train = df.iloc[fold.train_idx]
    val_isolates_per_class = val.groupby("species")["isolate_id"].nunique()
    return {
        "fold": fold.name,
        "train_images": len(train),
        "val_images": len(val),
        "train_isolates": train["isolate_id"].nunique(),
        "val_isolates": val["isolate_id"].nunique(),
        "val_classes_without_isolates": [c for c in classes if val_isolates_per_class.get(c, 0) == 0],
    }
