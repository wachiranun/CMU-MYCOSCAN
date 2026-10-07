import hashlib
import json
import logging
import re

import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.data import make_loader
from mycoscan.manifest import load_manifest, select
from mycoscan.pipeline import run_training
from mycoscan.splits import make_folds, partition_pools


def _groups(per_class=20, classes=("Flavi", "Nigri", "Alternaria", "Penicillium", "Rhizopus")):
    rows = []
    for c in classes:
        for g in range(per_class):
            modality = "colony" if g % 2 else "microscopic"
            for shot in range(3):
                rows.append({"species": c, "group": f"{c}-{g}", "modality": modality, "image_path": f"{c}-{g}-{shot}.png"})
    return pd.DataFrame(rows)


def test_pool_b_holds_thirty_percent_of_groups_per_class_and_modality():
    pools = partition_pools(_groups(), b_fraction=0.3, n_folds=5, seed=0)
    assert pools["group"].is_unique
    assert set(pools["pool"]) == {"A", "B"}
    assert pools[pools["pool"] == "B"].groupby("species").size().tolist() == [6] * 5
    assert pools[pools["pool"] == "B"].groupby(["species", "modality"]).size().tolist() == [3] * 10


def test_pool_a_groups_get_five_folds_and_pool_b_none():
    pools = partition_pools(_groups(), b_fraction=0.3, n_folds=5, seed=0)
    a, b = pools[pools["pool"] == "A"], pools[pools["pool"] == "B"]
    assert (b["fold"] == "").all()
    assert sorted(a["fold"].unique()) == ["0", "1", "2", "3", "4"]
    assert a.groupby("fold").size().tolist() == [14] * 5


def _partition(manifest, out):
    main(["partition", "--manifest", str(manifest), "--out", str(out)])


def test_partition_command_writes_file_and_sidecar_once(synthetic_manifest, tmp_path):
    out = tmp_path / "splits_v1.csv"
    _partition(synthetic_manifest, out)
    sidecar = tmp_path / "splits_v1.csv.sha256"
    assert sidecar.read_text().split()[0] == hashlib.sha256(out.read_bytes()).hexdigest()
    splits = pd.read_csv(out, dtype=str, keep_default_na=False)
    openfungi = select(load_manifest(synthetic_manifest), "all", "openfungi")
    assert set(splits["group"]) == set(openfungi["group"])
    assert (splits["manifest_sha256"] == hashlib.sha256(synthetic_manifest.read_bytes()).hexdigest()).all()

    before = (out.read_bytes(), sidecar.read_bytes())
    with pytest.raises(FileExistsError, match="splits_v1.csv"):
        _partition(synthetic_manifest, out)
    assert (out.read_bytes(), sidecar.read_bytes()) == before


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="frozen", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="all",
                source="openfungi", arch="resnet18", weights="none", image_size=64, epochs=1, batch_size=8,
                bootstrap=10, device="cpu", fit_final=False)
    return Config(**{**base, **kw})


def test_run_refuses_a_manifest_whose_hash_differs_from_the_splits_file(synthetic_manifest, tmp_path):
    _partition(synthetic_manifest, tmp_path / "splits.csv")
    edited = synthetic_manifest.with_name("manifest_edited.csv")
    edited.write_text(synthetic_manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        run_training(_cfg(edited, tmp_path, splits_file=str(tmp_path / "splits.csv")))


def test_training_loader_refuses_a_pool_b_row(synthetic_manifest):
    df = select(load_manifest(synthetic_manifest), "all", "openfungi").head(4).copy()
    df["pool"] = ["A", "A", "B", "A"]
    classes = {s: i for i, s in enumerate(sorted(df["species"].unique()))}
    with pytest.raises(ValueError, match=re.escape(df["image_path"].iloc[2])):
        make_loader(df, classes, 64, True, train=True, batch_size=2, num_workers=0)
    make_loader(df, classes, 64, True, train=False, batch_size=2, num_workers=0)


def test_grouped_cv_inside_pool_a_validates_each_group_once_and_never_sees_pool_b(synthetic_manifest, tmp_path):
    _partition(synthetic_manifest, tmp_path / "splits.csv")
    splits = pd.read_csv(tmp_path / "splits.csv", dtype=str, keep_default_na=False)
    pool_a = set(splits.loc[splits["pool"] == "A", "group"])
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="kfold", n_folds=5,
                                splits_file=str(tmp_path / "splits.csv")))
    preds = pd.read_csv(run_dir / "predictions.csv", dtype=str, keep_default_na=False)
    assert set(preds["group"]) == pool_a
    assert (preds.groupby("group")["fold"].nunique() == 1).all()
    assert preds["fold"].nunique() == 5
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert all(f["train_groups"] + f["val_groups"] == len(pool_a) for f in metrics["folds"])
    assert metrics["leaky"] is False


def test_image_random_split_is_marked_leaky_everywhere(synthetic_manifest, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="mycoscan"):
        run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="image_random", n_folds=2))
    assert json.loads((run_dir / "metrics.json").read_text())["leaky"] is True
    assert json.loads((run_dir / "config.json").read_text())["leaky"] is True
    assert any("leaky, comparison only" in r.getMessage() for r in caplog.records)


def test_image_random_folds_split_groups_across_train_and_val():
    df = _groups(per_class=4)
    folds = make_folds(df, "image_random", n_folds=3, seed=0)
    assert sorted(i for f in folds for i in f.val_idx) == list(range(len(df)))
    assert any(set(df["group"].iloc[f.train_idx]) & set(df["group"].iloc[f.val_idx]) for f in folds)
