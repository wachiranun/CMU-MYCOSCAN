import hashlib
import re

import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.data import make_loader
from mycoscan.manifest import load_manifest, select
from mycoscan.pipeline import run_training
from mycoscan.splits import apply_splits
from mycoscan.synthetic import make_synthetic


@pytest.fixture(scope="module")
def cmu_manifest(tmp_path_factory):
    """Ten isolates per class, so 15% of a class is a whole number of isolates (1.5 rounds to 2)."""
    return make_synthetic(tmp_path_factory.mktemp("cmu10"), size=32, fovs_per_device=1, openfungi_per_genus=2,
                          isolates_per_class=(10, 10))


def _seal(manifest, out, *extra):
    main(["seal", "--manifest", str(manifest), "--out", str(out), *extra])
    return pd.read_csv(out, dtype=str, keep_default_na=False)


def test_seal_takes_fifteen_percent_of_each_class_and_writes_the_hash_sidecar(cmu_manifest, tmp_path):
    out = tmp_path / "cmu_splits.csv"
    sealed = _seal(cmu_manifest, out)
    cmu = select(load_manifest(cmu_manifest), "all", "cmu")
    assert set(sealed["group"]) == set(cmu["group"])
    assert sealed.groupby("species")["split"].apply(lambda s: (s == "test").sum()).tolist() == [2] * 10
    assert set(sealed["split"]) == {"dev", "test"}
    sidecar = tmp_path / "cmu_splits.csv.sha256"
    assert sidecar.read_text().split()[0] == hashlib.sha256(out.read_bytes()).hexdigest()


def _first_isolates(manifest, n, name):
    """A manifest beside `manifest` holding only the first n isolates of each class: an earlier imaging batch."""
    rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    rank = pd.to_numeric(rows["isolate_id"].str[-1], errors="coerce")  # CMU<class><i>; blank for OpenFungi
    path = manifest.with_name(name)
    rows[(rows["source"] != "cmu") | (rank < n)].to_csv(path, index=False)
    return path


def test_resealing_after_new_isolates_assigns_only_the_new_ones(cmu_manifest, tmp_path):
    out = tmp_path / "cmu_splits.csv"
    first = _seal(_first_isolates(cmu_manifest, 8, "first_batch.csv"), out)
    assert len(first) == 80
    second = _seal(cmu_manifest, out)
    assert len(second) == 100
    kept = second.set_index("group").loc[first["group"], ["split", "fold"]]
    assert kept.reset_index().equals(first[["group", "split", "fold"]])
    assert second.groupby("species")["split"].apply(lambda s: (s == "test").sum()).tolist() == [2] * 10


def test_dev_folds_spread_each_class_as_evenly_as_the_count_allows(cmu_manifest, tmp_path):
    out = tmp_path / "cmu_splits.csv"
    _seal(_first_isolates(cmu_manifest, 8, "first_batch.csv"), out)
    sealed = _seal(cmu_manifest, out)
    dev = sealed[sealed["split"] == "dev"]
    per_fold = dev.groupby(["species", "fold"]).size().unstack(fill_value=0)
    assert list(per_fold.columns) == ["0", "1", "2", "3", "4"]
    assert ((per_fold.max(axis=1) - per_fold.min(axis=1)) <= 1).all()
    assert (sealed.loc[sealed["split"] == "test", "fold"] == "").all()


def test_training_with_the_sealed_file_validates_only_dev_isolates_and_the_guard_refuses_test_rows(cmu_manifest,
                                                                                                  tmp_path):
    out = tmp_path / "cmu_splits.csv"
    sealed = _seal(cmu_manifest, out)
    run_dir = run_training(Config(run_name="sealed", manifest=str(cmu_manifest), output_dir=str(tmp_path / "runs"),
                                  modality="microscopic", source="cmu", arch="resnet18", weights="none", image_size=32,
                                  epochs=1, batch_size=16, bootstrap=0, device="cpu", fit_final=False,
                                  splits_file=str(out)))
    preds = pd.read_csv(run_dir / "predictions.csv", dtype=str, keep_default_na=False)
    assert set(preds["group"]) == set(sealed.loc[sealed["split"] == "dev", "group"])
    folds = sealed.set_index("group")["fold"]
    assert (preds["fold"] == "fold" + preds["group"].map(folds)).all()

    df = apply_splits(select(load_manifest(cmu_manifest), "microscopic", "cmu"), sealed)
    test_row = df[df["split"] == "test"].head(1)
    mixed = pd.concat([df[df["split"] == "dev"].head(3), test_row])
    classes = {s: i for i, s in enumerate(sorted(df["species"].unique()))}
    with pytest.raises(ValueError, match=re.escape(test_row["image_path"].iloc[0])):
        make_loader(mixed, classes, 32, True, train=True, batch_size=2, num_workers=0)


def test_resealing_with_a_different_fold_count_is_refused(cmu_manifest, tmp_path):
    out = tmp_path / "cmu_splits.csv"
    _seal(_first_isolates(cmu_manifest, 8, "first_batch.csv"), out)
    before = out.read_bytes()
    with pytest.raises(ValueError, match="n-folds"):
        _seal(cmu_manifest, out, "--n-folds", "3")
    assert out.read_bytes() == before


def test_seal_refuses_classes_under_eight_isolates_and_names_them(cmu_manifest, tmp_path):
    few = _first_isolates(cmu_manifest, 7, "seven.csv")
    with pytest.raises(ValueError, match=r"Aspergillus_flavus \(7\).*--force"):
        _seal(few, tmp_path / "s.csv")
    assert not (tmp_path / "s.csv").exists()
    assert len(_seal(few, tmp_path / "s.csv", "--force")) == 70
