import json

import numpy as np
import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.fusion import fuse_runs
from mycoscan.paired import PairedConfig, compare_runs
from mycoscan.results import run_rows

CLASSES = ["a", "b", "c"]
FOLDS = ["fold0", "fold1", "fold2", "fold3"]
DEV = [f"i{k:02d}" for k in range(40)]
TEST = [f"t{k:02d}" for k in range(12)]


def _species(group):
    return CLASSES[int(group[1:]) % 3]


def _one_hot(index):
    row = np.zeros(len(CLASSES))
    row[index] = 1.0
    return row


def _perfect(group, rng):
    return _one_hot(CLASSES.index(_species(group)))


def _random(group, rng):
    return _one_hot(rng.integers(len(CLASSES)))


def _rows(groups, probs, rng, fold_of):
    """Two images per isolate, each with the isolate's probabilities."""
    rows = []
    for g in groups:
        p = probs(g, rng)
        for shot in range(2):
            rows.append({"image_path": f"{g}_{shot}.png", "species": _species(g), "group": g, "fold": fold_of(g),
                         "classifier": "network", "level": "image",
                         **{f"prob_{c}": v for c, v in zip(CLASSES, p)}})
    return pd.DataFrame(rows)


def _branch(root, name, probs, weights="imagenet", held_out="h0", classes=CLASSES, seed=0):
    """A run directory as run_training leaves it, with out-of-fold isolate predictions from `probs`."""
    run = root / name
    run.mkdir(parents=True)
    preds = _rows(DEV, probs, np.random.default_rng(seed), lambda g: FOLDS[int(g[1:]) % len(FOLDS)])
    preds = preds.rename(columns={f"prob_{c}": f"prob_{k}" for c, k in zip(CLASSES, classes)})
    preds.to_csv(run / "predictions.csv", index=False)
    (run / "config.json").write_text(json.dumps({"run_name": name, "arch": "resnet18", "weights": weights,
                                                 "pooling": "mean", "split": "kfold"}))
    (run / "summary.json").write_text(json.dumps({
        "run_name": name, "isolate_macro_f1": {"values": []},
        "provenance": {"seeds": [42], "splits_sha256": "s1", "held_out": {"groups": len(TEST), "sha256": held_out},
                       "folds": {"names": FOLDS, "membership_sha256": f"{name[:5]}-all",
                                 "per_fold_sha256": {f: f"{name[:5]}-{f}" for f in FOLDS}},
                       "pooled_predictions": {"42": {"path": "predictions.csv", "sha256": "x"}}}}))
    return run


def _test_dir(root, name, probs, groups=TEST, seed=1):
    out = root / name
    out.mkdir(parents=True)
    _rows(groups, probs, np.random.default_rng(seed), lambda g: "eval").to_csv(out / "predictions.csv", index=False)
    return out


def _json(path):
    return json.loads(path.read_text())


def test_the_tuned_weight_favours_the_perfect_branch_and_fused_accuracy_equals_it(tmp_path):
    colony, micro = _branch(tmp_path, "colony", _perfect), _branch(tmp_path, "micro", _random)
    out = fuse_runs(colony, micro, tmp_path / "fused", colony_test=_test_dir(tmp_path, "ct", _perfect),
                    micro_test=_test_dir(tmp_path, "mt", _random), n_boot=0)
    metrics = _json(out / "metrics.json")
    assert metrics["fusion"]["weight"] > 0.5  # the colony branch's share
    assert metrics["isolate_level"]["accuracy"] == 1.0 and metrics["isolate_level"]["n"] == len(DEV)
    assert all(w > 0.5 for w in metrics["fusion"]["fold_weights"].values())
    test = _json(out / "test" / "metrics.json")
    assert test["isolate_level"]["accuracy"] == 1.0 and test["isolate_level"]["n"] == len(TEST)


def test_the_weight_is_tuned_on_development_rows_only_and_test_rows_are_scored_once_with_it(tmp_path):
    colony, micro = _branch(tmp_path, "colony", _perfect), _branch(tmp_path, "micro", _random)
    weights = []
    for k, (ct, mt) in enumerate([(_perfect, _random), (_random, _perfect)]):
        out = fuse_runs(colony, micro, tmp_path / f"fused{k}", colony_test=_test_dir(tmp_path, f"ct{k}", ct),
                        micro_test=_test_dir(tmp_path, f"mt{k}", mt), n_boot=0)
        fusion, test = _json(out / "metrics.json")["fusion"], _json(out / "test" / "metrics.json")
        assert "development" in fusion["tuned_on"]
        assert test["fusion"]["weight"] == fusion["weight"] and test["fusion"]["test_scored"] == "once"
        preds = pd.read_csv(out / "test" / "predictions.csv")
        assert len(preds) == len(TEST) and (preds["weight"] == fusion["weight"]).all()
        assert (preds["split"] == "test").all()
        weights.append(fusion["weight"])
    assert weights[0] == weights[1]  # test rows that favour the other branch do not move the weight


def test_runs_with_different_classes_or_test_isolates_are_refused_naming_the_difference(tmp_path):
    colony = _branch(tmp_path, "colony", _perfect)
    other = _branch(tmp_path, "renamed", _perfect, classes=["a", "b", "z"])
    with pytest.raises(ValueError, match="'c'.*'z'|'z'.*'c'"):
        fuse_runs(colony, other, tmp_path / "f1")
    moved = _branch(tmp_path, "moved", _perfect, held_out="h1")
    with pytest.raises(ValueError, match="test isolates.*h0.*h1"):
        fuse_runs(colony, moved, tmp_path / "f2")
    micro = _branch(tmp_path, "micro", _random)
    with pytest.raises(ValueError, match="t11"):
        fuse_runs(colony, micro, tmp_path / "f3", colony_test=_test_dir(tmp_path, "ct", _perfect),
                  micro_test=_test_dir(tmp_path, "mt", _perfect, groups=TEST[:-1]))


def test_a_fused_run_appears_in_the_results_table_and_the_paired_comparison(tmp_path, capsys):
    stage1 = str(tmp_path / "stage1.pt")
    branches = tmp_path / "branches"
    runs = tmp_path / "runs"
    main(["fuse", "--colony", str(_branch(branches, "colony_seq", _perfect, weights=stage1)),
          "--micro", str(_branch(branches, "micro_seq", _random, weights=stage1)), "--out", str(runs / "fused_seq"),
          "--bootstrap", "0"])
    main(["fuse", "--colony", str(_branch(branches, "colony_direct", _random, seed=3)),
          "--micro", str(_branch(branches, "micro_direct", _random, seed=4)), "--out", str(runs / "fused_direct"),
          "--bootstrap", "0"])
    table = run_rows(runs)
    assert {"fused_seq", "fused_direct"} <= set(table["run_name"])
    folds = table[table["run_name"] == "fused_seq"]["fold"]
    assert set(folds) == {"pooled", *FOLDS}
    (pair,) = compare_runs([runs / "fused_seq", runs / "fused_direct"], PairedConfig(bootstrap=100))
    assert (pair["sequential"], pair["direct"], pair["n_pairs"]) == ("fused_seq", "fused_direct", len(FOLDS))
    assert pair["mean_difference"] > 0


def test_the_mlp_variant_fuses_cached_embeddings_of_synthetic_colony_and_microscopy_runs(synthetic_manifest, tmp_path):
    from mycoscan.config import Config
    from mycoscan.pipeline import run_training

    base = dict(manifest=str(synthetic_manifest), output_dir=str(tmp_path / "runs"), source="cmu", arch="small_cnn",
                weights="none", finetune="full", image_size=32, epochs=1, batch_size=8, bootstrap=0, device="cpu",
                split="kfold", n_folds=2, fit_final=False, tau_rule="none")
    colony = run_training(Config(run_name="colony", modality="colony", **base))
    micro = run_training(Config(run_name="micro", modality="microscopic", **base))
    out = fuse_runs(colony, micro, tmp_path / "fused_mlp", method="mlp", n_boot=0, device="cpu")
    for name in ("config.json", "predictions.csv", "metrics.json", "summary.json", "confusion_isolate_level.png",
                 "folds/fold0/metrics.json", "folds/fold1/metrics.json"):
        assert (out / name).is_file(), name
    preds = pd.read_csv(out / "predictions.csv")
    isolates = pd.read_csv(colony / "predictions.csv")["group"].nunique()
    assert len(preds) == isolates and set(preds["classifier"]) == {"fusion_mlp"}
    metrics = _json(out / "metrics.json")
    assert metrics["isolate_level"]["n"] == isolates and metrics["fusion"]["method"] == "mlp"
    # both branches start from the same backbone, so they share one cache, keyed by image
    assert len(list((tmp_path / "runs" / "feature_cache").glob("*.npz"))) == 1
