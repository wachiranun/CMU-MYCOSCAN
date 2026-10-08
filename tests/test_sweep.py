import json

import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.pipeline import run_training
from mycoscan.splits import subsample_groups
from mycoscan.sweep import run_sweep


def _groups(per_class, classes=("A", "B", "C")):
    return pd.DataFrame([{"species": c, "group": f"{c}-{g}", "image_path": f"{c}-{g}-{shot}.png"}
                         for c in classes for g in range(per_class[c] if isinstance(per_class, dict) else per_class)
                         for shot in range(1 + g % 3)])


def test_half_fraction_removes_half_the_groups_of_every_class_and_never_splits_one():
    df = _groups({"A": 20, "B": 10, "C": 5})
    kept, record = subsample_groups(df, 0.5, seed=0)
    assert record["groups_per_class"] == {"A": 10, "B": 5, "C": 3}  # round half up: 2.5 -> 3
    removed = set(record["removed_groups"])
    assert removed == set(df["group"]) - set(kept["group"])
    assert len(kept) + len(df[df["group"].isin(removed)]) == len(df)
    assert record["images_per_class"] == kept.groupby("species").size().to_dict()


def test_with_frozen_folds_subsampling_keeps_every_class_in_every_fold():
    df = _groups({"A": 10, "B": 10, "C": 10})
    df["fold"] = df["group"].str.split("-").str[1].astype(int).mod(5).astype(str)  # two groups per class per fold
    kept, record = subsample_groups(df, 0.1, seed=0, strata=("species", "fold"))
    assert (kept.groupby(["species", "fold"])["group"].nunique() == 1).all()
    assert len(kept.groupby(["species", "fold"])) == 15
    assert record["groups_per_class"] == {"A": 5, "B": 5, "C": 5}


def test_every_class_keeps_at_least_one_group():
    kept, record = subsample_groups(_groups({"A": 20, "B": 1, "C": 2}), 0.1, seed=0)
    assert record["groups_per_class"] == {"A": 2, "B": 1, "C": 1}


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="sw", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=32, epochs=1, batch_size=8, bootstrap=0,
                device="cpu", split="kfold", n_folds=2, fit_final=False)
    return Config(**{**base, **kw})


def test_every_fold_and_seed_records_wall_time_device_and_peak_memory(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path))
    folds = [json.loads(p.read_text())["resources"] for p in sorted(run_dir.glob("folds/*/metrics.json"))]
    pooled = json.loads((run_dir / "metrics.json").read_text())["resources"]
    for r in [*folds, pooled]:
        assert r["device"] == "cpu" and r["wall_seconds"] > 0
        assert r["peak_memory_mb"] == 0 and r["gpu_minutes"] == 0
    assert pooled["wall_seconds"] == pytest.approx(sum(r["wall_seconds"] for r in folds))


def _sweep(tmp_path, manifest, cells):
    (tmp_path / "base.toml").write_text(
        f'run_name = "syn"\nmanifest = "{manifest.as_posix()}"\noutput_dir = "{(tmp_path / "runs").as_posix()}"\n'
        'modality = "microscopic"\nsource = "cmu"\narch = "resnet18"\nweights = "none"\nimage_size = 32\n'
        'epochs = 1\nbatch_size = 8\nbootstrap = 0\ndevice = "cpu"\nsplit = "kfold"\nn_folds = 2\nfit_final = false\n',
        encoding="utf-8")
    body = 'base = "base.toml"\nname = "p7"\n' + "".join(f"\n[[cells]]\n{cell}\n" for cell in cells)
    path = tmp_path / "sweep.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_sweep_runs_each_cell_under_a_name_that_encodes_its_overrides_and_survives_a_failing_cell(
        synthetic_manifest, tmp_path):
    path = _sweep(tmp_path, synthetic_manifest, ["train_fraction = 0.5", "lr = 0.01\nseed = 3",
                                                 'classes = ["nope"]', 'split = "holdout"'])
    with pytest.raises(SystemExit) as exit_info:
        main(["sweep", str(path)])
    assert exit_info.value.code == 1
    sweep_dir = tmp_path / "runs" / "p7"
    ran = sorted(p.name for p in sweep_dir.iterdir() if (p / "metrics.json").is_file())
    assert ran == ["syn__lr-0.01_seed-3", "syn__split-holdout", "syn__train_fraction-0.5"]
    summary = json.loads((sweep_dir / "sweep_summary.json").read_text())
    status = {c["run_name"]: c["status"] for c in summary["cells"]}
    assert status == {"syn__train_fraction-0.5": "ok", "syn__lr-0.01_seed-3": "ok", "syn__classes-nope": "failed",
                      "syn__split-holdout": "ok"}
    failed = next(c for c in summary["cells"] if c["status"] == "failed")
    assert "not in config classes" in failed["error"] and failed["overrides"] == {"classes": ["nope"]}
    assert summary["failed"] == ["syn__classes-nope"]
    assert json.loads((sweep_dir / "syn__train_fraction-0.5" / "config.json").read_text())["train_fraction"] == 0.5


def test_a_sweep_refuses_cells_that_collide_or_set_their_own_run_name(synthetic_manifest, tmp_path):
    with pytest.raises(ValueError, match="same run"):
        run_sweep(_sweep(tmp_path, synthetic_manifest, ['augmentation = "trivial wide"', 'augmentation = "trivial-wide"']))
    with pytest.raises(ValueError, match="run_name"):
        run_sweep(_sweep(tmp_path, synthetic_manifest, ['run_name = "mine"']))
    assert not (tmp_path / "runs").exists()


def test_learning_curve_on_a_directory_without_runs_says_so(tmp_path):
    with pytest.raises(ValueError, match="no metrics.json"):
        main(["learning-curve", str(tmp_path), "--out", str(tmp_path / "lc.png")])


def test_learning_curve_writes_a_plot_and_per_seed_x_and_y(synthetic_manifest, tmp_path):
    path = _sweep(tmp_path, synthetic_manifest, ["train_fraction = 0.5\nseeds = [1, 2]",
                                                 "train_fraction = 1.0\nseeds = [1, 2]"])
    main(["sweep", str(path)])
    sweep_dir = tmp_path / "runs" / "p7"
    main(["learning-curve", str(sweep_dir), "--x", "images", "--out", str(tmp_path / "lc.png")])
    assert (tmp_path / "lc.png").stat().st_size > 0
    curve = pd.read_csv(tmp_path / "lc.csv")
    assert len(curve) == 4 and set(curve["seed"]) == {1, 2}
    for _, row in curve.iterrows():
        m = json.loads((sweep_dir / row["cell"] / f"seed{row['seed']}" / "metrics.json").read_text())
        per_class = m["subsample"]["images_per_class"]
        assert row["x"] == pytest.approx(sum(per_class.values()) / len(per_class))
        assert row["y"] == pytest.approx(m["isolate_level"]["macro"]["f1"])
    half, full = (curve[curve["cell"].str.endswith(f"train_fraction-{f}")]["x"].iloc[0] for f in ("0.5", "1.0"))
    assert half < full


def test_a_run_with_a_training_fraction_lists_the_removed_groups_and_never_trains_or_validates_on_them(
        synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, train_fraction=0.5))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    sub = metrics["subsample"]
    assert sub["train_fraction"] == 0.5 and sub["removed_groups"]
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert not set(preds["group"]) & set(sub["removed_groups"])
    assert sum(sub["groups_per_class"].values()) == preds["group"].nunique()
    with pytest.raises(ValueError, match="train_fraction"):
        _cfg(synthetic_manifest, tmp_path, train_fraction=0)
