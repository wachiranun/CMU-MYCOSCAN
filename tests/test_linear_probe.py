import json

import numpy as np
import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.manifest import load_manifest
from mycoscan.pipeline import run_training
from mycoscan.predict import Predictor


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="probe", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=32, batch_size=16, bootstrap=10,
                device="cpu", split="kfold", n_folds=2, fit_final=False, finetune="linear_probe")
    return Config(**{**base, **kw})


def _json(path):
    return json.loads(path.read_text())


def test_probe_run_writes_the_standard_outputs_with_isolate_metrics_for_both_classifiers(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path))
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert set(preds["classifier"]) == {"logreg", "knn"}
    assert (preds.groupby("classifier").size() == len(preds) // 2).all()
    metrics = _json(run_dir / "metrics.json")
    for name in ("logreg", "knn"):
        block = metrics["classifiers"][name]
        assert block["isolate_level"]["role"] == "primary"
        assert block["isolate_level"]["n"] == preds["group"].nunique()
        assert "ci95_isolate_bootstrap" in block["isolate_level"]
    assert metrics["classifier"] == "logreg"
    assert metrics["isolate_level"] == metrics["classifiers"]["logreg"]["isolate_level"]
    assert (run_dir / "confusion_isolate_level.png").is_file()
    assert set(_json(run_dir / "folds" / "fold0" / "metrics.json")["classifiers"]) == {"logreg", "knn"}


def test_a_second_run_reuses_the_feature_cache_and_a_changed_manifest_does_not(synthetic_manifest, tmp_path):
    first = _json(run_training(_cfg(synthetic_manifest, tmp_path, run_name="a")) / "metrics.json")["feature_cache"]
    second = _json(run_training(_cfg(synthetic_manifest, tmp_path, run_name="b")) / "metrics.json")["feature_cache"]
    assert first["extracted"] > 0 and first["reused"] == 0
    timing = "wall_seconds"
    assert {k: v for k, v in second.items() if k != timing} == {**{k: v for k, v in first.items() if k != timing},
                                                                "extracted": 0, "reused": first["extracted"]}

    edited = synthetic_manifest.with_name("manifest_probe_edit.csv")
    edited.write_text(synthetic_manifest.read_text() + "\n", encoding="utf-8")
    third = _json(run_training(_cfg(edited, tmp_path, run_name="c")) / "metrics.json")["feature_cache"]
    assert third["path"] != first["path"]
    assert third["extracted"] == first["extracted"]


def test_pool_b_images_are_never_embedded_or_fitted(synthetic_manifest, tmp_path):
    splits = tmp_path / "splits.csv"
    main(["partition", "--manifest", str(synthetic_manifest), "--out", str(splits)])
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, source="openfungi", modality="all", n_folds=5,
                                splits_file=str(splits)))
    pools = pd.read_csv(splits, dtype=str, keep_default_na=False).set_index("group")["pool"]
    df = load_manifest(synthetic_manifest)
    pool_b = set(df.loc[df["group"].map(pools) == "B", "image_path"])
    with np.load(_json(run_dir / "metrics.json")["feature_cache"]["path"]) as cache:
        embedded = set(cache["paths"].tolist())
    assert pool_b and not embedded & pool_b
    preds = pd.read_csv(run_dir / "predictions.csv")
    assert set(preds["group"].map(pools)) == {"A"}
    assert all(set(json.loads(p.read_text())["classifiers"]) == {"logreg", "knn"}
               for p in run_dir.glob("folds/*/metrics.json"))


def test_results_table_lists_a_probe_run_like_a_fine_tuned_run(synthetic_manifest, tmp_path):
    run_training(_cfg(synthetic_manifest, tmp_path, run_name="probe"))
    run_training(_cfg(synthetic_manifest, tmp_path, run_name="tuned", finetune="head", epochs=1))
    main(["results", str(tmp_path / "runs"), "--out", str(tmp_path / "table.csv")])
    runs = pd.read_csv(tmp_path / "table.csv")
    pooled = runs[runs["fold"] == "pooled"].set_index(["run_name", "classifier"])
    assert sorted(pooled.index) == [("probe", "knn"), ("probe", "logreg"), ("tuned", "network")]
    assert pooled["isolate_macro_f1"].notna().all() and pooled["isolate_accuracy"].notna().all()
    cells = pd.read_csv(tmp_path / "table_cells.csv")
    assert sorted(zip(cells["cell"], cells["classifier"])) == [("probe", "knn"), ("probe", "logreg"),
                                                                ("tuned", "network")]


def test_probe_cost_counts_feature_extraction_and_seeds_are_refused(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path))
    metrics = _json(run_dir / "metrics.json")
    folds = [_json(p)["resources"]["wall_seconds"] for p in run_dir.glob("folds/*/metrics.json")]
    extraction = metrics["feature_cache"]["wall_seconds"]
    assert extraction > 0
    assert metrics["resources"]["wall_seconds"] == pytest.approx(sum(folds) + extraction)
    with pytest.raises(ValueError, match="deterministic"):
        _cfg(synthetic_manifest, tmp_path, seeds=(1, 2))


def test_probe_checkpoint_is_the_backbone_with_the_logistic_regression_as_its_head(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="holdout"))
    preds = pd.read_csv(run_dir / "predictions.csv")
    logreg = preds[preds["classifier"] == "logreg"]
    predictor = Predictor(run_dir / "model.pt")
    classes = predictor.classes
    for _, row in logreg.head(4).iterrows():
        probs = predictor.predict(row["image_path"])
        assert [probs[c] for c in classes] == pytest.approx(row[[f"prob_{c}" for c in classes]].tolist(), abs=1e-4)
