import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mycoscan.cli import main
from mycoscan.config import Config


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="ms", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=32, epochs=1, batch_size=8, bootstrap=0,
                device="cpu", split="kfold", n_folds=2, fit_final=False)
    return Config(**{**base, **kw})


def _json(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def two_by_two(synthetic_manifest, tmp_path_factory):
    from mycoscan.pipeline import run_training

    return run_training(_cfg(synthetic_manifest, tmp_path_factory.mktemp("ms"), seeds=(1, 2)))


def test_two_folds_and_two_seeds_write_four_fold_outputs_two_pooled_tables_and_a_summary(two_by_two):
    fold_metrics = sorted(two_by_two.glob("seed*/folds/*/metrics.json"))
    assert len(fold_metrics) == 4
    assert len(list(two_by_two.glob("seed*/folds/*/model.pt"))) == 4
    assert {(m["seed"], m["fold"]) for m in map(_json, fold_metrics)} == {(1, "fold0"), (1, "fold1"), (2, "fold0"),
                                                                       (2, "fold1")}
    assert all((two_by_two / f"seed{s}" / "predictions.csv").is_file() for s in (1, 2))
    assert all(_json(two_by_two / f"seed{s}" / "metrics.json")["fold"] == "pooled" for s in (1, 2))
    f1 = [_json(p)["isolate_level"]["macro"]["f1"] for p in fold_metrics]
    summary = _json(two_by_two / "summary.json")["isolate_macro_f1"]
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(np.mean(f1))
    assert summary["sd"] == pytest.approx(np.std(f1, ddof=1))


def test_seeds_change_training_but_not_fold_membership(two_by_two):
    one, two = (pd.read_csv(two_by_two / f"seed{s}" / "predictions.csv") for s in (1, 2))
    assert one.groupby("group")["fold"].first().equals(two.groupby("group")["fold"].first())
    probs = [c for c in one.columns if c.startswith("prob_")]
    assert not np.allclose(one[probs].to_numpy(), two.set_index("image_path").loc[one["image_path"], probs].to_numpy())


def _metrics_file(root, name, cell, seed, fold, f1, accuracy, commit="abc123"):
    path = root / name / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_name": cell, "cell": cell, "seed": seed, "fold": fold,
                                "isolate_level": {"accuracy": accuracy, "macro": {"f1": f1}},
                                "provenance": {"commit": commit}}))


def test_cell_rows_aggregate_fold_rows_and_leave_pooled_rows_out(tmp_path):
    _metrics_file(tmp_path, "a/fold0", "A", 1, "fold0", 0.8, 0.9)
    _metrics_file(tmp_path, "a/fold1", "A", 1, "fold1", 0.6, 0.7)
    _metrics_file(tmp_path, "a", "A", 1, "pooled", 0.99, 0.99)
    _metrics_file(tmp_path, "b", "B", 7, "holdout", 0.9, 0.95)
    main(["results", str(tmp_path), "--out", str(tmp_path / "table.csv")])
    runs = pd.read_csv(tmp_path / "table.csv")
    assert len(runs) == 4
    cells = pd.read_csv(tmp_path / "table_cells.csv").set_index("cell")
    assert cells.loc["A", "n_runs"] == 2
    assert cells.loc["A", "isolate_macro_f1_mean"] == pytest.approx(0.7)
    assert cells.loc["A", "isolate_macro_f1_sd"] == pytest.approx(0.141421, abs=1e-6)
    assert cells.loc["A", "isolate_accuracy_mean"] == pytest.approx(0.8)
    assert cells.loc["B", "isolate_macro_f1_mean"] == pytest.approx(0.9)
    assert np.isnan(cells.loc["B", "isolate_macro_f1_sd"])


def test_results_table_lists_every_fold_and_pooled_run_with_its_provenance_commit(two_by_two, tmp_path):
    main(["results", str(two_by_two), "--out", str(tmp_path / "table.csv")])
    runs = pd.read_csv(tmp_path / "table.csv", dtype={"commit": str})
    assert {"run", "run_name", "cell", "seed", "fold", "isolate_macro_f1", "isolate_accuracy", "commit"} <= set(runs)
    folds = runs[runs["fold"] != "pooled"]
    assert sorted(zip(folds["seed"], folds["fold"])) == [(1, "fold0"), (1, "fold1"), (2, "fold0"), (2, "fold1")]
    assert (runs.loc[runs["fold"] == "pooled", "seed"].sort_values() == [1, 2]).all()
    fold0 = _json(two_by_two / "seed1" / "folds" / "fold0" / "metrics.json")
    row = folds[(folds["seed"] == 1) & (folds["fold"] == "fold0")].iloc[0]
    assert row["isolate_macro_f1"] == pytest.approx(fold0["isolate_level"]["macro"]["f1"])
    assert row["cell"] == "ms"
    commit = fold0["provenance"]["commit"]
    assert (row["commit"] == commit) or (pd.isna(row["commit"]) and commit is None)


def test_provenance_records_seeds_folds_and_pooled_predictions(two_by_two):
    prov = _json(two_by_two / "summary.json")["provenance"]
    assert prov["seeds"] == [1, 2]
    assert prov["folds"]["names"] == ["fold0", "fold1"]
    for s in (1, 2):
        table = two_by_two / f"seed{s}" / "predictions.csv"
        assert prov["pooled_predictions"][str(s)]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    fold_prov = _json(two_by_two / "seed1" / "folds" / "fold0" / "metrics.json")["provenance"]
    assert fold_prov["folds"] == prov["folds"] and fold_prov["seeds"] == [1, 2]
