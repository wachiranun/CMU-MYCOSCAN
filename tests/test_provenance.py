import hashlib
import json
import subprocess
import sys

import pytest

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.models import build_model, save_checkpoint
from mycoscan.pipeline import evaluate_checkpoint, run_training
from mycoscan.provenance import git_state
from mycoscan.synthetic import PLACEHOLDER_CMU_CLASSES


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="prov", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=64, epochs=1, batch_size=8, bootstrap=10,
                device="cpu", split="holdout", fit_final=False)
    return Config(**{**base, **kw})


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_metrics_carry_provenance_with_an_independently_checkable_manifest_hash(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path))
    prov = json.loads((run_dir / "metrics.json").read_text())["provenance"]
    assert {"commit", "dirty", "manifest_sha256", "splits_sha256", "config", "packages"} <= set(prov)
    assert prov["manifest_sha256"] == _sha256(synthetic_manifest)
    assert prov["splits_sha256"] is None
    assert prov["config"]["arch"] == "resnet18"
    assert {"python", "torch", "timm", "numpy", "pandas"} <= set(prov["packages"])
    assert prov["commit"] is None or len(prov["commit"]) == 40


def test_run_with_a_splits_file_records_its_hash(synthetic_manifest, tmp_path):
    splits = tmp_path / "splits.csv"
    main(["partition", "--manifest", str(synthetic_manifest), "--out", str(splits)])
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, source="openfungi", modality="all",
                                splits_file=str(splits)))
    prov = json.loads((run_dir / "metrics.json").read_text())["provenance"]
    assert prov["splits_sha256"] == _sha256(splits)


def test_evaluation_records_the_checkpoint_and_manifest_hashes(synthetic_manifest, tmp_path):
    ckpt = tmp_path / "model.pt"
    classes = list(PLACEHOLDER_CMU_CLASSES)
    save_checkpoint(ckpt, build_model("resnet18", len(classes), "none", 64), "resnet18", classes, 64, True,
                    {"modality": "colony"})
    metrics = evaluate_checkpoint(ckpt, synthetic_manifest, tmp_path / "eval", source="cmu", n_boot=10, device="cpu")
    assert metrics["provenance"]["checkpoint_sha256"] == _sha256(ckpt)
    assert metrics["provenance"]["manifest_sha256"] == _sha256(synthetic_manifest)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def test_git_state_reports_commit_and_dirty_working_tree(tmp_path):
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    clean = git_state(tmp_path)
    assert len(clean["commit"]) == 40 and clean["dirty"] is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_state(tmp_path) == {"commit": clean["commit"], "dirty": True}


def test_git_state_outside_a_repository_is_unknown(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def test_mlflow_tracking_without_mlflow_installed_names_the_extra(synthetic_manifest, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match=r"mycoscan\[mlflow\]"):
        run_training(_cfg(synthetic_manifest, tmp_path, tracking="mlflow"))
    assert not (tmp_path / "runs").exists()


def test_mlflow_tracking_logs_params_metrics_and_artifacts(synthetic_manifest, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # MLflow puts artifacts under ./mlruns
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, tracking="mlflow", run_name="tracked"))
    [run] = mlflow.search_runs(search_all_experiments=True, filter_string="tags.mlflow.runName = 'tracked'",
                               output_format="list")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert run.data.params["arch"] == "resnet18"
    assert run.data.params["manifest_sha256"] == _sha256(synthetic_manifest)
    assert run.data.metrics["isolate_macro_f1"] == pytest.approx(metrics["isolate_level"]["macro"]["f1"])
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"metrics.json", "predictions.csv", "config.json"} <= artifacts
