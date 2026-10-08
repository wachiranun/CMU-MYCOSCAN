import json

import pandas as pd
import pytest
import torch

from mycoscan.cli import main
from mycoscan.config import Config
from mycoscan.manifest import load_manifest
from mycoscan.pipeline import run_training
from mycoscan.stage1 import export_stage1


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="of_stage1", manifest=str(manifest), output_dir=str(tmp_path / "runs"), source="openfungi",
                modality="microscopic", arch="resnet18", weights="none", finetune="full", image_size=32, epochs=1,
                batch_size=8, device="cpu")
    return Config(**{**base, **kw})


@pytest.fixture(scope="module")
def exported(synthetic_manifest, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("stage1")
    splits = tmp_path / "splits_v1.csv"
    main(["partition", "--manifest", str(synthetic_manifest), "--out", str(splits)])
    cfg = _cfg(synthetic_manifest, tmp_path, splits_file=str(splits))
    return export_stage1(cfg), splits, tmp_path


def test_export_writes_a_head_stripped_checkpoint_named_by_modality_and_backbone(exported):
    path, splits, _ = exported
    assert path.name == "of_micro_resnet18.pt"
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["head_stripped"] is True and ckpt["arch"] == "resnet18" and ckpt["modality"] == "microscopic"
    assert not any(k.startswith("fc.") for k in ckpt["state_dict"])
    assert "layer4.1.conv2.weight" in ckpt["state_dict"]
    prov = ckpt["provenance"]
    assert prov["splits_sha256"] and prov["manifest_sha256"] and "commit" in prov
    assert prov["stage1_name"] == "of_micro_resnet18" and prov["config"]["arch"] == "resnet18"
    assert json.loads(path.with_suffix(".json").read_text())["stage1_name"] == "of_micro_resnet18"


def test_export_trains_on_every_pool_a_group_of_its_modality_and_no_pool_b_group(exported, synthetic_manifest):
    path, splits, _ = exported
    pools = pd.read_csv(splits, dtype=str, keep_default_na=False).set_index("group")["pool"]
    micro = load_manifest(synthetic_manifest).query("source == 'openfungi' and modality == 'microscopic'")
    pool_a = sorted(g for g in micro["group"].unique() if pools[g] == "A")
    trained = torch.load(path, weights_only=False)["provenance"]["trained_groups"]
    assert trained["n"] == len(pool_a) and trained["groups"] == pool_a


def test_export_refuses_to_run_without_a_splits_file_that_holds_pool_b_out(synthetic_manifest, tmp_path):
    with pytest.raises(ValueError, match="Pool B"):
        export_stage1(_cfg(synthetic_manifest, tmp_path))


def test_stage2_run_reports_only_head_parameters_new_and_carries_the_stage1_provenance(exported, synthetic_manifest):
    path, _, tmp_path = exported
    run_dir = run_training(Config(run_name="stage2", manifest=str(synthetic_manifest), output_dir=str(tmp_path / "s2"),
                                  source="cmu", modality="microscopic", arch="resnet18", weights=str(path),
                                  image_size=32, epochs=1, batch_size=8, bootstrap=0, device="cpu", split="holdout"))
    stage1 = json.loads((run_dir / "metrics.json").read_text())["provenance"]["stage1"]
    assert stage1["newly_initialised"] and all(k.startswith("fc.") for k in stage1["newly_initialised"])
    assert stage1["checkpoint"] == str(path)
    assert stage1["provenance"] == json.loads(path.with_suffix(".json").read_text())


def test_a_stage1_checkpoint_of_another_arch_is_rejected(exported, synthetic_manifest):
    path, _, tmp_path = exported
    with pytest.raises(ValueError, match="resnet18"):
        run_training(Config(run_name="mismatch", manifest=str(synthetic_manifest), output_dir=str(tmp_path / "mm"),
                            source="cmu", modality="microscopic", arch="resnet34", weights=str(path), image_size=32,
                            epochs=1, device="cpu", split="holdout"))
