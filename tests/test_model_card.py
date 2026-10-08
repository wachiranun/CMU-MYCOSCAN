import json

import pytest
import torch

from mycoscan.config import Config
from mycoscan.pipeline import run_training
from mycoscan.provenance import sha256_file


@pytest.fixture(scope="module")
def mil_run(synthetic_manifest, tmp_path_factory):
    cfg = Config(run_name="card", manifest=str(synthetic_manifest), output_dir=str(tmp_path_factory.mktemp("card")),
                 modality="microscopic", source="cmu", arch="small_cnn", weights="none", finetune="full", image_size=32,
                 epochs=1, batch_size=8, bootstrap=0, device="cpu", split="kfold", n_folds=2, bag="isolate",
                 pooling="gated_attention", attention_dim=16, tau_rule="min_coverage", tau_target=0.5)
    return run_training(cfg)


def test_the_final_checkpoint_carries_modality_pooling_tau_classes_and_provenance(mil_run, synthetic_manifest):
    ckpt = torch.load(mil_run / "model.pt", weights_only=False)
    assert ckpt["modality"] == "microscopic" and ckpt["pooling"] == "gated_attention"
    assert any(k.startswith("pool.") for k in ckpt["state_dict"])
    assert ckpt["tau"] is not None and ckpt["classes"]
    assert ckpt["provenance"]["manifest_sha256"] == sha256_file(synthetic_manifest)


def test_a_model_card_beside_the_final_model_holds_classes_modality_tau_data_hashes_and_isolate_metrics(
        mil_run, synthetic_manifest):
    card = json.loads((mil_run / "model_card.json").read_text(encoding="utf-8"))
    ckpt = torch.load(mil_run / "model.pt", weights_only=False)
    metrics = json.loads((mil_run / "metrics.json").read_text())
    assert card["classes"] == ckpt["classes"]
    assert card["modality"] == "microscopic"
    assert card["pooling"]["type"] == "gated_attention"
    assert card["tau"] == ckpt["tau"]
    assert card["intended_use"]
    assert card["checkpoint"]["sha256"] == sha256_file(mil_run / "model.pt")
    assert card["training_data"]["manifest_sha256"] == sha256_file(synthetic_manifest)
    assert card["training_data"]["fold_membership_sha256"] == metrics["provenance"]["folds"]["membership_sha256"]
    headline = card["headline_metrics"]["isolate_level"]
    assert headline["n"] == metrics["isolate_level"]["n"]
    assert headline["accuracy"] == metrics["isolate_level"]["accuracy"]
    assert headline["macro_f1"] == metrics["isolate_level"]["macro"]["f1"]
