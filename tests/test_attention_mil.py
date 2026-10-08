import json

import numpy as np
import pandas as pd
import pytest
import torch

from mycoscan.config import Config
from mycoscan.mil import AttentionPool, MILModel
from mycoscan.models import build_model, load_checkpoint
from mycoscan.pipeline import evaluate_checkpoint, run_training


def _mil(pooling="gated_attention", heads=1, hierarchy="none"):
    torch.manual_seed(0)
    return MILModel(build_model("small_cnn", 3, "none", 32), 3, pooling, heads, 16, hierarchy).eval()


@pytest.mark.parametrize("heads", [1, 3])
def test_padded_instances_receive_zero_attention_and_weights_sum_to_one(heads):
    pool = AttentionPool(8, 16, heads)
    h = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    h[0, 2:] = 1e4  # padding that would dominate the scores if it were scored
    pooled, weights = pool(h, mask)
    assert weights.shape == (2, 5, heads)
    assert (weights[0, 2:] == 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, heads))
    alone, _ = pool(h[:1, :2], mask[:1, :2])
    assert torch.allclose(pooled[0], alone[0], atol=1e-6)


def test_hierarchical_isolate_probabilities_are_the_attention_pooling_of_its_device_probabilities():
    model = _mil(hierarchy="device_then_isolate")
    x = torch.rand(2, 4, 3, 32, 32)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    devices = torch.tensor([[0, 1, 1, -1], [0, 0, 0, 0]])
    out = model.attend(x, mask, devices)
    assert out.device_weights is not None and out.device_logits is not None and out.device_mask is not None
    assert out.device_mask.tolist() == [[True, True], [True, False]]
    assert (out.device_weights[1, 1] == 0).all()
    pooled = (out.device_weights.unsqueeze(-1) * out.device_logits.softmax(-1)).sum(dim=1)
    assert torch.allclose(out.logits.softmax(-1), pooled, atol=1e-6)
    # instance weights sum to 1 within each device, not across the isolate
    assert torch.allclose(out.weights[0, :1, 0].sum(), torch.tensor(1.0))
    assert torch.allclose(out.weights[0, 1:3, 0].sum(), torch.tensor(1.0))


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="mil", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="small_cnn", weights="none", finetune="full", image_size=32, epochs=1,
                batch_size=8, bootstrap=0, device="cpu", split="holdout", fit_final=False, tau_rule="none",
                bag="isolate", pooling="gated_attention", attention_dim=16)
    return Config(**{**base, **kw})


def _bag_probs(preds, level="bag"):
    rows = preds[preds["level"] == level].set_index("bag")
    return rows[[c for c in rows if c.startswith("prob_")]].sort_index()


def test_a_gated_attention_run_saves_its_pooling_weights_and_reloads_to_identical_bag_predictions(
        synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path))
    state = torch.load(run_dir / "model.pt", weights_only=False)["state_dict"]
    assert any(k.startswith("pool.") for k in state)
    model, meta = load_checkpoint(run_dir / "model.pt")
    assert isinstance(model, MILModel) and meta["pooling"] == "gated_attention"
    evaluate_checkpoint(run_dir / "model.pt", synthetic_manifest, tmp_path / "eval", source="cmu", n_boot=0,
                        device="cpu")
    trained = _bag_probs(pd.read_csv(run_dir / "predictions.csv"))
    reloaded = _bag_probs(pd.read_csv(tmp_path / "eval" / "predictions.csv")).loc[trained.index]
    np.testing.assert_allclose(reloaded.to_numpy(), trained.to_numpy(), atol=1e-5)


def test_the_attention_sidecar_has_a_row_per_instance_summing_to_one_per_bag_and_metrics_hold_its_entropy(
        synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="kfold", n_folds=2))
    preds = pd.read_csv(run_dir / "predictions.csv")
    attention = pd.read_csv(run_dir / "attention.csv")
    images = preds[preds["level"] == "image"]
    assert len(attention) == len(images)
    assert sorted(attention["instance"]) == sorted(images["image_path"])
    assert np.allclose(attention.groupby(["fold", "bag"])["attention"].sum(), 1.0)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    entropy = metrics["attention"]["mean_entropy"]
    sizes = attention.groupby(["fold", "bag"]).size()
    assert 0 <= entropy <= np.log(sizes.max()) + 1e-6
    assert metrics["attention"]["n_bags"] == len(sizes)


def test_the_multi_head_variant_writes_one_weight_column_per_head(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, pooling="mh_attention", attention_heads=3))
    attention = pd.read_csv(run_dir / "attention.csv")
    heads = [c for c in attention if c.startswith("attention")]
    assert heads == ["attention_h0", "attention_h1", "attention_h2"]
    for column in heads:
        assert np.allclose(attention.groupby("bag")[column].sum(), 1.0)
    assert "mean_entropy" in json.loads((run_dir / "metrics.json").read_text())["attention"]


def test_hierarchical_pooling_writes_device_rows_and_isolate_rows_pooled_from_them(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, pooling_hierarchy="device_then_isolate"))
    preds = pd.read_csv(run_dir / "predictions.csv")
    images, devices, isolates = (preds[preds["level"] == lv] for lv in ("image", "bag", "isolate"))
    assert sorted(devices["bag"]) == sorted((images["group"] + "|" + images["device"]).unique())
    assert sorted(isolates["bag"]) == sorted(images["group"].unique())
    weights = pd.read_csv(run_dir / "attention_devices.csv").set_index("instance")["attention"]
    device_probs, isolate_probs = _bag_probs(preds), _bag_probs(preds, "isolate")
    for isolate, row in isolate_probs.iterrows():
        mine = device_probs[device_probs.index.str.startswith(f"{isolate}|")]
        pooled = (mine.mul(weights.loc[mine.index], axis=0)).sum()
        np.testing.assert_allclose(row.to_numpy(), pooled.to_numpy(), atol=1e-5)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["isolate_level"]["n"] == len(isolates)


def test_attention_pooling_and_hierarchy_are_refused_without_the_bags_they_need(synthetic_manifest, tmp_path):
    with pytest.raises(ValueError, match="bag"):
        _cfg(synthetic_manifest, tmp_path, bag="none")
    with pytest.raises(ValueError, match="pooling_hierarchy"):
        _cfg(synthetic_manifest, tmp_path, bag="isolate_device", pooling_hierarchy="device_then_isolate")
    with pytest.raises(ValueError, match="pooling_hierarchy"):
        _cfg(synthetic_manifest, tmp_path, pooling="mean", pooling_hierarchy="device_then_isolate")
