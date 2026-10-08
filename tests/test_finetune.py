import sys
from dataclasses import replace

import pandas as pd
import pytest
import torch

from mycoscan.config import Config
from mycoscan.manifest import load_manifest, select
from mycoscan.models import build_model
from mycoscan.pipeline import build_optimizer, fit_model, prepare_model, run_training, seed_everything
from mycoscan.predict import Predictor


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="ft", manifest=str(manifest), output_dir=str(tmp_path), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=64, epochs=1, batch_size=8, bootstrap=0,
                device="cpu", split="holdout", fit_final=False)
    return Config(**{**base, **kw})


def _trainable_modules(model, depth):
    return {".".join(n.split(".")[:depth]) for n, p in model.named_parameters() if p.requires_grad}


def test_partial_two_blocks_on_convnext_unfreezes_the_last_two_stages_and_the_head(synthetic_manifest, tmp_path):
    model = prepare_model(_cfg(synthetic_manifest, tmp_path, arch="convnext_tiny", finetune="partial",
                               partial_blocks=2), n_classes=4)
    assert _trainable_modules(model, 2) == {"stages.2", "stages.3", "head.norm", "head.fc"}


def test_layer_decay_lowers_the_learning_rate_by_its_factor_per_block_from_head_to_stem(synthetic_manifest, tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, finetune="full", lr=1e-3, layer_decay=0.8)
    model = prepare_model(cfg, n_classes=4)
    groups = build_optimizer(model, cfg).param_groups
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    # resnet18's blocks end at act1 (the stem), layer1, layer2, layer3 and layer4; the head comes after them.
    expected = {"fc": 1e-3, "layer4": 8e-4, "layer3": 6.4e-4, "layer2": 5.12e-4, "layer1": 4.096e-4,
                "conv1": 3.2768e-4, "bn1": 3.2768e-4}
    for name, p in model.named_parameters():
        assert lr_of[id(p)] == pytest.approx(expected[name.split(".")[0]]), name
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_a_vit_stem_gets_its_own_step_below_the_first_block_and_decay_defaults_to_the_plan_value(synthetic_manifest,
                                                                                                  tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, arch="vit_tiny_patch16_224", finetune="full", lr=1.0)
    assert cfg.layer_decay == 0.8
    model = prepare_model(cfg, n_classes=4)
    lr_of = {id(p): g["lr"] for g in build_optimizer(model, cfg).param_groups for p in g["params"]}
    lr = {name: lr_of[id(p)] for name, p in model.named_parameters()}
    assert lr["head.1.weight"] == 1.0
    assert lr["blocks.11.attn.qkv.weight"] == pytest.approx(0.8)
    assert lr["blocks.0.attn.qkv.weight"] == pytest.approx(0.8 ** 12)
    assert lr["patch_embed.proj.weight"] == lr["cls_token"] == lr["pos_embed"] == pytest.approx(0.8 ** 13)


def test_without_layer_decay_every_trainable_parameter_shares_one_learning_rate(synthetic_manifest, tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, finetune="partial", lr=1e-3, layer_decay=1.0)
    model = prepare_model(cfg, n_classes=4)
    [group] = build_optimizer(model, cfg).param_groups
    assert group["lr"] == 1e-3
    assert len(group["params"]) == sum(p.requires_grad for p in model.parameters())


def _val_probs_match_the_saved_model(run_dir, classes):
    preds = pd.read_csv(run_dir / "predictions.csv")
    predictor = Predictor(run_dir / "model.pt")
    for _, row in preds.head(4).iterrows():
        probs = predictor.predict(row["image_path"])
        assert [probs[c] for c in classes] == pytest.approx(row[[f"prob_{c}" for c in classes]].tolist(), abs=1e-5)


def test_ema_checkpoint_holds_averaged_weights_and_evaluation_uses_them(synthetic_manifest, tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, finetune="full", ema=True, ema_decay=0.9)
    run_dir = run_training(cfg)
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    classes = sorted(df["species"].unique())
    val_groups = set(pd.read_csv(run_dir / "predictions.csv")["group"])
    raw, _ = fit_model(df[~df["group"].isin(val_groups)], replace(cfg, ema=False), classes, "cpu", seed=cfg.seed)
    saved = torch.load(run_dir / "model.pt", weights_only=False)["state_dict"]
    assert not torch.equal(saved["fc.1.weight"], raw.state_dict()["fc.1.weight"])
    _val_probs_match_the_saved_model(run_dir, classes)


def test_ema_warms_up_so_a_short_run_does_not_save_the_initial_weights(synthetic_manifest, tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, finetune="full", ema=True, lr=1e-2)
    assert cfg.ema_decay == 0.999
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    classes = sorted(df["species"].unique())
    seed_everything(0)
    initial = build_model(cfg.arch, len(classes), "none", cfg.image_size).state_dict()["fc.1.weight"]
    averaged, _ = fit_model(df, cfg, classes, "cpu", seed=0)
    raw, _ = fit_model(df, replace(cfg, ema=False), classes, "cpu", seed=0)
    ema_w, raw_w = averaged.state_dict()["fc.1.weight"], raw.state_dict()["fc.1.weight"]
    assert (ema_w - raw_w).norm() < (ema_w - initial).norm()


def test_lora_trains_only_adapters_and_head_and_the_reloaded_checkpoint_predicts_the_same(synthetic_manifest, tmp_path):
    pytest.importorskip("peft")
    cfg = _cfg(synthetic_manifest, tmp_path, arch="vit_tiny_patch16_224", finetune="lora", lora_rank=4, lr=1e-2)
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    classes = sorted(df["species"].unique())
    trainable = [n for n, p in prepare_model(cfg, len(classes)).named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)

    run_dir = run_training(cfg)
    seed_everything(cfg.seed)
    initial = build_model(cfg.arch, len(classes), "none", cfg.image_size).state_dict()
    saved = torch.load(run_dir / "model.pt", weights_only=False)["state_dict"]
    assert saved.keys() == initial.keys()  # adapters merged: a plain model that loads without peft
    adapted = [k for k in saved if k.endswith(("qkv.weight", "proj.weight", "fc1.weight", "fc2.weight"))
               and k.startswith("blocks.")]
    assert adapted
    for k in adapted:
        delta = saved[k] - initial[k]
        assert 0 < torch.linalg.matrix_rank(delta) <= 4, k
    for k in saved:
        if k not in adapted and not k.startswith("head."):
            assert torch.equal(saved[k], initial[k]), k
    _val_probs_match_the_saved_model(run_dir, classes)


def test_lora_without_peft_names_the_extra(synthetic_manifest, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(ImportError, match=r"mycoscan\[lora\]"):
        run_training(_cfg(synthetic_manifest, tmp_path / "runs", arch="vit_tiny_patch16_224", finetune="lora"))
    assert not (tmp_path / "runs").exists()


def test_finetune_keys_are_validated(synthetic_manifest, tmp_path):
    with pytest.raises(ValueError, match="partial_blocks"):
        _cfg(synthetic_manifest, tmp_path, partial_blocks=0)
    with pytest.raises(ValueError, match="layer_decay"):
        _cfg(synthetic_manifest, tmp_path, layer_decay=1.5)
    with pytest.raises(ValueError, match="lora_rank"):
        _cfg(synthetic_manifest, tmp_path, lora_rank=0)
    with pytest.raises(ValueError, match="ema_decay"):
        _cfg(synthetic_manifest, tmp_path, ema_decay=1.0)
