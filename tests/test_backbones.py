import json

import pytest
import torch
from torch import nn
from torchvision import models as tvm

from mycoscan.config import Config
from mycoscan.manifest import load_manifest, select
from mycoscan.models import adapter, apply_finetune, build_model, load_checkpoint, resolve_weights, train_mode
from mycoscan.pipeline import fit_model, run_training

VERIFIED = [
    "convnext_tiny", "convnext_small", "tf_efficientnetv2_s", "densenet121", "resnet50",
    "vit_small_patch14_dinov2", "vit_small_patch16_dinov3", "vit_base_patch16_224", "convnext_small.dinov3_lvd1689m",
]


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="bb", manifest=str(manifest), output_dir=str(tmp_path), modality="microscopic",
                source="cmu", weights="none", image_size=64, epochs=1, batch_size=8, bootstrap=10, device="cpu",
                split="holdout")
    return Config(**{**base, **kw})


@pytest.mark.parametrize("arch", VERIFIED)
def test_verified_backbone_builds_and_takes_a_training_step(arch):
    torch.manual_seed(0)
    model = build_model(arch, 3, "none", image_size=64)
    apply_finetune(model, "partial")
    a = adapter(model)
    assert model.get_submodule(a.cam_layer) is not None
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert 0 < len(trainable) < len(list(model.parameters()))
    train_mode(model)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 3)
    nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert all(p.grad is not None for p in model.get_submodule(a.head).parameters())


@pytest.mark.parametrize(("arch", "weights", "expected"), [
    ("densenet121", "imagenet", "densenet121.tv_in1k"),
    ("resnet50", "imagenet", "resnet50.tv2_in1k"),
    ("convnext_tiny", "imagenet", "convnext_tiny.fb_in1k"),
    ("tf_efficientnetv2_s", "imagenet", "tf_efficientnetv2_s.in1k"),
    ("vit_base_patch16_224", "imagenet", "vit_base_patch16_224.augreg_in1k"),
    ("convnext_tiny", "imagenet22k", "convnext_tiny.fb_in22k"),
    ("vit_base_patch16_224", "imagenet22k", "vit_base_patch16_224.augreg_in21k"),
    ("convnext_small", "dino", "convnext_small.dinov3_lvd1689m"),
    ("vit_small_patch14_dinov2", "dino", "vit_small_patch14_dinov2.lvd142m"),
    ("convnext_tiny.fb_in1k", "imagenet", "convnext_tiny.fb_in1k"),
])
def test_weights_resolve_to_a_timm_pretrained_tag(arch, weights, expected):
    assert resolve_weights(arch, weights) == expected


def test_weights_without_a_matching_tag_fail_at_config_time(tmp_path):
    with pytest.raises(ValueError, match="lvd142m"):
        Config(run_name="x", manifest="m.csv", arch="vit_small_patch14_dinov2", weights="imagenet")
    with pytest.raises(ValueError, match="augreg_in21k_ft_in1k"):
        Config(run_name="x", manifest="m.csv", arch="vit_tiny_patch16_224", weights="imagenet")
    with pytest.raises(ValueError, match="resnet50"):
        Config(run_name="x", manifest="m.csv", arch="resnet50", weights="dino")
    with pytest.raises(ValueError, match="not_a_model"):
        Config(run_name="x", manifest="m.csv", arch="not_a_model", weights="none")


def test_checkpoint_saved_by_torchvision_densenet_code_loads_and_predicts_identically(tmp_path):
    torch.manual_seed(0)
    old = tvm.densenet121(weights=None)
    old.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(old.classifier.in_features, 4))
    old.eval()
    torch.save({"state_dict": old.state_dict(), "arch": "densenet121", "classes": list("abcd"),
                "image_size": 64, "autocontrast": True, "modality": "microscopic"}, tmp_path / "old.pt")
    model, meta = load_checkpoint(tmp_path / "old.pt")
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(old(x), model(x))
    assert meta["classes"] == list("abcd")


def test_head_only_vit_leaves_every_non_head_parameter_and_layernorm_frozen(synthetic_manifest, tmp_path):
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    classes = sorted(df["species"].unique())
    cfg = _cfg(synthetic_manifest, tmp_path, arch="vit_tiny_patch16_224", finetune="head")
    torch.manual_seed(0)
    before = {k: v.clone() for k, v in build_model(cfg.arch, len(classes), "none", 64).state_dict().items()}
    torch.manual_seed(0)
    model, _ = fit_model(df, cfg, classes, "cpu", seed=0)
    head = adapter(model).head
    changed = {k for k, v in model.state_dict().items() if not torch.equal(before[k], v)}
    assert changed and all(k.startswith(head + ".") for k in changed)
    assert not model.norm.weight.requires_grad


def test_frozen_batchnorm_stays_in_eval_mode_during_training():
    model = build_model("resnet18", 3, "none", image_size=64)
    apply_finetune(model, "partial")
    train_mode(model)
    assert not model.bn1.training and not model.layer3[0].bn1.training
    assert model.layer4[0].bn1.training


def test_amp_and_grad_clip_run_on_cpu_and_are_recorded(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, arch="resnet18", finetune="full", amp=True,
                                grad_clip=1.0, fit_final=False))
    resolved = json.loads((run_dir / "config.json").read_text())
    assert (resolved["amp"], resolved["grad_clip"]) == (True, 1.0)
    assert (run_dir / "metrics.json").is_file()


@pytest.mark.network
def test_convnext_imagenet22k_run_writes_standard_outputs(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, arch="convnext_tiny", weights="imagenet22k",
                                fit_final=False))
    for name in ("config.json", "predictions.csv", "metrics.json", "model.pt", "confusion_isolate_level.png"):
        assert (run_dir / name).is_file()


@pytest.mark.network
def test_dino_vit_head_only_run_changes_only_the_head(synthetic_manifest, tmp_path):
    cfg = _cfg(synthetic_manifest, tmp_path, arch="vit_tiny_patch16_dinov3_qkvb", weights="dino", finetune="head",
               fit_final=False)
    run_dir = run_training(cfg)
    trained, meta = load_checkpoint(run_dir / "model.pt")
    fresh = build_model(cfg.arch, len(meta["classes"]), "dino", 64).state_dict()
    head = adapter(trained).head
    for k, v in trained.state_dict().items():
        if not k.startswith(head + "."):
            assert torch.equal(fresh[k], v), k
