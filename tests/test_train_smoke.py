import json

import pandas as pd
import pytest
import torch

from mycoscan.config import Config
from mycoscan.explain import explain_images, gradcam, smoothgrad
from mycoscan.manifest import load_manifest, select
from mycoscan.models import apply_finetune, build_model, save_checkpoint
from mycoscan.pipeline import fit_model, run_training
from mycoscan.predict import Predictor
from mycoscan.synthetic import PLACEHOLDER_CMU_CLASSES


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="smoke", manifest=str(manifest), output_dir=str(tmp_path), modality="microscopic",
                source="cmu", weights="none", image_size=64, epochs=1, batch_size=8, bootstrap=10, device="cpu")
    return Config(**{**base, **kw})


def test_head_only_training_leaves_backbone_and_bn_stats_untouched(synthetic_manifest, tmp_path):
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    classes = sorted(df["species"].unique())
    torch.manual_seed(0)
    before = {k: v.clone() for k, v in build_model("densenet121", len(classes), "none").state_dict().items()}
    torch.manual_seed(0)
    model, history = fit_model(df, _cfg(synthetic_manifest, tmp_path, finetune="head"), classes, "cpu", seed=0)
    after = model.state_dict()
    assert len(history) == 1
    assert torch.equal(before["features.conv0.weight"], after["features.conv0.weight"])
    assert torch.equal(before["features.norm5.running_mean"], after["features.norm5.running_mean"])
    assert not torch.equal(before["classifier.1.weight"], after["classifier.1.weight"])


def test_partial_unfreezes_only_head_and_last_block():
    model = build_model("resnet50", 4, "none")
    apply_finetune(model, "resnet50", "partial")
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}


def test_staged_transfer_reuses_backbone_and_replaces_head(tmp_path):
    stage1 = build_model("densenet121", 5, "none")
    save_checkpoint(tmp_path / "s1.pt", stage1, "densenet121", list("abcde"), 64, True, {})
    stage2 = build_model("densenet121", 10, str(tmp_path / "s1.pt"))
    key = "features.denseblock4.denselayer16.conv2.weight"
    assert torch.equal(stage1.state_dict()[key], stage2.state_dict()[key])
    assert stage2.classifier[1].out_features == 10


def test_end_to_end_train_eval_explain_predict(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="holdout", finetune="partial",
                                classes=PLACEHOLDER_CMU_CLASSES))
    preds = pd.read_csv(run_dir / "predictions.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    all_isolates = set(select(load_manifest(synthetic_manifest), "microscopic", "cmu")["isolate_id"])
    assert len(all_isolates) == 25
    assert preds["isolate_id"].nunique() == 5
    assert metrics["image_level"]["n"] == len(preds) == 10
    assert metrics["isolate_level"]["n"] == 5
    assert (run_dir / "confusion_isolate_level.png").stat().st_size > 0

    reloaded = load_manifest(run_dir / "predictions.csv")
    assert reloaded["image_path"].tolist() == preds["image_path"].tolist()

    probs = Predictor(run_dir / "model.pt").predict(preds["image_path"].iloc[0])
    assert sorted(probs) == sorted(PLACEHOLDER_CMU_CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert list(probs.values()) == sorted(probs.values(), reverse=True)

    sheet = explain_images(run_dir / "model.pt", preds["image_path"].iloc[:2].tolist(), tmp_path / "xai")
    assert len(sheet) == 2
    assert all((tmp_path / "xai" / name).stat().st_size > 0 for name in sheet["panel"])
    assert len(pd.read_csv(tmp_path / "xai" / "review_sheet.csv")) == 2


def test_gradcam_and_saliency_are_unit_range_maps_even_with_frozen_backbone():
    torch.manual_seed(0)
    model = build_model("densenet121", 3, "none").eval()
    apply_finetune(model, "densenet121", "head")
    x = torch.randn(1, 3, 64, 64)
    cam = gradcam(model, model.features.denseblock4, x, target=1)
    sal = smoothgrad(model, x, target=1, n=3)
    assert cam.shape == sal.shape == (64, 64)
    assert (float(sal.min()), float(sal.max())) == (0.0, 1.0)
    assert (float(cam.min()), float(cam.max())) == (0.0, 1.0)
