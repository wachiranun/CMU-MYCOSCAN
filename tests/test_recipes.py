import json
import math

import numpy as np
import pytest
import torch
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageDraw

from mycoscan.config import Config
from mycoscan.losses import build_loss
from mycoscan.models import build_model, save_checkpoint
from mycoscan.pipeline import run_training
from mycoscan.predict import Predictor
from mycoscan.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transform, load_image, plate_circle_crop


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="recipe", manifest=str(manifest), output_dir=str(tmp_path), modality="colony",
                source="cmu", arch="resnet18", weights="none", image_size=64, epochs=1, batch_size=8, bootstrap=10,
                device="cpu", split="holdout", fit_final=False)
    return Config(**{**base, **kw})


@pytest.mark.parametrize("keys", [
    {"loss": "ce"},
    {"loss": "weighted_ce", "imbalance": "none"},
    {"loss": "focal", "focal_gamma": 1.5, "label_smoothing": 0.1},
    {"augmentation": "none"},
    {"augmentation": "trivial_wide", "plate_crop": True},
])
def test_each_loss_and_augmentation_runs_and_is_recorded(synthetic_manifest, tmp_path, keys):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, **keys))
    resolved = json.loads((run_dir / "config.json").read_text())
    assert {k: resolved[k] for k in keys} == keys
    assert (run_dir / "metrics.json").is_file()


def test_sampler_and_weighted_loss_are_mutually_exclusive():
    with pytest.raises(ValueError, match=r"imbalance='sampler'.*loss='weighted_ce'"):
        Config(run_name="x", manifest="m.csv", imbalance="sampler", loss="weighted_ce", weights="none")


def test_old_loss_imbalance_value_points_to_weighted_ce():
    with pytest.raises(ValueError, match="weighted_ce"):
        Config(run_name="x", manifest="m.csv", imbalance="loss", weights="none")


def test_focal_loss_matches_hand_computed_value_and_reduces_to_ce_at_gamma_zero():
    logits, y = torch.tensor([[0.0, 0.0]]), torch.tensor([0])
    focal = build_loss("focal", torch.tensor([0, 1]), 2, label_smoothing=0.0, focal_gamma=2.0)
    assert focal(logits, y).item() == pytest.approx(0.25 * math.log(2))
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    y = torch.tensor([2, 0])
    ce = build_loss("ce", y, 3, label_smoothing=0.1, focal_gamma=0.0)
    flat = build_loss("focal", y, 3, label_smoothing=0.1, focal_gamma=0.0)
    assert flat(logits, y).item() == pytest.approx(ce(logits, y).item())


def test_weighted_ce_upweights_the_rare_class():
    loss = build_loss("weighted_ce", torch.tensor([0, 0, 0, 1]), 2, label_smoothing=0.0, focal_gamma=0.0)
    assert loss.weight.tolist() == pytest.approx([2 / 3, 2.0])


def _image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8))


def test_trivial_wide_differs_between_calls():
    torch.manual_seed(0)
    tf = build_transform(64, autocontrast=True, train=True, augmentation="trivial_wide")
    assert not torch.equal(tf(_image()), tf(_image()))


def test_no_augmentation_equals_the_validation_transform():
    tf = build_transform(64, autocontrast=True, train=True, augmentation="none")
    val = build_transform(64, autocontrast=True, train=False)
    assert torch.equal(tf(_image()), tf(_image()))
    assert torch.equal(tf(_image()), val(_image()))


def _plate(size=(200, 160), centre=(120, 70), radius=50, colony=15):
    img = Image.new("RGB", size, (60, 60, 70))
    draw = ImageDraw.Draw(img)
    cx, cy = centre
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(225, 215, 185))
    draw.ellipse([cx - colony, cy - colony, cx + colony, cy + colony], fill=(150, 40, 30))
    return img


def test_plate_circle_crop_centres_the_plate_and_removes_the_background():
    out = np.asarray(plate_circle_crop(_plate())).astype(int)
    h, w, _ = out.shape
    assert abs(h - 100) <= 3 and abs(w - 100) <= 3
    colony = (out[..., 0] > 120) & (out[..., 1] < 80)
    ys, xs = np.nonzero(colony)
    assert abs(ys.mean() - (h - 1) / 2) < 1.5 and abs(xs.mean() - (w - 1) / 2) < 1.5
    background = (np.abs(out - (60, 60, 70)).sum(axis=2) < 15)
    assert background.sum() < 0.01 * h * w
    assert out[0, 0].tolist() == [0, 0, 0]


def test_plate_crop_applies_only_to_colony_images_and_only_when_configured():
    plate = _plate()
    assert load_image(plate, plate_crop=True, modality="colony").size != plate.size
    assert load_image(plate, plate_crop=True, modality="microscopic").size == plate.size
    assert load_image(plate, plate_crop=False, modality="colony").size == plate.size


def test_predictor_crops_by_the_modality_of_each_image(tmp_path):
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "m.pt", build_model("resnet18", 3, "none", 64), "resnet18", list("abc"), 64, True,
                    {"modality": "all", "plate_crop": True})
    predictor = Predictor(tmp_path / "m.pt")
    plate = _plate()
    cropped = plate_circle_crop(plate)
    assert predictor.predict(plate, modality="colony") == predictor.predict(cropped, modality="microscopic")
    assert predictor.predict(plate, modality="colony") != predictor.predict(plate, modality="microscopic")


def test_plate_circle_crop_leaves_an_image_without_a_plate_alone():
    flat = Image.new("RGB", (64, 48), (200, 180, 150))
    assert np.array_equal(np.asarray(plate_circle_crop(flat)), np.asarray(flat))


def _hues(x: torch.Tensor) -> np.ndarray:
    rgb = (x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None])
    hsv = rgb_to_hsv(rgb.clamp(0, 1).permute(1, 2, 0).numpy())
    coloured = (hsv[..., 1] > 0.2) & (hsv[..., 2] > 0.2)
    return hsv[..., 0][coloured]


@pytest.mark.parametrize("pigment", [(174, 70, 60), (60, 140, 70), (70, 80, 170), (200, 170, 40)])
@pytest.mark.parametrize("augmentation", ["none", "standard", "trivial_wide"])
def test_no_recipe_changes_the_hue_of_a_pigment_patch(augmentation, pigment):
    patch = Image.new("RGB", (96, 96), pigment)
    expected = rgb_to_hsv(np.array(pigment, float) / 255)[0]
    torch.manual_seed(0)
    tf = build_transform(64, autocontrast=True, train=True, augmentation=augmentation)
    measured = 0
    for _ in range(60):
        hues = _hues(tf(patch))  # empty when a draw desaturates or darkens the patch to grey or black
        if hues.size:
            measured += 1
            distance = np.minimum(np.abs(hues - expected), 1 - np.abs(hues - expected))
            assert distance.max() < 0.03
    assert measured > 40
