import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mycoscan.data import balanced_sample_weights, make_loader
from mycoscan.manifest import load_manifest, select
from mycoscan.transforms import AutoContrast, build_transform, to_rgb


def _image():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (80, 100, 3), dtype=np.uint8))


def test_eval_transform_is_deterministic_and_sized():
    tf = build_transform(64, autocontrast=True, train=False)
    a, b = tf(_image()), tf(_image())
    assert tuple(a.shape) == (3, 64, 64)
    assert torch.equal(a, b)


def test_train_transform_augments():
    torch.manual_seed(0)
    tf = build_transform(64, autocontrast=True, train=True)
    a, b = tf(_image()), tf(_image())
    assert tuple(a.shape) == (3, 64, 64)
    assert not torch.equal(a, b)


def test_autocontrast_stretches_dim_image():
    ramp = np.tile(np.linspace(100, 120, 50).astype(np.uint8), (50, 1))
    flat = Image.fromarray(np.stack([ramp] * 3, axis=2))
    out = np.asarray(AutoContrast()(flat))
    assert (int(out.min()), int(out.max())) == (0, 255)


def test_autocontrast_keeps_pigment_hue():
    a = np.zeros((40, 40, 3), np.uint8)
    a[:] = (225, 215, 185)
    a[10:30, 10:30] = (174, 70, 60)
    r, g, b = np.asarray(AutoContrast()(Image.fromarray(a)))[20, 20].astype(int)
    assert r - g > 100 and r - b > 100


def test_to_rgb_rescales_16_bit_frames():
    frame = Image.fromarray(np.linspace(0, 40000, 64 * 64).reshape(64, 64).astype(np.uint16))
    out = np.asarray(to_rgb(frame))
    assert out.shape == (64, 64, 3)
    assert (int(out.min()), int(out.max())) == (0, 255)
    assert int(out[32, 0, 0]) == 127


def test_to_rgb_puts_transparent_pixels_on_white():
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[0, 0] = (200, 0, 0, 255)
    out = np.asarray(to_rgb(Image.fromarray(rgba, "RGBA")))
    assert out[0, 0].tolist() == [200, 0, 0]
    assert out[1, 1].tolist() == [255, 255, 255]


def test_validation_loader_returns_real_images_unaugmented(synthetic_manifest):
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu").head(6)
    classes = {s: i for i, s in enumerate(sorted(df["species"].unique()))}
    val = make_loader(df, classes, 64, True, train=False, batch_size=6, num_workers=0)
    assert torch.equal(next(iter(val))[0], next(iter(val))[0])
    train = make_loader(df, classes, 64, True, train=True, batch_size=6, num_workers=0, imbalance="none")
    assert not torch.equal(next(iter(train))[0], next(iter(train))[0])


def test_sample_weights_balance_classes_and_isolates():
    species = pd.Series(["a", "a", "a", "b"])
    isolates = pd.Series(["a1", "a1", "a2", "b1"])
    assert balanced_sample_weights(species, isolates).tolist() == [0.25, 0.25, 0.5, 1.0]


def test_manifest_rejects_isolate_with_two_species(tmp_path):
    (tmp_path / "x.png").write_bytes(b"")
    pd.DataFrame({"image_path": ["x.png", "x.png"], "species": ["A", "B"], "isolate_id": ["I1", "I1"],
                  "modality": ["colony", "colony"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ValueError, match="I1"):
        load_manifest(tmp_path / "m.csv")


def test_manifest_requires_isolate_for_cmu_and_group_id_for_openfungi(tmp_path):
    (tmp_path / "x.png").write_bytes(b"")
    rows = {"image_path": ["x.png", "x.png"], "species": ["A", "A"], "isolate_id": [" I1 ", ""],
            "group_id": ["", " G1 "], "modality": ["colony", "colony"], "source": ["cmu", "openfungi"]}
    pd.DataFrame(rows).to_csv(tmp_path / "ok.csv", index=False)
    assert load_manifest(tmp_path / "ok.csv")["group"].tolist() == ["I1", "G1"]
    rows["source"] = ["cmu", "cmu"]
    pd.DataFrame(rows).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="no isolate_id"):
        load_manifest(tmp_path / "bad.csv")


def test_manifest_rejects_unknown_modality(tmp_path):
    (tmp_path / "x.png").write_bytes(b"")
    pd.DataFrame({"image_path": ["x.png"], "species": ["A"], "isolate_id": ["I1"],
                  "modality": ["histology"]}).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(ValueError, match="histology"):
        load_manifest(tmp_path / "m.csv")


def test_manifest_paths_are_absolute_when_loaded_by_relative_path(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.png").write_bytes(b"")
    pd.DataFrame({"image_path": ["x.png"], "species": ["A"], "isolate_id": ["I1"],
                  "modality": ["colony"]}).to_csv(tmp_path / "sub" / "m.csv", index=False)
    monkeypatch.chdir(tmp_path)
    assert load_manifest("sub/m.csv")["image_path"].tolist() == [str((tmp_path / "sub" / "x.png").resolve())]
