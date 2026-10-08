import pytest
import torch

from mycoscan.manifest import load_manifest, select
from mycoscan.mil import MILModel
from mycoscan.models import build_model, save_checkpoint
from mycoscan.predict import Predictor
from mycoscan.transforms import TileSpec, build_transform, load_image, tile_crop

CLASSES = ["a", "b", "c"]
MIL_META = {"modality": "microscopic", "bag": "isolate", "pooling": "gated_attention", "attention_heads": 1,
            "attention_dim": 16, "pooling_hierarchy": "none"}


def _mil_checkpoint(path, tau=None):
    torch.manual_seed(0)
    model = MILModel(build_model("small_cnn", 3, "none", 32), 3, "gated_attention", 1, 16)
    save_checkpoint(path, model, "small_cnn", CLASSES, 32, True, {**MIL_META, "tau": tau})
    return path


def _plain_checkpoint(path, pooling="mean"):
    torch.manual_seed(0)
    save_checkpoint(path, build_model("small_cnn", 3, "none", 32), "small_cnn", CLASSES, 32, True,
                    {"modality": "microscopic", "bag": "isolate", "pooling": pooling})
    return path


@pytest.fixture(scope="module")
def images(synthetic_manifest):
    return select(load_manifest(synthetic_manifest), "microscopic", "cmu")["image_path"].head(4).tolist()


def test_predict_bag_on_an_attention_mil_checkpoint_gives_probabilities_summing_to_one_and_a_top_2(images, tmp_path):
    result = Predictor(_mil_checkpoint(tmp_path / "mil.pt")).predict_bag(images)
    probs = result["probabilities"]
    assert list(probs) == sorted(CLASSES, key=probs.get, reverse=True)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert result["top2"] == list(probs)[:2]
    assert result["no_call"] is False  # no tau stored: every call is made


def test_predict_bag_is_a_no_call_when_the_top_probability_is_below_tau(images, tmp_path):
    result = Predictor(_mil_checkpoint(tmp_path / "mil.pt", tau=0.99)).predict_bag(images)
    assert max(result["probabilities"].values()) < 0.99
    assert result["no_call"] is True
    assert result["top2"] == list(result["probabilities"])[:2]  # the Top-2 is still given for the mycologist


def test_predict_bag_with_mean_pooling_is_the_mean_of_the_per_image_predictions(images, tmp_path):
    predictor = Predictor(_plain_checkpoint(tmp_path / "mean.pt"))
    singles = [predictor.predict(img) for img in images]
    pooled = predictor.predict_bag(images)["probabilities"]
    for c in CLASSES:
        assert pooled[c] == pytest.approx(sum(s[c] for s in singles) / len(singles), abs=1e-6)


def test_predict_is_unchanged_for_a_single_image_checkpoint(images, tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "single.pt"
    model = build_model("small_cnn", 3, "none", 32).eval()
    save_checkpoint(path, model, "small_cnn", CLASSES, 32, True, {"modality": "microscopic"})
    x = build_transform(32, True, train=False)(load_image(images[0], False, "microscopic")).unsqueeze(0)
    with torch.no_grad():
        expected = dict(zip(CLASSES, model(x).softmax(dim=1)[0].tolist()))
    probs = Predictor(path).predict(images[0])
    assert list(probs) == sorted(CLASSES, key=expected.get, reverse=True)
    assert probs == pytest.approx(expected)


def test_predict_bag_on_a_tile_checkpoint_pools_the_tiles_of_each_image_as_in_training(images, tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "tiles.pt"
    model = MILModel(build_model("small_cnn", 3, "none", 32), 3, "gated_attention", 1, 16).eval()
    save_checkpoint(path, model, "small_cnn", CLASSES, 32, True,
                    {**MIL_META, "bag": "tiles", "tile_grid": [2, 1], "tile_size": 32})
    tf = build_transform(32, True, train=False)
    img = load_image(images[0], False, "microscopic")
    x = torch.stack([tf(tile_crop(img, TileSpec(2, 1, 32), k)) for k in range(2)]).unsqueeze(0)
    with torch.no_grad():
        expected = dict(zip(CLASSES, model(x, torch.ones(1, 2, dtype=torch.bool)).softmax(dim=1)[0].tolist()))
    assert Predictor(path).predict_bag(images[:1])["probabilities"] == pytest.approx(expected)
