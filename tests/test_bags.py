import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mycoscan.bags import BagDataset, assert_no_bag_leakage, collate_bags, instances, pool_probabilities
from mycoscan.config import Config
from mycoscan.metrics import evaluate_predictions
from mycoscan.pipeline import evaluate_checkpoint, run_training
from mycoscan.splits import Fold
from mycoscan.transforms import TileSpec, build_transform


def _bag(n, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 4, 4, generator=gen), n % 3, {"bag": f"b{n}"}


def test_bags_of_ten_and_thirty_instances_batch_together_with_a_mask():
    x, mask, labels, meta = collate_bags([_bag(10, 0), _bag(30, 1)])
    assert x.shape == (2, 30, 3, 4, 4)
    assert mask.sum(dim=1).tolist() == [10, 30]
    assert not mask[0, 10:].any()
    assert torch.equal(x[0, :10], _bag(10, 0)[0])
    assert labels.tolist() == [1, 0] and [m["bag"] for m in meta] == ["b10", "b30"]


@pytest.mark.parametrize("how", ["mean", "max"])
def test_masked_instances_contribute_nothing_to_the_pooled_output(how):
    gen = torch.Generator().manual_seed(0)
    probs = torch.rand(2, 30, 4, generator=gen).softmax(dim=-1)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, :10], mask[1] = True, True
    garbage = probs.clone()
    garbage[0, 10:] = 1e6  # padding filled with values that would dominate a mean or a max
    pooled, weights = pool_probabilities(garbage, mask, how)
    alone, _ = pool_probabilities(probs[:1, :10], torch.ones(1, 10, dtype=torch.bool), how)
    assert torch.allclose(pooled[0], alone[0])
    assert torch.allclose(pooled.sum(dim=1), torch.ones(2))
    assert (weights[0, 10:] == 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_mean_and_max_pooling_of_a_hand_built_bag():
    probs = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    mean, _ = pool_probabilities(probs, mask, "mean")
    top, _ = pool_probabilities(probs, mask, "max")
    assert mean[0].tolist() == pytest.approx([1.4 / 3, 1.6 / 3])
    # per class maxima 0.9 and 0.8, renormalised: the max call is a, the mean call b
    assert top[0].tolist() == pytest.approx([0.9 / 1.7, 0.8 / 1.7])


def _table():
    """Isolate i1 (class a): images [0.9, 0.1], [0.2, 0.8], [0.3, 0.7]; isolate i2 (class b): [0.4, 0.6]."""
    return pd.DataFrame({"group": ["i1", "i1", "i1", "i2"], "species": ["a", "a", "a", "b"],
                         "prob_a": [0.9, 0.2, 0.3, 0.4], "prob_b": [0.1, 0.8, 0.7, 0.6]})


def test_max_pooling_on_a_prediction_table_gives_the_expected_isolate_call():
    mean = evaluate_predictions(_table(), ["a", "b"], n_boot=0, pooling="mean")["isolate_level"]
    top = evaluate_predictions(_table(), ["a", "b"], n_boot=0, pooling="max")["isolate_level"]
    # mean calls i1 b (0.467 / 0.533); max calls it a (0.9 / 0.8)
    assert mean["confusion_matrix"] == [[0, 1], [0, 1]]
    assert top["confusion_matrix"] == [[1, 0], [0, 1]]
    assert top["accuracy"] == 1.0


def test_isolate_level_is_pooled_from_bag_rows_and_image_level_from_image_rows():
    images = _table().assign(level="image", bag=lambda d: d["group"])
    # bag rows deliberately disagree with the mean of their images: isolate level must read them
    bags = pd.DataFrame({"group": ["i1", "i2"], "species": ["a", "b"], "prob_a": [0.7, 0.4], "prob_b": [0.3, 0.6],
                         "level": "bag", "bag": ["i1", "i2"]})
    m = evaluate_predictions(pd.concat([images, bags], ignore_index=True), ["a", "b"], n_boot=0)
    assert m["isolate_level"]["n"] == 2 and m["isolate_level"]["accuracy"] == 1.0
    assert m["image_level"]["n"] == 4 and m["image_level"]["accuracy"] == pytest.approx(2 / 4)


def _cfg(manifest, tmp_path, **kw):
    base = dict(run_name="bag", manifest=str(manifest), output_dir=str(tmp_path / "runs"), modality="microscopic",
                source="cmu", arch="resnet18", weights="none", image_size=32, epochs=1, batch_size=8, bootstrap=0,
                device="cpu", split="kfold", n_folds=2, fit_final=False, tau_rule="none")
    return Config(**{**base, **kw})


def _json(path):
    return json.loads(path.read_text())


def test_isolate_bags_with_mean_pooling_reproduce_the_mean_of_images_isolate_metrics(synthetic_manifest, tmp_path):
    single = run_training(_cfg(synthetic_manifest, tmp_path, run_name="single"))
    bagged = run_training(_cfg(synthetic_manifest, tmp_path, run_name="bagged", bag="isolate"))
    a, b = (_json(d / "metrics.json")["isolate_level"] for d in (single, bagged))
    assert b["confusion_matrix"] == a["confusion_matrix"]
    for key in ("accuracy", "top2_accuracy", "kappa"):
        assert b[key] == pytest.approx(a[key])
    assert b["macro"] == pytest.approx(a["macro"], nan_ok=True)
    preds = pd.read_csv(bagged / "predictions.csv")
    images, bags = preds[preds["level"] == "image"], preds[preds["level"] == "bag"]
    assert len(images) == len(pd.read_csv(single / "predictions.csv"))
    assert sorted(bags["bag"]) == sorted(images["group"].unique())
    assert bags.set_index("bag")["n_instances"].to_dict() == images.groupby("bag").size().to_dict()


def test_a_tile_bag_of_one_high_resolution_image_has_the_configured_size(tmp_path):
    Image.new("RGB", (600, 400), (120, 80, 40)).save(tmp_path / "plate.png")
    df = pd.DataFrame({"image_path": [str(tmp_path / "plate.png")], "species": ["a"], "group": ["g1"],
                       "device": ["unknown"], "modality": ["colony"]})
    rows = instances(df, "tiles", TileSpec(3, 2, 100))
    assert len(rows) == 6 and rows["bag"].nunique() == 1 and sorted(rows["tile"]) == list(range(6))
    x, label, meta = BagDataset(rows, {"a": 0}, build_transform(32, False, train=False), tiles=TileSpec(3, 2, 100))[0]
    assert x.shape == (6, 3, 32, 32) and label == 0 and meta["bag"] == rows["bag"].iat[0]


def test_tile_bag_run_writes_one_bag_row_per_image(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, bag="tiles", tile_grid=(2, 2), tile_size=32))
    preds = pd.read_csv(run_dir / "predictions.csv")
    tiles, bags = preds[preds["level"] == "tile"], preds[preds["level"] == "bag"]
    assert (bags["n_instances"] == 4).all() and len(tiles) == 4 * len(bags)
    assert bags["image_path"].is_unique and set(bags["image_path"]) == set(tiles["image_path"])
    metrics = _json(run_dir / "metrics.json")
    assert metrics["image_level"]["n"] == len(bags)
    assert metrics["isolate_level"]["n"] == bags["group"].nunique()


def test_isolate_device_bags_stay_inside_one_group_and_one_side_of_every_fold():
    df = pd.DataFrame({"group": ["i1", "i1", "i1", "i2"], "device": ["phone", "scope", "phone", "phone"],
                       "image_path": list("abcd")})
    bagged = instances(df, "isolate_device", None)
    assert bagged.groupby("bag")["group"].nunique().max() == 1
    assert bagged["bag"].nunique() == 3
    assert_no_bag_leakage(bagged, Fold("ok", np.array([0, 1, 2]), np.array([3])))
    with pytest.raises(AssertionError, match="phone"):
        assert_no_bag_leakage(bagged, Fold("split_bag", np.array([0, 1]), np.array([2, 3])))


def test_evaluation_pools_with_the_bag_and_pooling_stored_in_the_checkpoint(synthetic_manifest, tmp_path):
    run_dir = run_training(_cfg(synthetic_manifest, tmp_path, split="holdout", bag="isolate_device", pooling="max"))
    m = evaluate_checkpoint(run_dir / "model.pt", synthetic_manifest, tmp_path / "eval", source="cmu", n_boot=0,
                            device="cpu")
    preds = pd.read_csv(tmp_path / "eval" / "predictions.csv")
    bags = preds[preds["level"] == "bag"]
    assert set(bags["bag"]) == set(preds.loc[preds["level"] == "image", "group"] + "|"
                                   + preds.loc[preds["level"] == "image", "device"])
    assert m["isolate_level"]["n"] == bags["group"].nunique()
    assert m["image_level"]["n"] == int((preds["level"] == "image").sum())


def test_a_subgroup_covering_every_image_means_exactly_what_the_headline_means():
    # isolate i1 on two devices; bag rows are the max pooling of each device's images, renormalised
    images = pd.DataFrame({"group": ["i1", "i1", "i1", "i2"], "species": ["a", "a", "a", "b"],
                           "device": ["phone", "phone", "scope", "phone"], "source": "cmu", "level": "image",
                           "prob_a": [0.9, 0.2, 0.3, 0.4], "prob_b": [0.1, 0.8, 0.7, 0.6]})
    images["bag"] = images["group"] + "|" + images["device"]
    bags = pd.DataFrame({"group": ["i1", "i1", "i2"], "species": ["a", "a", "b"], "source": "cmu", "level": "bag",
                         "bag": ["i1|phone", "i1|scope", "i2|phone"], "device": ["phone", "scope", "phone"],
                         "prob_a": [0.9 / 1.7, 0.3, 0.4], "prob_b": [0.8 / 1.7, 0.7, 0.6]})
    m = evaluate_predictions(pd.concat([images, bags], ignore_index=True), ["a", "b"], n_boot=0, pooling="max",
                             subgroups=["source"])
    whole = m["subgroups"]["source"]["cmu"]["isolate_level"]
    assert whole["confusion_matrix"] == m["isolate_level"]["confusion_matrix"]
    assert whole["per_class"]["a"]["auc"] == pytest.approx(m["isolate_level"]["per_class"]["a"]["auc"])
