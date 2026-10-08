import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from mycoscan.cli import main
from mycoscan.config import Config, config_with
from mycoscan.explain import gradcam
from mycoscan.models import adapter, build_model
from mycoscan.pipeline import run_training
from mycoscan.synthetic import make_planted_duplicates

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_small_cnn_builds_from_the_registry_without_pretrained_weights_and_trains():
    torch.manual_seed(0)
    model = build_model("small_cnn", 3, "none", image_size=128)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 3)
    nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert all(p.grad is not None for p in model.parameters())
    cam = gradcam(model, model.get_submodule(adapter(model).cam_layer), torch.randn(1, 3, 128, 128), target=1)
    assert cam.shape == (128, 128)
    with pytest.raises(ValueError, match="small_cnn"):
        Config(run_name="x", manifest="m.csv", arch="small_cnn", weights="imagenet")


def test_the_two_p0_configs_differ_only_in_split_and_name():
    grouped, leaky = (config_with(CONFIGS / f"p0_small_cnn_micro_{k}.toml", {"splits_file": ""})
                      for k in ("grouped", "leaky"))
    assert (grouped.arch, grouped.weights, grouped.image_size) == ("small_cnn", "none", 128)
    assert (grouped.split, leaky.split) == ("kfold", "image_random")
    differ = {k for k, v in vars(grouped).items() if vars(leaky)[k] != v}
    assert differ == {"run_name", "split"}


@pytest.fixture(scope="module")
def p0_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("p0")
    manifest = make_planted_duplicates(root / "data", groups_per_class=6, shots=3)
    tiny = {"manifest": str(manifest), "splits_file": "", "output_dir": str(root / "runs"), "image_size": 32,
            "epochs": 8, "batch_size": 8, "lr": 3e-3, "n_folds": 3, "augmentation": "none", "bootstrap": 0,
            "device": "cpu", "fit_final": False, "tau_rule": "none"}
    runs = {k: run_training(config_with(CONFIGS / f"p0_small_cnn_micro_{k}.toml", tiny)) for k in ("grouped", "leaky")}
    return root / "runs", runs


def test_on_planted_duplicates_the_leaky_split_scores_well_above_the_grouped_split(p0_runs):
    _, runs = p0_runs
    grouped, leaky = (json.loads((runs[k] / "metrics.json").read_text()) for k in ("grouped", "leaky"))
    # labels are unrelated to image content, so only a shot of a seen plate can be recognised
    assert grouped["image_level"]["accuracy"] < 0.65
    assert leaky["image_level"]["accuracy"] > grouped["image_level"]["accuracy"] + 0.2
    assert (leaky["leaky"], grouped["leaky"]) == (True, False)


def test_results_table_places_the_two_rows_together_and_marks_the_leaky_one(p0_runs, tmp_path):
    root, _ = p0_runs
    main(["results", str(root), "--out", str(tmp_path / "table.csv")])
    runs = pd.read_csv(tmp_path / "table.csv", keep_default_na=False)
    pooled = runs.index[runs["fold"] == "pooled"].tolist()
    assert pooled == [0, 1]
    assert runs.loc[pooled, "run_name"].tolist() == ["p0_small_cnn_micro_grouped", "p0_small_cnn_micro_leaky"]
    assert runs.loc[pooled, "marker"].tolist() == ["", "LEAKY, comparison only"]
    cells = pd.read_csv(tmp_path / "table_cells.csv", keep_default_na=False).set_index("cell")
    assert cells.loc["p0_small_cnn_micro_leaky", "marker"] == "LEAKY, comparison only"
