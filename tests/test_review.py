import json

import pandas as pd
import pytest
import torch
from PIL import Image

from mycoscan.cli import main
from mycoscan.explain import (REVIEW_COLUMNS, attention_rollout, explain_images, sample_for_review, score_review)
from mycoscan.manifest import load_manifest, select
from mycoscan.models import build_model, save_checkpoint


def test_attention_rollout_on_a_tiny_vit_is_a_unit_range_map_of_input_size():
    torch.manual_seed(0)
    model = build_model("vit_tiny_patch16_224", 3, "none", image_size=64).eval()
    fused = model.blocks[0].attn.fused_attn
    heat = attention_rollout(model, torch.randn(1, 3, 64, 64))
    assert heat.shape == (64, 64)
    assert (float(heat.min()), float(heat.max())) == (0.0, 1.0)
    assert model.blocks[0].attn.fused_attn == fused  # the model is left as it was


def _checkpoint(tmp_path, arch, classes):
    torch.manual_seed(0)
    path = tmp_path / f"{arch}.pt"
    save_checkpoint(path, build_model(arch, len(classes), "none", 64), arch, classes, 64, True,
                    {"modality": "microscopic"})
    return path


@pytest.fixture(scope="module")
def micro(synthetic_manifest):
    df = select(load_manifest(synthetic_manifest), "microscopic", "cmu")
    return df.groupby("species").head(1).head(2)


def _titles(panel):
    with Image.open(panel) as img:
        return img.text["Title"]


@pytest.mark.parametrize(("arch", "method"), [("vit_tiny_patch16_224", "attention_rollout"), ("resnet18", "gradcam")])
def test_explain_picks_the_map_from_the_checkpoint_and_writes_the_same_sheet_columns(arch, method, micro, tmp_path):
    ckpt = _checkpoint(tmp_path, arch, sorted(micro["species"]))
    sheet = explain_images(ckpt, micro["image_path"].tolist(), tmp_path / "xai", true_labels=micro["species"].tolist())
    written = pd.read_csv(tmp_path / "xai" / "review_sheet.csv", keep_default_na=False)
    assert list(written.columns) == list(REVIEW_COLUMNS)
    assert set(sheet["method"]) == {method}
    assert all((tmp_path / "xai" / name).stat().st_size > 0 for name in sheet["panel"])


def test_without_reveal_neither_the_panels_nor_the_sheet_show_the_true_label(micro, tmp_path):
    labels = micro["species"].tolist()
    ckpt = _checkpoint(tmp_path, "resnet18", sorted(labels))
    blind = explain_images(ckpt, micro["image_path"].tolist(), tmp_path / "blind", true_labels=labels)
    sheet_text = (tmp_path / "blind" / "review_sheet.csv").read_text()
    for label, path, panel in zip(labels, micro["image_path"], blind["panel"]):
        original, prediction, _ = _titles(tmp_path / "blind" / panel).split(" | ")
        assert label not in original and "true" not in prediction  # the model's own call may still be shown
        assert path not in sheet_text
    assert (blind["true_species"] == "").all()
    key = pd.read_csv(tmp_path / "blind" / "review_key.csv")
    assert key["true_species"].tolist() == labels

    shown = explain_images(ckpt, micro["image_path"].tolist(), tmp_path / "shown", true_labels=labels, reveal=True)
    assert shown["true_species"].tolist() == labels
    assert labels[0] in _titles(tmp_path / "shown" / shown["panel"].iat[0])


def test_scoring_a_hand_filled_sheet_gives_percent_structure_focused_and_kappa(tmp_path):
    sheet = pd.DataFrame({c: [""] * 6 for c in REVIEW_COLUMNS})
    sheet["panel"] = [f"{k:04d}.png" for k in range(6)]
    sheet["rater1_focus"] = ["structure", "structure", "partial", "background", "structure", "partial"]
    sheet["rater2_focus"] = ["structure", "partial", "partial", "background", "structure", "structure"]
    sheet.to_csv(tmp_path / "filled.csv", index=False)
    scores = score_review([tmp_path / "filled.csv"])
    assert scores["rater1"]["n"] == 6
    assert scores["rater1"]["structure"] == pytest.approx(3 / 6)
    assert scores["rater2"]["structure"] == pytest.approx(3 / 6)
    # agreement 4/6; rater1 margins S3 P2 B1, rater2 S3 P2 B1: pe = (9 + 4 + 1) / 36; kappa = (2/3 - 7/18) / (11/18)
    assert scores["agreement"]["percent"] == pytest.approx(4 / 6)
    assert scores["agreement"]["kappa"] == pytest.approx((2 / 3 - 7 / 18) / (11 / 18))


def test_scoring_merges_one_sheet_per_rater_and_refuses_an_off_scale_value(tmp_path):
    base = pd.DataFrame({c: [""] * 2 for c in REVIEW_COLUMNS}).assign(panel=["0000.png", "0001.png"])
    base.assign(rater1_focus=["structure", "background"]).to_csv(tmp_path / "r1.csv", index=False)
    base.assign(rater2_focus=["structure", "structure"]).to_csv(tmp_path / "r2.csv", index=False)
    scores = score_review([tmp_path / "r1.csv", tmp_path / "r2.csv"])
    assert (scores["rater1"]["n"], scores["rater2"]["n"], scores["agreement"]["n"]) == (2, 2, 2)
    base.assign(rater1_focus=["structure", "maybe"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="0001.png"):
        score_review([tmp_path / "bad.csv"])


def test_sampling_per_class_is_reproducible_and_never_exceeds_what_a_class_has():
    df = pd.DataFrame({"species": ["a"] * 10 + ["b"] * 2, "image_path": [f"p{k}" for k in range(12)]})
    one, again, other = (sample_for_review(df, 3, seed) for seed in (0, 0, 1))
    assert one["image_path"].tolist() == again["image_path"].tolist()
    assert one["image_path"].tolist() != other["image_path"].tolist()
    assert one.groupby("species").size().to_dict() == {"a": 3, "b": 2}


def test_review_command_samples_from_the_named_pool_only(synthetic_manifest, tmp_path):
    splits = tmp_path / "splits.csv"
    main(["partition", "--manifest", str(synthetic_manifest), "--out", str(splits)])
    df = select(load_manifest(synthetic_manifest), "all", "openfungi")
    ckpt = _checkpoint(tmp_path, "resnet18", sorted(df["species"].unique()))
    main(["explain", "--checkpoint", str(ckpt), "--manifest", str(synthetic_manifest), "--splits-file", str(splits),
          "--pool", "B", "--per-class", "1", "--seed", "3", "--out", str(tmp_path / "review")])
    key = pd.read_csv(tmp_path / "review" / "review_key.csv")
    pools = pd.read_csv(splits, dtype=str, keep_default_na=False).set_index("group")["pool"]
    by_path = df.set_index("image_path")["group"]
    assert set(key["image_path"].map(by_path).map(pools)) == {"B"}
    assert key.groupby("true_species").size().max() == 1


def test_scoring_tolerates_a_rater_sheet_that_lacks_some_panels(tmp_path):
    base = pd.DataFrame({c: [""] * 3 for c in REVIEW_COLUMNS}).assign(panel=["0000.png", "0001.png", "0002.png"])
    base.assign(rater1_focus=["structure", "partial", "background"]).to_csv(tmp_path / "r1.csv", index=False)
    base.head(2).assign(rater2_focus=["structure", "structure"]).to_csv(tmp_path / "r2.csv", index=False)
    scores = score_review([tmp_path / "r1.csv", tmp_path / "r2.csv"])
    assert (scores["rater1"]["n"], scores["rater2"]["n"], scores["agreement"]["n"]) == (3, 2, 2)
    with pytest.raises(ValueError, match="no review sheets"):
        score_review([])
