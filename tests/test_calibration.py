import json

import numpy as np
import pandas as pd
import pytest
import torch

from mycoscan.cli import main
from mycoscan.config import Config

from mycoscan.metrics import accuracy_coverage, choose_tau, evaluate_predictions, expected_calibration_error
from mycoscan.pipeline import evaluate_checkpoint, run_training


def test_ece_on_two_bins_matches_hand_computation():
    # bin [0, 0.5): confidences 0.2, 0.4 (mean 0.3), one right (0.5), gap 0.2
    # bin [0.5, 1]: confidences 0.6, 0.7, 0.9 (mean 0.7333), two right (0.6667), gap 0.0667
    # ECE = 2/5 * 0.2 + 3/5 * 0.0667 = 0.12
    calib = expected_calibration_error(np.array([0.2, 0.4, 0.6, 0.7, 0.9]), np.array([0, 1, 1, 0, 1], bool), n_bins=2)
    assert calib["ece"] == pytest.approx(0.12)
    low, high = calib["bins"]
    assert (low["lo"], low["hi"], low["n"]) == (0.0, 0.5, 2)
    assert (low["confidence"], low["accuracy"]) == pytest.approx((0.3, 0.5))
    assert (high["n"], high["confidence"], high["accuracy"]) == pytest.approx((3, 0.7333333, 2 / 3))


def test_empty_bins_are_listed_with_no_confidence_or_accuracy():
    calib = expected_calibration_error(np.array([0.95, 0.99]), np.array([1, 1], bool), n_bins=4)
    assert [b["n"] for b in calib["bins"]] == [0, 0, 0, 2]
    assert np.isnan(calib["bins"][0]["accuracy"])
    assert calib["ece"] == pytest.approx(0.03)


def test_accuracy_and_coverage_at_a_known_tau_match_hand_computation():
    conf = np.array([0.3, 0.5, 0.6, 0.8, 0.9, 0.95])
    correct = np.array([0, 1, 0, 1, 1, 1], bool)
    at_06, at_1 = accuracy_coverage(conf, correct, [0.6, 1.0])
    # tau 0.6 accepts 0.6, 0.8, 0.9, 0.95: four of six, three of them right
    assert (at_06["tau"], at_06["n_accepted"]) == (0.6, 4)
    assert (at_06["coverage"], at_06["accuracy"]) == pytest.approx((4 / 6, 3 / 4))
    assert at_1["n_accepted"] == 0 and at_1["coverage"] == 0.0 and np.isnan(at_1["accuracy"])


def _dev_table(split=""):
    """Six single-image isolates of classes a/b; confidence of the top class and whether it is right:
    0.55 wrong, 0.6 right, 0.7 wrong, 0.8 right, 0.9 right, 0.95 right."""
    top = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
    species = ["b", "a", "b", "a", "a", "a"]
    return pd.DataFrame({"group": [f"i{k}" for k in range(6)], "species": species, "prob_a": top,
                         "prob_b": [1 - p for p in top], "split": split})


def test_tau_is_the_lowest_threshold_reaching_the_target_accuracy():
    # accepted from 0.55: 4/6; from 0.6: 4/5; from 0.7: 3/4; from 0.8: 3/3
    chosen = choose_tau(_dev_table(), ["a", "b"], rule="min_accuracy", target=0.8)
    assert chosen["tau"] == pytest.approx(0.6)
    assert (chosen["coverage"], chosen["accuracy"]) == pytest.approx((5 / 6, 4 / 5))
    assert choose_tau(_dev_table(), ["a", "b"], rule="min_accuracy", target=0.9)["tau"] == pytest.approx(0.8)
    assert choose_tau(_dev_table(), ["a", "b"], rule="min_coverage", target=0.6)["tau"] == pytest.approx(0.7)


def test_an_unreachable_target_gives_no_tau_and_says_why():
    table = _dev_table().assign(species=["b", "b", "b", "b", "b", "b"])
    chosen = choose_tau(table, ["a", "b"], rule="min_accuracy", target=0.9)
    assert chosen["tau"] is None
    assert "0.9" in chosen["reason"]


def test_tuning_tau_on_test_rows_fails_loudly():
    table = _dev_table().assign(split=["dev", "dev", "test", "dev", "dev", "dev"])
    with pytest.raises(ValueError, match="i2"):
        choose_tau(table, ["a", "b"], rule="min_accuracy", target=0.8)
    with pytest.raises(ValueError, match="Pool B"):
        choose_tau(_dev_table().assign(pool="B"), ["a", "b"], rule="min_accuracy", target=0.8)


def test_metric_block_reports_ece_reliability_bins_curve_and_metrics_at_a_given_tau():
    m = evaluate_predictions(_dev_table(), ["a", "b"], n_boot=0, tau=0.8, calibration_bins=2)
    iso = m["isolate_level"]
    # every confidence is in [0.5, 1]: mean 0.75, accuracy 4/6
    assert iso["calibration"]["ece"] == pytest.approx(abs(0.75 - 4 / 6))
    assert [b["n"] for b in iso["calibration"]["bins"]] == [0, 6]
    at = iso["reject_option"]["at_tau"]
    assert (at["tau"], at["n_accepted"], at["coverage"], at["accuracy"]) == pytest.approx((0.8, 3, 0.5, 1.0))
    curve = {p["tau"]: p for p in iso["reject_option"]["curve"]}
    assert curve[0.0]["coverage"] == 1.0 and curve[0.6]["n_accepted"] == 5
    assert "at_tau" not in evaluate_predictions(_dev_table(), ["a", "b"], n_boot=0)["isolate_level"]["reject_option"]
    # tau is tuned on isolates; the image level keeps its curve but is not scored at an isolate threshold
    assert "at_tau" not in m["image_level"]["reject_option"] and m["image_level"]["reject_option"]["curve"]


@pytest.fixture(scope="module")
def tau_run(synthetic_manifest, tmp_path_factory):
    cfg = Config(run_name="tau", manifest=str(synthetic_manifest), output_dir=str(tmp_path_factory.mktemp("tau")),
                 modality="microscopic", source="cmu", arch="resnet18", weights="none", image_size=32, epochs=1,
                 batch_size=8, bootstrap=0, device="cpu", split="kfold", n_folds=2, tau_rule="min_coverage",
                 tau_target=0.5)
    return run_training(cfg)


def test_tau_is_chosen_on_out_of_fold_isolates_and_stored_in_the_final_checkpoint(tau_run):
    metrics = json.loads((tau_run / "metrics.json").read_text())
    assert metrics["tau"]["source"] == "development"
    chosen = metrics["tau"]["selection"]
    assert (chosen["rule"], chosen["target"]) == ("min_coverage", 0.5)
    assert metrics["tau"]["value"] == chosen["tau"]
    assert chosen["coverage"] >= 0.5
    assert chosen["n_isolates"] == metrics["isolate_level"]["n"]
    assert metrics["isolate_level"]["reject_option"]["at_tau"]["tau"] == chosen["tau"]
    ckpt = torch.load(tau_run / "model.pt", weights_only=False)
    assert ckpt["tau"] == chosen["tau"]
    assert ckpt["tau_selection"]["rule"] == "min_coverage"
    assert (tau_run / "reliability_isolate_level.png").stat().st_size > 0


def test_evaluation_applies_the_stored_tau_once_without_retuning(tau_run, synthetic_manifest, tmp_path):
    stored = torch.load(tau_run / "model.pt", weights_only=False)["tau"]
    m = evaluate_checkpoint(tau_run / "model.pt", synthetic_manifest, tmp_path / "eval", source="cmu", n_boot=0,
                            device="cpu")
    assert m["tau"]["value"] == stored and m["tau"]["source"] == "checkpoint"
    at = m["isolate_level"]["reject_option"]["at_tau"]
    assert at["tau"] == stored and 0 <= at["coverage"] <= 1


def test_results_table_has_tau_coverage_and_accuracy_at_tau_columns(tau_run, tmp_path):
    main(["results", str(tau_run), "--out", str(tmp_path / "table.csv")])
    pooled = pd.read_csv(tmp_path / "table.csv").query("fold == 'pooled'").iloc[0]
    metrics = json.loads((tau_run / "metrics.json").read_text())
    at = metrics["isolate_level"]["reject_option"]["at_tau"]
    assert pooled["tau"] == pytest.approx(metrics["tau"]["value"])
    assert (pooled["isolate_coverage_at_tau"], pooled["isolate_accuracy_at_tau"]) == pytest.approx(
        (at["coverage"], at["accuracy"]))
