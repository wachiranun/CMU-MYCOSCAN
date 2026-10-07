import numpy as np
import pandas as pd
import pytest

from mycoscan.metrics import auc_ovr, classification_report, evaluate_predictions


def _confident_probs(pred, n):
    p = np.full((len(pred), n), 0.1)
    p[np.arange(len(pred)), pred] = 0.8
    return p


def test_confusion_derived_metrics_match_hand_computation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    r = classification_report(y_true, _confident_probs(y_pred, 3), ["a", "b", "c"])
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert r["accuracy"] == pytest.approx(4 / 6)
    a, b, c = (r["per_class"][k] for k in "abc")
    assert (a["sensitivity"], a["specificity"], a["ppv"], a["npv"], a["f1"]) == pytest.approx((0.5, 0.75, 0.5, 0.75, 0.5))
    assert (b["sensitivity"], b["specificity"], b["ppv"], b["npv"], b["f1"]) == pytest.approx((1.0, 0.75, 2 / 3, 1.0, 0.8))
    assert (c["sensitivity"], c["specificity"], c["ppv"], c["npv"], c["f1"]) == pytest.approx((0.5, 1.0, 1.0, 0.8, 2 / 3))
    assert r["macro"]["sensitivity"] == pytest.approx(2 / 3)


def test_never_predicted_class_has_ppv_zero_in_macro():
    r = classification_report(np.array([0, 0, 1, 1]), _confident_probs(np.array([0, 0, 0, 0]), 2), ["a", "b"])
    assert r["per_class"]["b"]["ppv"] == 0.0
    assert r["macro"]["ppv"] == pytest.approx(0.25)


def test_image_bootstrap_resamples_whole_isolates_within_class():
    preds = pd.DataFrame({
        "isolate_id": ["a1"] * 10 + ["a2"] * 10 + ["b1"] * 10 + ["b2"] * 10,
        "species": ["a"] * 20 + ["b"] * 20,
        "prob_a": [0.9] * 10 + [0.2] * 10 + [0.1] * 20,
        "prob_b": [0.1] * 10 + [0.8] * 10 + [0.9] * 20,
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=400, seed=1)["image_level"]["ci95_isolate_bootstrap"]
    assert ci["method"] == "isolates within class"
    assert ci["accuracy"] == pytest.approx([0.5, 1.0])


def test_auc_matches_known_value():
    assert auc_ovr(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), positive=1) == pytest.approx(0.75)


def test_auc_is_nan_without_negatives():
    assert np.isnan(auc_ovr(np.array([1, 1]), np.array([0.2, 0.9]), positive=1))


def test_isolate_vote_overrules_minority_of_wrong_images():
    preds = pd.DataFrame({
        "isolate_id": ["i1", "i1", "i1", "i2"],
        "species": ["a", "a", "a", "b"],
        "prob_a": [0.9, 0.9, 0.4, 0.2],
        "prob_b": [0.1, 0.1, 0.6, 0.8],
    })
    m = evaluate_predictions(preds, ["a", "b"], n_boot=0)
    assert m["image_level"]["accuracy"] == pytest.approx(0.75)
    assert m["isolate_level"]["accuracy"] == pytest.approx(1.0)
    assert m["isolate_level"]["n"] == 2


def test_bootstrap_ci_of_perfect_classifier_is_degenerate():
    preds = pd.DataFrame({
        "isolate_id": ["i1", "i2", "i3", "i4"],
        "species": ["a", "a", "b", "b"],
        "prob_a": [0.9, 0.8, 0.1, 0.3],
        "prob_b": [0.1, 0.2, 0.9, 0.7],
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=50)["isolate_level"]["ci95_isolate_bootstrap"]
    assert ci["accuracy"] == [1.0, 1.0]


def test_bootstrap_with_one_isolate_per_class_is_not_frozen():
    preds = pd.DataFrame({
        "isolate_id": ["a1", "a1", "b1", "b1"],
        "species": ["a", "a", "b", "b"],
        "prob_a": [0.9, 0.9, 0.9, 0.9],
        "prob_b": [0.1, 0.1, 0.1, 0.1],
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=200, seed=0)["image_level"]["ci95_isolate_bootstrap"]
    assert ci["method"] == "isolates"
    assert ci["accuracy"] == [0.0, 1.0]
