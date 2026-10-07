import json

import numpy as np
import pandas as pd
import pytest

from mycoscan.metrics import (auc_ovr, classification_report, cohen_kappa, evaluate_predictions, map_reference_labels,
                              wilson_interval)


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
        "group": ["a1"] * 10 + ["a2"] * 10 + ["b1"] * 10 + ["b2"] * 10,
        "species": ["a"] * 20 + ["b"] * 20,
        "prob_a": [0.9] * 10 + [0.2] * 10 + [0.1] * 20,
        "prob_b": [0.1] * 10 + [0.8] * 10 + [0.9] * 20,
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=400, seed=1)["image_level"]["ci95_isolate_bootstrap"]
    assert ci["method"] == "groups within class"
    assert ci["accuracy"] == pytest.approx([0.5, 1.0])


def test_auc_matches_known_value():
    assert auc_ovr(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), positive=1) == pytest.approx(0.75)


def test_auc_is_nan_without_negatives():
    assert np.isnan(auc_ovr(np.array([1, 1]), np.array([0.2, 0.9]), positive=1))


def test_group_level_aggregation_keys_on_group_not_isolate_id():
    preds = pd.DataFrame({
        "isolate_id": ["img:a", "img:b", "img:c"],
        "group": ["g1", "g1", "g2"],
        "species": ["a", "a", "b"],
        "prob_a": [0.9, 0.2, 0.1],
        "prob_b": [0.1, 0.8, 0.9],
    })
    m = evaluate_predictions(preds, ["a", "b"], n_boot=0)
    assert m["isolate_level"]["n"] == 2
    assert m["isolate_level"]["accuracy"] == pytest.approx(1.0)


def test_isolate_vote_overrules_minority_of_wrong_images():
    preds = pd.DataFrame({
        "group": ["i1", "i1", "i1", "i2"],
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
        "group": ["i1", "i2", "i3", "i4"],
        "species": ["a", "a", "b", "b"],
        "prob_a": [0.9, 0.8, 0.1, 0.3],
        "prob_b": [0.1, 0.2, 0.9, 0.7],
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=50)["isolate_level"]["ci95_isolate_bootstrap"]
    assert ci["accuracy"] == [1.0, 1.0]


def test_bootstrap_with_one_isolate_per_class_is_not_frozen():
    preds = pd.DataFrame({
        "group": ["a1", "a1", "b1", "b1"],
        "species": ["a", "a", "b", "b"],
        "prob_a": [0.9, 0.9, 0.9, 0.9],
        "prob_b": [0.1, 0.1, 0.1, 0.1],
    })
    ci = evaluate_predictions(preds, ["a", "b"], n_boot=200, seed=0)["image_level"]["ci95_isolate_bootstrap"]
    assert ci["method"] == "groups"
    assert ci["accuracy"] == [0.0, 1.0]


def _hand_table():
    """Six single-image isolates; top-1 calls a b b b c a, second choice given explicitly."""
    probs = np.array([
        [0.7, 0.2, 0.1],  # a, called a
        [0.4, 0.5, 0.1],  # a, called b, a second
        [0.1, 0.8, 0.1],  # b, called b
        [0.3, 0.6, 0.1],  # b, called b
        [0.1, 0.2, 0.7],  # c, called c
        [0.6, 0.1, 0.3],  # c, called a, c second
    ])
    return pd.DataFrame({
        "group": [f"i{k}" for k in range(6)],
        "species": list("aabbcc"),
        "device": ["phone", "phone", "phone", "scope", "scope", "scope"],
        **{f"prob_{c}": probs[:, k] for k, c in enumerate("abc")},
    })


def test_wilson_interval_matches_known_endpoints():
    assert wilson_interval(8, 10) == pytest.approx([0.4901625, 0.9433178], abs=1e-6)
    assert wilson_interval(0, 5) == pytest.approx([0.0, 0.4344825], abs=1e-6)
    assert wilson_interval(5, 5) == pytest.approx([0.5655175, 1.0], abs=1e-6)


def test_kappa_matches_hand_computation():
    # po = 4/6; row sums 2,2,2 and column sums 2,3,1 give pe = 12/36; kappa = (2/3 - 1/3) / (2/3)
    assert cohen_kappa(np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])) == pytest.approx(0.5)
    assert np.isnan(cohen_kappa(np.array([[3, 0], [0, 0]])))


def test_top2_wilson_kappa_and_taxonomic_rollups_on_a_hand_table():
    m = evaluate_predictions(_hand_table(), list("abc"), n_boot=0,
                             genus_map={"a": "G1", "b": "G1", "c": "G2"}, order_map={"G1": "O1", "G2": "O1"})
    iso = m["isolate_level"]
    assert iso["accuracy"] == pytest.approx(4 / 6)
    assert iso["top2_accuracy"] == pytest.approx(1.0)
    assert iso["wilson95"]["accuracy"] == pytest.approx(wilson_interval(4, 6))
    assert iso["wilson95"]["top2_accuracy"] == pytest.approx(wilson_interval(6, 6))
    assert iso["kappa"] == pytest.approx(0.5)
    # genus: only the c-called-a isolate crosses G2 -> G1
    assert iso["genus_level"]["accuracy"] == pytest.approx(5 / 6)
    assert iso["order_level"]["accuracy"] == pytest.approx(1.0)


def test_isolate_level_is_primary_and_image_level_secondary():
    m = evaluate_predictions(_hand_table(), list("abc"), n_boot=0)
    assert list(m).index("isolate_level") < list(m).index("image_level")
    assert (m["isolate_level"]["role"], m["image_level"]["role"]) == ("primary", "secondary")


def test_genus_rollup_names_uncovered_classes_instead_of_guessing():
    m = evaluate_predictions(_hand_table(), list("abc"), n_boot=0, genus_map={"a": "G1"})
    assert "genus_level" not in m["isolate_level"]
    assert m["taxonomic_rollup_skipped"] == "no genus for classes ['b', 'c']"
    m = evaluate_predictions(_hand_table(), list("abc"), n_boot=0, genus_map={"a": "G1", "b": "G1", "c": "G2"},
                             order_map={"G1": "O1"})
    assert m["taxonomic_rollup_skipped"] == "no order for genera ['G2']"


def test_per_class_bootstrap_cis_for_supported_classes_and_a_reason_for_the_rest():
    table = pd.concat([_hand_table(), _hand_table().assign(group=lambda d: d["group"] + "x")], ignore_index=True)
    table["prob_d"] = 0.0
    ci = evaluate_predictions(table, list("abcd"), n_boot=200, seed=0)["isolate_level"]["ci95_isolate_bootstrap"]
    assert sorted(ci["per_class"]) == list("abc")
    assert set(ci["per_class"]["a"]) == {"sensitivity", "specificity", "ppv", "npv", "f1", "accuracy_ovr", "auc"}
    lo, hi = ci["per_class"]["b"]["sensitivity"]
    assert lo == hi == 1.0
    assert ci["per_class_absent"] == {"d": "no support in this table"}
    assert ci["replicates"] == 200
    assert {"top2_accuracy", "kappa"} <= set(ci)


def test_subgroup_blocks_reuse_the_same_report_per_value():
    table = _hand_table()
    m = evaluate_predictions(table, list("abc"), n_boot=0, subgroups=["device"])
    assert sorted(m["subgroups"]["device"]) == ["phone", "scope"]
    phone = m["subgroups"]["device"]["phone"]
    alone = evaluate_predictions(table[table["device"] == "phone"], list("abc"), n_boot=0)
    assert phone["n_images"] == 3
    assert json.dumps(phone["isolate_level"]) == json.dumps(alone["isolate_level"])
    with pytest.raises(ValueError, match="phase"):
        evaluate_predictions(table, list("abc"), n_boot=0, subgroups=["phase"])


def test_never_predicted_class_keeps_ppv_zero_through_evaluate_predictions():
    table = _hand_table().assign(prob_c=0.0)
    iso = evaluate_predictions(table, list("abc"), n_boot=0)["isolate_level"]
    assert iso["per_class"]["c"]["ppv"] == 0.0
    assert iso["macro"]["ppv"] == pytest.approx(np.mean([iso["per_class"][k]["ppv"] for k in "abc"]))


def test_label_mapping_sums_model_classes_into_reference_labels():
    preds = pd.DataFrame({
        "group": ["g1", "g2", "g3", "g4"],
        "species": ["Fusarium", "Fusarium", "Flavi", "Alternaria"],
        "prob_FSSC": [0.3, 0.1, 0.0, 0.2],
        "prob_FOSC": [0.3, 0.1, 0.1, 0.2],
        "prob_A_flavus": [0.2, 0.2, 0.8, 0.2],
        "prob_A_fumigatus": [0.2, 0.6, 0.1, 0.4],
    })
    label_map = {"Fusarium": ["FSSC", "FOSC"], "Flavi": ["A_flavus"]}
    mapped, classes, info = map_reference_labels(preds, ["FSSC", "FOSC", "A_flavus", "A_fumigatus"], label_map)
    assert classes == ["Fusarium", "Flavi", "unmapped"]
    assert mapped["prob_Fusarium"].tolist() == pytest.approx([0.6, 0.2, 0.1])
    assert info["unmapped_reference_labels"] == ["Alternaria"]
    assert info["unmapped_model_classes"] == ["A_fumigatus"]
    m = evaluate_predictions(preds, ["FSSC", "FOSC", "A_flavus", "A_fumigatus"], n_boot=0, label_map=label_map)
    # g1 Fusarium right; g2 called A_fumigatus (unmapped) so wrong; g3 Flavi right
    assert m["isolate_level"]["n"] == 3
    assert m["isolate_level"]["accuracy"] == pytest.approx(2 / 3)
    assert m["scored_classes"] == ["Fusarium", "Flavi", "unmapped"]


def test_label_map_naming_a_class_the_model_lacks_is_refused():
    preds = pd.DataFrame({"group": ["g"], "species": ["X"], "prob_a": [1.0]})
    with pytest.raises(ValueError, match="'b'"):
        map_reference_labels(preds, ["a"], {"X": ["b"]})
