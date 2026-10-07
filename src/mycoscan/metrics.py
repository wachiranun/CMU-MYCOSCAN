"""Classification metrics from the confusion matrix, one-vs-rest AUC, group-level
aggregation, and group-clustered bootstrap confidence intervals.

The unit is the prediction table's `group` column: the isolate for CMU rows (the
clinical unit, hence the `isolate_level` key) and the pseudo-group of repeated shots
for OpenFungi rows. Images of one group are correlated, so the bootstrap resamples
groups, not images, and it resamples within each species so every replicate keeps
every class. When some class has only one group, it resamples groups without strata.
A class that has validation support but is never predicted gets PPV 0, not "undefined".

Isolate level is the primary result; image level is kept but marked secondary. Wilson
intervals treat the units of a level as independent, which holds for isolates and not
for images, another reason the image-level block is secondary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PER_CLASS = ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy_ovr", "auc")
OVERALL = ("accuracy", "top2_accuracy", "kappa")
Z95 = 1.959963984540054
UNMAPPED = "unmapped"


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def auc_ovr(y_true: np.ndarray, scores: np.ndarray, positive: int) -> float:
    """Mann-Whitney AUC of `scores` for class `positive` vs the rest; nan if a side is empty."""
    pos = y_true == positive
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def wilson_interval(successes: int, n: int, z: float = Z95) -> list[float]:
    """Wilson score interval for a binomial proportion; [nan, nan] when n is 0."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def topk_hits(y_true: np.ndarray, probs: np.ndarray, k: int) -> np.ndarray:
    """True where the reference class is among the k highest probabilities."""
    top = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return (top == y_true[:, None]).any(axis=1)


def cohen_kappa(cm: np.ndarray) -> float:
    """Cohen's kappa between reference (rows) and prediction (columns)."""
    n = cm.sum()
    if not n:
        return float("nan")
    observed = np.trace(cm) / n
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n**2
    return float(_ratio(observed - expected, 1 - expected)) if expected < 1 else float("nan")


def classification_report(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    n = len(classes)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, n)
    total = cm.sum()
    per_class = {}
    for k, name in enumerate(classes):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        sens = _ratio(tp, tp + fn)
        ppv = _ratio(tp, tp + fp) if tp + fp else (0.0 if tp + fn else float("nan"))
        per_class[name] = {
            "support": int(tp + fn),
            "sensitivity": sens,
            "specificity": _ratio(tn, tn + fp),
            "ppv": ppv,
            "npv": _ratio(tn, tn + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "accuracy_ovr": _ratio(tp + tn, total),
            "auc": auc_ovr(y_true, probs[:, k], k),
        }
    present = [c for c in classes if per_class[c]["support"] > 0]
    macro = {m: _nanmean([per_class[c][m] for c in present]) for m in PER_CLASS}
    correct, top2 = int(np.trace(cm)), int(topk_hits(y_true, probs, min(2, n)).sum())
    return {
        "n": int(total),
        "accuracy": _ratio(correct, total),
        "top2_accuracy": _ratio(top2, total),
        "kappa": cohen_kappa(cm),
        "wilson95": {"accuracy": wilson_interval(correct, int(total)), "top2_accuracy": wilson_interval(top2, int(total))},
        "macro": macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes_without_support": [c for c in classes if c not in present],
    }


def rollup_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], parent_of: Mapping[str, str]) -> dict:
    """Accuracy after mapping reference and predicted classes to a parent taxon (genus, order)."""
    parent = np.array([parent_of[c] for c in classes])
    correct, n = int((parent[y_true] == parent[y_pred]).sum()), len(y_true)
    return {"accuracy": _ratio(correct, n), "wilson95": wilson_interval(correct, n)}


def prob_columns(classes: list[str]) -> list[str]:
    return [f"prob_{c}" for c in classes]


def aggregate_by_group(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Mean-probability vote over all images of a group (the isolate for CMU data)."""
    cols = prob_columns(classes)
    return preds.groupby("group").agg({"species": "first", **{c: "mean" for c in cols}}).reset_index()


def map_reference_labels(preds: pd.DataFrame, classes: list[str],
                         label_map: Mapping[str, Sequence[str]]) -> tuple[pd.DataFrame, list[str], dict]:
    """Score an external set whose labels are coarser than the model's classes.

    label_map sends each reference label to the model classes it covers, e.g.
    {"Fusarium": ["FSSC", "FOSC"]}. Model probabilities are summed into reference labels;
    model classes no label covers are summed into an `unmapped` column, so a prediction
    of one of them counts as wrong. Rows whose reference label is not in the map are dropped.
    """
    unknown = sorted({c for targets in label_map.values() for c in targets} - set(classes))
    if unknown:
        raise ValueError(f"label map names classes the model does not have: {unknown}")
    claimed = [c for targets in label_map.values() for c in targets]
    if len(claimed) != len(set(claimed)):
        raise ValueError("label map assigns one model class to more than one reference label")
    model_unmapped = [c for c in classes if c not in claimed]
    ref_classes = list(label_map) + ([UNMAPPED] if model_unmapped else [])
    kept = preds[preds["species"].isin(label_map)]
    summed = {f"prob_{ref}": kept[prob_columns(list(label_map.get(ref, model_unmapped)))].sum(axis=1)
              for ref in ref_classes}
    kept = kept.drop(columns=prob_columns(classes)).assign(**summed)
    info = {
        "label_map": {ref: list(targets) for ref, targets in label_map.items()},
        "unmapped_reference_labels": sorted(set(preds["species"]) - set(label_map)),
        "unmapped_reference_rows": int(len(preds) - len(kept)),
        "unmapped_model_classes": model_unmapped,
    }
    return kept.reset_index(drop=True), ref_classes, info


def _bootstrap(preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int) -> dict:
    cols = prob_columns(classes)
    class_idx = {c: i for i, c in enumerate(classes)}
    groups = preds.groupby("group").indices
    species_of = preds.groupby("group")["species"].first()
    strata = [species_of.index[species_of == s].tolist() for s in species_of.unique()]
    # A class with a single group has nothing to resample within; stratifying
    # would freeze it and report a falsely narrow interval.
    stratified = min(len(stratum) for stratum in strata) >= 2
    if not stratified:
        strata = [species_of.index.tolist()]
    y_all = preds["species"].map(class_idx).to_numpy()
    p_all = preds[cols].to_numpy()
    supported = [c for c in classes if c in set(species_of)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {**{k: [] for k in OVERALL}, **{f"macro_{m}": [] for m in PER_CLASS}}
    per_class: dict[str, dict[str, list[float]]] = {c: {m: [] for m in PER_CLASS} for c in supported}
    for _ in range(n_boot):
        picked = [iso for stratum in strata for iso in rng.choice(stratum, len(stratum))]
        rows = np.concatenate([groups[iso] for iso in picked])
        rep = classification_report(y_all[rows], p_all[rows], classes)
        for k in OVERALL:
            samples[k].append(rep[k])
        for m in PER_CLASS:
            samples[f"macro_{m}"].append(rep["macro"][m])
            for c in supported:
                per_class[c][m].append(rep["per_class"][c][m])

    def ci(values: list[float]) -> list[float]:
        finite = [v for v in values if not np.isnan(v)]
        return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))] if finite else [float("nan")] * 2

    return {
        "method": "groups within class" if stratified else "groups",
        "replicates": n_boot,
        **{k: ci(v) for k, v in samples.items()},
        "per_class": {c: {m: ci(v) for m, v in metrics.items()} for c, metrics in per_class.items()},
        "per_class_absent": {c: "no support in this table" for c in classes if c not in supported},
    }


def _level_reports(preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int,
                   genus_map: Mapping[str, str], order_map: Mapping[str, str]) -> dict:
    class_idx = {c: i for i, c in enumerate(classes)}
    cols = prob_columns(classes)
    result = {}
    for level, table, role in (("isolate_level", aggregate_by_group(preds, classes), "primary"),
                               ("image_level", preds, "secondary")):
        y_true, probs = table["species"].map(class_idx).to_numpy(), table[cols].to_numpy()
        report = {"role": role, **classification_report(y_true, probs, classes)}
        if genus_map:
            report["genus_level"] = rollup_accuracy(y_true, probs.argmax(axis=1), classes, genus_map)
            if order_map:
                order_of = {c: order_map[genus_map[c]] for c in classes}
                report["order_level"] = rollup_accuracy(y_true, probs.argmax(axis=1), classes, order_of)
        if n_boot:
            report["ci95_isolate_bootstrap"] = _bootstrap(table, classes, n_boot, seed)
        result[level] = report
    return result


def _check_taxonomy(classes: list[str], genus_map: Mapping[str, str], order_map: Mapping[str, str]) -> str:
    """Why the rollup cannot run, or "" when every class has a genus (and, if an order map is given, an order)."""
    missing = [c for c in classes if c not in genus_map]
    if missing:
        return f"no genus for classes {missing}"
    missing_order = sorted({genus_map[c] for c in classes} - set(order_map))
    return f"no order for genera {missing_order}" if order_map and missing_order else ""


def evaluate_predictions(preds: pd.DataFrame, classes: list[str], n_boot: int = 2000, seed: int = 0,
                         genus_map: Mapping[str, str] | None = None, order_map: Mapping[str, str] | None = None,
                         subgroups: Sequence[str] = (), label_map: Mapping[str, Sequence[str]] | None = None) -> dict:
    """preds: one row per image with species, group and prob_<class> columns.

    genus_map / order_map: species -> genus and genus -> order for taxonomic rollups.
    subgroups: columns of `preds` (e.g. device, phase); the same report is repeated per value.
    label_map: reference label -> model classes, for an external set with coarser labels.
    """
    result: dict = {}
    if label_map:
        preds, classes, result["label_mapping"] = map_reference_labels(preds, classes, label_map)
        if preds.empty:
            raise ValueError(f"no rows left after label mapping; reference labels: {result['label_mapping']['unmapped_reference_labels']}")
    result["scored_classes"] = list(classes)
    genus, order = genus_map or {}, order_map or {}
    why_not = _check_taxonomy(classes, genus, order) if genus else ""
    if why_not:
        result["taxonomic_rollup_skipped"] = why_not
        genus, order = {}, {}
    result.update(_level_reports(preds, classes, n_boot, seed, genus, order))
    missing = [c for c in subgroups if c not in preds]
    if missing:
        raise ValueError(f"subgroup columns not in the prediction table: {missing}")
    if subgroups:
        result["subgroups"] = {
            col: {str(value): {"n_images": len(rows), **_level_reports(rows, classes, n_boot, seed, genus, order)}
                  for value, rows in preds.groupby(col, dropna=False)}
            for col in subgroups
        }
    return result
