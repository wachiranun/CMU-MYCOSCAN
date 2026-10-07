"""Classification metrics from the confusion matrix, one-vs-rest AUC, isolate-level
aggregation, and isolate-clustered bootstrap confidence intervals.

Images of one isolate are correlated, so the bootstrap resamples isolates, not images,
and it resamples within each species so every replicate keeps every class. When some
class has only one isolate (the 80/20 holdout), it resamples isolates without strata.
A class that has validation support but is never predicted gets PPV 0, not "undefined".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PER_CLASS = ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy_ovr", "auc")


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
    return {
        "n": int(total),
        "accuracy": _ratio(np.trace(cm), total),
        "macro": macro,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes_without_support": [c for c in classes if c not in present],
    }


def prob_columns(classes: list[str]) -> list[str]:
    return [f"prob_{c}" for c in classes]


def aggregate_by_isolate(preds: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Mean-probability vote over all images of an isolate (the clinical unit)."""
    cols = prob_columns(classes)
    return preds.groupby("isolate_id").agg({"species": "first", **{c: "mean" for c in cols}}).reset_index()


def _bootstrap(preds: pd.DataFrame, classes: list[str], n_boot: int, seed: int) -> dict:
    cols = prob_columns(classes)
    class_idx = {c: i for i, c in enumerate(classes)}
    groups = preds.groupby("isolate_id").indices
    species_of = preds.groupby("isolate_id")["species"].first()
    strata = [species_of.index[species_of == s].tolist() for s in species_of.unique()]
    # A class with a single isolate has nothing to resample within; stratifying
    # would freeze it and report a falsely narrow interval.
    stratified = min(len(stratum) for stratum in strata) >= 2
    if not stratified:
        strata = [species_of.index.tolist()]
    y_all = preds["species"].map(class_idx).to_numpy()
    p_all = preds[cols].to_numpy()
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"accuracy": [], **{f"macro_{m}": [] for m in PER_CLASS}}
    for _ in range(n_boot):
        picked = [iso for stratum in strata for iso in rng.choice(stratum, len(stratum))]
        rows = np.concatenate([groups[iso] for iso in picked])
        rep = classification_report(y_all[rows], p_all[rows], classes)
        samples["accuracy"].append(rep["accuracy"])
        for m in PER_CLASS:
            samples[f"macro_{m}"].append(rep["macro"][m])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}
    return {"method": "isolates within class" if stratified else "isolates", "replicates": n_boot, **ci}


def evaluate_predictions(preds: pd.DataFrame, classes: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """preds: one row per image with species, isolate_id and prob_<class> columns."""
    class_idx = {c: i for i, c in enumerate(classes)}
    cols = prob_columns(classes)
    iso = aggregate_by_isolate(preds, classes)
    result = {}
    for level, table in (("image_level", preds), ("isolate_level", iso)):
        report = classification_report(table["species"].map(class_idx).to_numpy(), table[cols].to_numpy(), classes)
        if n_boot:
            report["ci95_isolate_bootstrap"] = _bootstrap(table, classes, n_boot, seed)
        result[level] = report
    return result
