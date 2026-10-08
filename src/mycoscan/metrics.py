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
TAU_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)  # thresholds of the reported accuracy-coverage curve


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


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    """ECE over `n_bins` equal-width confidence bins, [lo, hi) and the last one closed, and the reliability
    table behind it: per bin its count, mean confidence and accuracy (nan for an empty bin)."""
    which = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        inside = which == b
        n = int(inside.sum())
        conf = float(confidence[inside].mean()) if n else float("nan")
        acc = float(correct[inside].mean()) if n else float("nan")
        if n:
            ece += n / len(confidence) * abs(acc - conf)
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n, "confidence": conf, "accuracy": acc})
    return {"ece": ece if len(confidence) else float("nan"), "n_bins": n_bins, "bins": bins}


def accuracy_coverage(confidence: np.ndarray, correct: np.ndarray, taus) -> list[dict]:
    """The reject option at each threshold: a call is made when confidence >= tau, and "no call" otherwise.
    Coverage is the share of calls made, accuracy that of calls made that are right (nan with no calls)."""
    points = []
    for tau in taus:
        accepted = confidence >= tau
        n = int(accepted.sum())
        points.append({"tau": float(tau), "n_accepted": n, "coverage": _ratio(n, len(confidence)),
                       "accuracy": float(correct[accepted].mean()) if n else float("nan")})
    return points


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


def aggregate_by_group(preds: pd.DataFrame, classes: list[str], pooling: str = "mean", key: str = "group") -> pd.DataFrame:
    """One row per group (the isolate for CMU data), or per value of `key`, from its rows' probabilities: their
    mean, or with `max` each class's highest probability over the rows, renormalised to sum to 1."""
    cols = prob_columns(classes)
    keep = {"species": "first", **({"group": "first"} if key != "group" else {})}
    table = preds.groupby(key).agg({**keep, **{c: pooling for c in cols}}).reset_index()
    if pooling == "max":
        table[cols] = table[cols].div(table[cols].sum(axis=1), axis=0)
    return table


def split_levels(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A prediction table's image rows, and the rows its isolates are pooled from.

    Without bag rows both are the image rows. With isolate or isolate-and-device bags (`level` image and bag),
    isolates are pooled from the bag rows. With tile bags (`level` tile and bag) each bag is one image, so the
    bag rows are the images, and isolates are pooled from them."""
    if "level" not in preds:
        return preds, preds
    bags = preds[preds["level"] == "bag"]
    if bags.empty:
        return preds, preds
    if (preds["level"] == "tile").any():
        return bags, bags
    return preds[preds["level"] == "image"], bags


def isolate_table(preds: pd.DataFrame, classes: list[str], pooling: str = "mean") -> pd.DataFrame:
    return aggregate_by_group(split_levels(preds)[1], classes, pooling)


def _confidence_and_correct(table: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probs = table[prob_columns(classes)].to_numpy()
    y_true = table["species"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return probs.max(axis=1), probs.argmax(axis=1) == y_true


TAU_RULES = ("none", "min_accuracy", "min_coverage")


def choose_tau(preds: pd.DataFrame, classes: list[str], rule: str, target: float, pooling: str = "mean") -> dict:
    """The reject threshold tau from development predictions, by `rule`:

    min_accuracy: the lowest tau whose accepted isolates are at least `target` accurate (the most coverage).
    min_coverage: the highest tau that still makes a call on at least `target` of isolates.

    Candidates are the isolates' own confidences, so tau is the confidence of the least confident accepted
    development isolate. Refuses any sealed test or Pool B row: tau is never tuned where it is evaluated."""
    from .splits import held_out_mask

    held_out = held_out_mask(preds)
    if held_out.any():
        groups = sorted(set(preds.loc[held_out, "group"]))
        raise ValueError(f"tau is tuned on development rows only, but {int(held_out.sum())} rows are sealed test "
                         f"or Pool B, in groups {groups[:10]}")
    if rule not in TAU_RULES[1:]:
        raise ValueError(f"tau rule {rule!r}; expected one of {list(TAU_RULES[1:])}")
    confidence, correct = _confidence_and_correct(isolate_table(preds, classes, pooling), classes)
    candidates = accuracy_coverage(confidence, correct, np.unique(confidence))
    key = "accuracy" if rule == "min_accuracy" else "coverage"
    passing = [p for p in candidates if p[key] >= target]
    record = {"rule": rule, "target": target, "pooling": pooling, "n_isolates": len(confidence),
              "chosen_on": "development out-of-fold isolates"}
    if not passing:
        return {**record, "tau": None, "reason": f"no threshold reaches {key} {target} on development isolates"}
    best = passing[0] if rule == "min_accuracy" else passing[-1]
    return {**record, "tau": best["tau"], "coverage": best["coverage"], "accuracy": best["accuracy"]}


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


def reject_option(confidence: np.ndarray, correct: np.ndarray, tau: float | None) -> dict:
    block: dict = {"curve": accuracy_coverage(confidence, correct, TAU_GRID)}
    if tau is not None:
        at = accuracy_coverage(confidence, correct, [tau])[0]
        block["at_tau"] = {**at, "wilson95": wilson_interval(round(at["accuracy"] * at["n_accepted"]), at["n_accepted"])
                           if at["n_accepted"] else [float("nan")] * 2}
    return block


def _level_reports(images: pd.DataFrame, units: pd.DataFrame, classes: list[str], n_boot: int, seed: int,
                   genus_map: Mapping[str, str], order_map: Mapping[str, str], pooling: str, tau: float | None,
                   calibration_bins: int) -> dict:
    """Isolate level pools `units` by group; image level scores `images` as they are. tau, chosen on isolates,
    is applied to the isolate level only."""
    class_idx = {c: i for i, c in enumerate(classes)}
    cols = prob_columns(classes)
    result = {}
    for level, table, role, level_tau in (("isolate_level", aggregate_by_group(units, classes, pooling), "primary", tau),
                                          ("image_level", images, "secondary", None)):
        y_true, probs = table["species"].map(class_idx).to_numpy(), table[cols].to_numpy()
        report = {"role": role, **classification_report(y_true, probs, classes)}
        confidence, correct = _confidence_and_correct(table, classes)
        report["calibration"] = expected_calibration_error(confidence, correct, calibration_bins)
        report["reject_option"] = reject_option(confidence, correct, level_tau)
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
                         subgroups: Sequence[str] = (), label_map: Mapping[str, Sequence[str]] | None = None,
                         pooling: str = "mean", tau: float | None = None, calibration_bins: int = 10) -> dict:
    """preds: one row per image with species, group and prob_<class> columns, and with bags also one row
    per bag, told apart by a `level` column (see split_levels).

    genus_map / order_map: species -> genus and genus -> order for taxonomic rollups.
    subgroups: columns of `preds` (e.g. device, phase); the same report is repeated per value, with
        isolates pooled from the subgroup's own images as the headline pools them (through their bags first).
    label_map: reference label -> model classes, for an external set with coarser labels.
    pooling: mean | max, how a group's rows become its isolate prediction.
    tau: reject threshold; each level then reports accuracy and coverage at it (tau is never tuned here).
    calibration_bins: equal-width confidence bins of the ECE and reliability table.
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
    images, units = split_levels(preds)
    through_bags = units is not images  # image and bag rows: pool a subgroup's images by bag, then by isolate

    def reports(images: pd.DataFrame, units: pd.DataFrame) -> dict:
        return _level_reports(images, units, classes, n_boot, seed, genus, order, pooling, tau, calibration_bins)

    result.update(reports(images, units))
    missing = [c for c in subgroups if c not in preds]
    if missing:
        raise ValueError(f"subgroup columns not in the prediction table: {missing}")
    if subgroups:
        result["subgroups"] = {
            col: {str(value): {"n_images": len(rows),
                               **reports(rows, aggregate_by_group(rows, classes, pooling, "bag") if through_bags else rows)}
                  for value, rows in images.groupby(col, dropna=False)}
            for col in subgroups
        }
    return result
