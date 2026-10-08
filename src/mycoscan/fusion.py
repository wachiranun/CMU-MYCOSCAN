"""Late fusion of a colony run and a microscopy run at the isolate level.

`mycoscan fuse` reads two finished runs, pools each run's out-of-fold predictions into
isolates (by the run's own pooling) and joins them on the isolate. The runs must agree on
classes, seeds, folds and test isolates; anything else is refused, naming the difference.
An isolate imaged in one modality only is left out and listed.

`weighted` fuses p = w * p_colony + (1 - w) * p_micro, with w on a grid of 0 to 1 in 0.05
steps, chosen by isolate macro-F1 (ties go to the w nearest 0.5). `mlp` trains a small MLP on
the concatenated isolate embeddings of both branches: each image embedded by the branch's
frozen initial backbone, through the same feature cache a linear probe uses, and averaged
over the isolate. Frozen features keep the MLP's inputs free of what the branch networks
learned from the isolates it is scored on.

Both are cross-fitted on the development folds: each fold's isolates are fused by a weight
(or MLP) fitted on the other folds, so the fused out-of-fold numbers are as honest as the
branches'. The final weight (or MLP) is fitted on every development isolate of every seed
and applied once to the test rows, when the evaluations of both branches' final models on
the test set are given. Test rows never reach a fit.

The fused run directory has the files of a training run (predictions, metrics, per-fold
metrics, summary), so it enters the results table and the paired comparison like any run.
Its rows are isolates, so its image level repeats its isolate level.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import cache_key, cached_features
from .metrics import classification_report, evaluate_predictions, isolate_table, prob_columns, split_levels
from .models import build_model, is_checkpoint
from .paired import MIXED_TRANSFER
from .pipeline import resolve_device, write_report, write_summary
from .provenance import git_state

FUSION_METHODS = ("weighted", "mlp")
WEIGHT_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)

Fit = Callable[[pd.DataFrame], tuple[Callable[[pd.DataFrame], np.ndarray], dict]]


@dataclass(frozen=True)
class Branch:
    dir: Path
    name: str
    config: dict
    provenance: dict
    classes: list[str]

    @property
    def seeds(self) -> list:
        return list(self.provenance["seeds"])

    @property
    def folds(self) -> dict:
        return self.provenance["folds"]

    @property
    def test_isolates(self) -> dict:
        return {"splits file": self.provenance.get("splits_sha256"),
                "held-out groups": (self.provenance.get("held_out") or {}).get("sha256")}

    def predictions(self, seed) -> pd.DataFrame:
        return pd.read_csv(self.dir / self.provenance["pooled_predictions"][str(seed)]["path"])

    def isolates(self, preds: pd.DataFrame) -> pd.DataFrame:
        return isolate_rows(preds, self.classes, self.config.get("pooling", "mean"))


def load_branch(run_dir: str | Path) -> Branch:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["provenance"]
    first = provenance["pooled_predictions"][str(provenance["seeds"][0])]["path"]
    columns = pd.read_csv(run_dir / first, nrows=0).columns
    return Branch(run_dir, config.get("run_name", run_dir.name), config, provenance,
                  [c.removeprefix("prob_") for c in columns if c.startswith("prob_")])


def isolate_rows(preds: pd.DataFrame, classes: list[str], pooling: str) -> pd.DataFrame:
    """One row per isolate of the primary classifier's rows: group, species, fold and pooled probabilities."""
    if "classifier" in preds:
        preds = preds[preds["classifier"] == preds["classifier"].iat[0]]
    table = isolate_table(preds, classes, pooling)
    return table.assign(fold=table["group"].map(split_levels(preds)[1].groupby("group")["fold"].first()))


def check_aligned(colony: Branch, micro: Branch) -> None:
    """Refuse two runs whose classes, seeds, folds or test isolates differ."""
    names = f"{colony.name} and {micro.name}"
    if colony.classes != micro.classes:
        only_c = [c for c in colony.classes if c not in micro.classes]
        only_m = [c for c in micro.classes if c not in colony.classes]
        detail = (f"only in {colony.name}: {only_c}, only in {micro.name}: {only_m}" if only_c or only_m
                  else f"same classes in another order: {colony.classes} and {micro.classes}")
        raise ValueError(f"{names} have different classes: {detail}")
    for what in colony.test_isolates:
        a, b = colony.test_isolates[what], micro.test_isolates[what]
        if a != b:
            raise ValueError(f"{names} hold out different test isolates: {what} {a} and {b}")
    if sorted(colony.seeds) != sorted(micro.seeds):
        raise ValueError(f"{names} have different seeds: {colony.seeds} and {micro.seeds}")
    if colony.folds["names"] != micro.folds["names"]:
        raise ValueError(f"{names} have different folds: {colony.folds['names']} and {micro.folds['names']}")


def join_isolates(colony: pd.DataFrame, micro: pd.DataFrame, what: str) -> tuple[pd.DataFrame, dict]:
    """The isolates both tables score, with each branch's probabilities, and those only one scores.
    Refuses an isolate whose species or fold differ between the branches."""
    joined = colony.merge(micro, on="group", suffixes=("_colony", "_micro"))
    for column in ("species", "fold"):
        differ = joined.loc[joined[f"{column}_colony"] != joined[f"{column}_micro"], "group"].tolist()
        if differ:
            raise ValueError(f"{what}: isolates with a different {column} in the two runs: {differ[:10]}")
    joined = joined.rename(columns={"species_colony": "species", "fold_colony": "fold"}).drop(
        columns=["species_micro", "fold_micro"])
    unmatched = {"only_colony": sorted(set(colony["group"]) - set(micro["group"])),
                 "only_micro": sorted(set(micro["group"]) - set(colony["group"]))}
    return joined.reset_index(drop=True), unmatched


def branch_probs(rows: pd.DataFrame, classes: list[str], branch: str) -> np.ndarray:
    return rows[[f"{c}_{branch}" for c in prob_columns(classes)]].to_numpy()


def labels(rows: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return rows["species"].map({c: i for i, c in enumerate(classes)}).to_numpy()


def tune_weight(rows: pd.DataFrame, classes: list[str]) -> tuple[float, list[dict]]:
    """The colony weight with the best isolate macro-F1 on `rows`, ties to the weight nearest 0.5, and the grid."""
    y, colony, micro = labels(rows, classes), branch_probs(rows, classes, "colony"), branch_probs(rows, classes, "micro")
    grid = [{"weight": float(w), "macro_f1": classification_report(y, w * colony + (1 - w) * micro, classes)["macro"]["f1"]}
            for w in WEIGHT_GRID]
    best = max(g["macro_f1"] for g in grid)
    weight = min((g["weight"] for g in grid if g["macro_f1"] == best), key=lambda w: (abs(w - 0.5), w))
    return weight, grid


def weighted_fit(classes: list[str]) -> Fit:
    def fit(train: pd.DataFrame):
        weight, grid = tune_weight(train, classes)

        def apply(rows: pd.DataFrame) -> np.ndarray:
            return weight * branch_probs(rows, classes, "colony") + (1 - weight) * branch_probs(rows, classes, "micro")
        return apply, {"weight": weight, "grid": grid}
    return fit


def mlp_fit(classes: list[str], embeddings: pd.DataFrame, seed: int) -> Fit:
    """An MLP on the concatenated colony and microscopy embeddings of each isolate (`embeddings` by group)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(train: pd.DataFrame):
        y = labels(train, classes)
        model = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=1000, random_state=seed))
        model.fit(embeddings.loc[train["group"]].to_numpy(), y)

        def apply(rows: pd.DataFrame) -> np.ndarray:
            probs = np.zeros((len(rows), len(classes)))
            probs[:, model.classes_] = model.predict_proba(embeddings.loc[rows["group"]].to_numpy())
            return probs
        return apply, {"trained_isolates": int(train["group"].nunique())}
    return fit


def cross_fit(rows: pd.DataFrame, fit: Fit) -> tuple[np.ndarray, dict]:
    """Each fold's rows fused by a fit on the other folds; a single fold (holdout) is fitted on itself."""
    out: np.ndarray | None = None
    per_fold = {}
    for fold in rows["fold"].unique():
        inside = (rows["fold"] == fold).to_numpy()
        apply, info = fit(rows[~inside] if (~inside).any() else rows)
        fused = apply(rows[inside])
        if out is None:
            out = np.zeros((len(rows), fused.shape[1]))
        out[inside] = fused
        per_fold[str(fold)] = info
    assert out is not None
    return out, per_fold


def isolate_embeddings(branch: Branch, preds: list[pd.DataFrame], device: str) -> pd.DataFrame:
    """The mean embedding of each isolate's images by the branch's frozen initial backbone, from its feature cache."""
    cfg = branch.config
    images = pd.concat(preds, ignore_index=True)
    images = images[images["image_path"].notna()].drop_duplicates("image_path").reset_index(drop=True)
    key = cache_key(branch.provenance["manifest_sha256"], cfg["arch"], cfg["weights"], cfg["image_size"],
                    cfg["autocontrast"], cfg["plate_crop"])
    torch.manual_seed(0)  # weights = "none" is a random backbone; the same one every time
    backbone = build_model(cfg["arch"], len(branch.classes), cfg["weights"], cfg["image_size"])
    features, _ = cached_features(images, Path(cfg["output_dir"]) / "feature_cache" / f"{key}.npz", backbone,
                                  cfg["image_size"], cfg["autocontrast"], cfg["plate_crop"], device,
                                  cfg.get("batch_size", 32))
    return pd.DataFrame(features).groupby(images["group"].to_numpy()).mean()


def fused_table(rows: pd.DataFrame, probs: np.ndarray, classes: list[str], method: str, split: str,
                weights: np.ndarray | None) -> pd.DataFrame:
    table = rows[["group", "species", "fold"]].copy()
    table["classifier"] = f"fusion_{method}"
    table["split"] = split
    if weights is not None:
        table["weight"] = weights
    table["predicted"] = [classes[i] for i in probs.argmax(axis=1)]
    table[prob_columns(classes)] = probs
    return table


def _combined_sha(a: str | None, b: str | None) -> str:
    return hashlib.sha256(f"{a}\n{b}".encode("utf-8")).hexdigest()


def transfer_weights(colony: Branch, micro: Branch) -> str:
    """The fused run's `weights`, read by the paired comparison: a branch's, when both share a transfer role."""
    roles = {is_checkpoint(colony.config["weights"]), is_checkpoint(micro.config["weights"])}
    return colony.config["weights"] if len(roles) == 1 else MIXED_TRANSFER


def fuse_runs(colony_run: str | Path, micro_run: str | Path, out: str | Path, method: str = "weighted",
              colony_test: str | Path | None = None, micro_test: str | Path | None = None, n_boot: int = 2000,
              seed: int = 0, device: str = "auto") -> Path:
    """Fuse two runs into the run directory `out`; colony_test and micro_test are `mycoscan eval` directories
    of the two branches' final models on the test set, given together or not at all."""
    if method not in FUSION_METHODS:
        raise ValueError(f"fusion method {method!r}; expected one of {list(FUSION_METHODS)}")
    if (colony_test is None) != (micro_test is None):
        raise ValueError("give the test evaluations of both branches, or neither")
    colony, micro = load_branch(colony_run), load_branch(micro_run)
    check_aligned(colony, micro)
    classes, seeds, out = colony.classes, colony.seeds, Path(out)
    dev_preds = {s: (colony.predictions(s), micro.predictions(s)) for s in seeds}
    dev, unmatched = {}, {}
    for s, (c, m) in dev_preds.items():
        dev[s], unmatched[str(s)] = join_isolates(colony.isolates(c), micro.isolates(m), f"seed {s}")
    test = None
    if colony_test is not None and micro_test is not None:
        c_test, m_test = (pd.read_csv(Path(d) / "predictions.csv") for d in (colony_test, micro_test))
        test, test_unmatched = join_isolates(colony.isolates(c_test), micro.isolates(m_test), "test")
        if test_unmatched["only_colony"] or test_unmatched["only_micro"]:
            raise ValueError(f"the test evaluations score different isolates: only colony: "
                             f"{test_unmatched['only_colony'][:10]}, only micro: {test_unmatched['only_micro'][:10]}")

    if method == "weighted":
        fit = weighted_fit(classes)
    else:
        resolved = resolve_device(device)
        colony_preds = [c for c, _ in dev_preds.values()] + ([c_test] if test is not None else [])
        micro_preds = [m for _, m in dev_preds.values()] + ([m_test] if test is not None else [])
        embeddings = pd.concat([isolate_embeddings(colony, colony_preds, resolved),
                                isolate_embeddings(micro, micro_preds, resolved)],
                               axis=1, join="inner", keys=["colony", "micro"])
        fit = mlp_fit(classes, embeddings, seed)

    name = out.name
    leaky = any(b.config.get("split") == "image_random" for b in (colony, micro))
    folds = colony.folds["names"]
    holdout = folds == ["holdout"]
    provenance = {**git_state(), "seeds": seeds, "splits_sha256": colony.provenance.get("splits_sha256"),
                  "held_out": colony.provenance.get("held_out"),
                  "folds": {"names": folds,
                            "membership_sha256": _combined_sha(colony.folds.get("membership_sha256"),
                                                               micro.folds.get("membership_sha256")),
                            "per_fold_sha256": {f: _combined_sha(colony.folds["per_fold_sha256"].get(f),
                                                                 micro.folds["per_fold_sha256"].get(f))
                                                for f in folds}},
                  "branches": {"colony": {"run": str(colony.dir), "commit": colony.provenance.get("commit")},
                               "micro": {"run": str(micro.dir), "commit": micro.provenance.get("commit")}}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "run_name": name, "fusion": method, "colony_run": str(colony.dir), "micro_run": str(micro.dir),
        "colony_test": str(colony_test) if colony_test else None, "micro_test": str(micro_test) if micro_test else None,
        "arch": f"{colony.config['arch']}+{micro.config['arch']}", "weights": transfer_weights(colony, micro),
        "classes": classes, "split": colony.config.get("split"), "bootstrap": n_boot, "seed": seed}, indent=2),
        encoding="utf-8")
    base = {"run_name": name, "cell": name, "split": colony.config.get("split", ""), "leaky": leaky,
            "provenance": provenance}

    apply, final_info = fit(pd.concat(list(dev.values()), ignore_index=True))
    record = {"method": method, "tuned_on": "development out-of-fold isolates of every seed; each fold's rows "
                                            "fused by a fit on the other folds",
              "cross_fitted": not holdout,
              "unmatched_isolates": unmatched,
              **({"weight": final_info["weight"], "grid": final_info["grid"]} if method == "weighted" else {})}
    fold_scores, pooled = [], {}
    for s, rows in dev.items():
        seed_dir = out / f"seed{s}" if len(seeds) > 1 else out
        seed_dir.mkdir(parents=True, exist_ok=True)
        probs, per_fold = cross_fit(rows, fit)
        weights = rows["fold"].astype(str).map({f: i["weight"] for f, i in per_fold.items()}).to_numpy() \
            if method == "weighted" else None
        table = fused_table(rows, probs, classes, method, "dev", weights)
        fusion = {**record, **({"fold_weights": {f: i["weight"] for f, i in per_fold.items()}}
                               if method == "weighted" else {})}
        metrics = write_report(seed_dir, table, classes, n_boot, s,
                               {**base, "seed": s, "fold": "holdout" if holdout else "pooled", "fusion": fusion,
                                "tau": {"value": None, "source": "fusion: no reject threshold"},
                                "note": "fused out-of-fold isolate rows; image level repeats isolate level"})
        if holdout:
            fold_scores.append(metrics)
        else:
            for fold in folds:
                fold_dir = seed_dir / "folds" / fold
                fold_dir.mkdir(parents=True, exist_ok=True)
                fold_metrics = {**base, "seed": s, "fold": fold, "fusion": {"method": method, **{
                    k: v for k, v in per_fold.get(fold, {}).items() if k != "grid"}},
                                **evaluate_predictions(table[table["fold"] == fold].reset_index(drop=True), classes,
                                                       0, s)}
                (fold_dir / "metrics.json").write_text(json.dumps(fold_metrics, indent=2), encoding="utf-8")
                fold_scores.append(fold_metrics)
        pooled[s] = seed_dir / "predictions.csv"
    write_summary(out, base, fold_scores, pooled)

    if test is not None:
        test_dir = out / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        weight = final_info.get("weight")
        table = fused_table(test, apply(test), classes, method, "test",
                            np.full(len(test), weight) if weight is not None else None)
        write_report(test_dir, table, classes, n_boot, seed,
                     {"run_name": f"{name} (test)", "leaky": leaky, "provenance": provenance,
                      "tau": {"value": None, "source": "fusion: no reject threshold"},
                      "fusion": {"method": method, "test_scored": "once", "fitted_on": record["tuned_on"],
                                 **({"weight": weight} if weight is not None else {}),
                                 "colony_test": str(colony_test), "micro_test": str(micro_test)}})
    return out
