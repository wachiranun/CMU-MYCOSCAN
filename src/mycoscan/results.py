"""Benchmark tables generated from finished runs, never typed by hand.

Every `metrics.json` under a directory is one run row: one fold of one seed, a
seed's pooled out-of-fold predictions (`fold = "pooled"`), or a holdout run. A
linear-probe run gives one row per classifier. Cell rows aggregate the fold rows
of each config cell (a sweep cell, or one multi-seed run) as mean and SD over
folds and seeds; pooled rows are left out of that, since they repeat the folds.

Run-level rows (pooled, holdout, evaluation) come first, in run-name order, then the
fold rows, so twin runs named alike (P0's `..._grouped` and `..._leaky`) sit side by
side. A leaky run's rows carry the marker "LEAKY, comparison only".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_COLUMNS = {
    "isolate_macro_f1": ("isolate_level", "macro", "f1"),
    "isolate_accuracy": ("isolate_level", "accuracy"),
    "isolate_top2_accuracy": ("isolate_level", "top2_accuracy"),
    "n_isolates": ("isolate_level", "n"),
    "isolate_ece": ("isolate_level", "calibration", "ece"),
    "isolate_coverage_at_tau": ("isolate_level", "reject_option", "at_tau", "coverage"),
    "isolate_accuracy_at_tau": ("isolate_level", "reject_option", "at_tau", "accuracy"),
    "image_macro_f1": ("image_level", "macro", "f1"),
    "image_accuracy": ("image_level", "accuracy"),
}
RUN_COLUMNS = {
    "tau": ("tau", "value"),
    "train_fraction": ("subsample", "train_fraction"),
    "device": ("resources", "device"),
    "wall_seconds": ("resources", "wall_seconds"),
    "gpu_minutes": ("resources", "gpu_minutes"),
    "peak_memory_mb": ("resources", "peak_memory_mb"),
    "commit": ("provenance", "commit"),
    "dirty": ("provenance", "dirty"),
    "manifest_sha256": ("provenance", "manifest_sha256"),
}
X_AXES = {"images": "images_per_class", "groups": "groups_per_class"}
LEAKY_MARKER = "LEAKY, comparison only"
CELL_METRICS = ("isolate_macro_f1", "isolate_accuracy")


def _dig(block: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(block, dict) or key not in block:
            return float("nan")
        block = block[key]
    return block


def _mean_per_class(m: dict, key: str) -> float:
    """Images or groups per class a run kept after train_fraction subsampling, averaged over classes."""
    per_class = _dig(m, ("subsample", key))
    return float(np.mean(list(per_class.values()))) if isinstance(per_class, dict) and per_class else float("nan")


def run_rows(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.relative_to(root).as_posix() or "."
        common = {"run": run, "run_name": m.get("run_name", path.parent.name), "cell": m.get("cell", run),
                  "seed": m.get("seed"), "fold": m.get("fold", ""), "split": m.get("split", ""),
                  "leaky": m.get("leaky", False), "marker": LEAKY_MARKER if m.get("leaky") else "",
                  **{k: _dig(m, p) for k, p in RUN_COLUMNS.items()},
                  **{column: _mean_per_class(m, column) for column in X_AXES.values()}}
        blocks = m.get("classifiers") or {m.get("classifier", "network"): m}
        for classifier, block in blocks.items():
            rows.append({**common, "classifier": classifier, **{k: _dig(block, p) for k, p in LEVEL_COLUMNS.items()}})
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    fold_row = ~runs["fold"].isin(["pooled", "holdout", ""])
    return runs.assign(_fold_row=fold_row).sort_values(["_fold_row", "run_name"], kind="stable").drop(
        columns="_fold_row").reset_index(drop=True)


def cell_rows(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    folds = runs[runs["fold"] != "pooled"]
    for (cell, classifier), group in folds.groupby(["cell", "classifier"], sort=True):
        leaky = bool(group["leaky"].any())
        row = {"cell": cell, "classifier": classifier, "leaky": leaky, "marker": LEAKY_MARKER if leaky else "",
               "n_runs": len(group), "seeds": group["seed"].nunique(), "folds": group["fold"].nunique()}
        for metric in CELL_METRICS:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_results_table(root: str | Path, out: str | Path) -> tuple[Path, Path]:
    """`out` gets one row per run; `<out stem>_cells.csv` beside it one row per cell and classifier."""
    runs = run_rows(root)
    if runs.empty:
        raise ValueError(f"no metrics.json under {root}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cells_out = out.with_name(f"{out.stem}_cells.csv")
    runs.to_csv(out, index=False)
    cell_rows(runs).to_csv(cells_out, index=False)
    return out, cells_out


def learning_curve(root: str | Path, out: str | Path, x: str = "images", classifier: str = "") -> tuple[Path, Path]:
    """Isolate macro-F1 against images (or groups) per class, one point per cell and seed: the seed's pooled
    out-of-fold score, or its holdout fold. The plot shows each seed and the mean with a +-SD band; the CSV
    beside it (`<out stem>.csv`) holds every point. `classifier` picks one of a probe run's classifiers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = run_rows(root)
    if runs.empty:
        raise ValueError(f"no metrics.json under {root}")
    # By default each run's primary classifier: the network, or a linear probe's logistic regression.
    runs = runs[runs["classifier"] == classifier] if classifier else runs[runs["classifier"].isin(["network", "logreg"])]
    per_seed = runs[runs["fold"].isin(["pooled", "holdout"])]
    points = per_seed.rename(columns={X_AXES[x]: "x", "isolate_macro_f1": "y"})[["cell", "seed", "x", "y"]]
    points = points.sort_values(["x", "seed"]).reset_index(drop=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    points.to_csv(csv, index=False)

    by_x = points.groupby("x")["y"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for seed, seed_points in points.groupby("seed"):
        ax.plot(seed_points["x"], seed_points["y"], marker="o", lw=0.8, alpha=0.4, label=f"seed {seed}")
    ax.plot(by_x["x"], by_x["mean"], color="black", lw=1.8, label="mean")
    ax.fill_between(by_x["x"], by_x["mean"] - by_x["std"].fillna(0), by_x["mean"] + by_x["std"].fillna(0),
                    color="black", alpha=0.12, label="mean +- SD")
    ax.set_xlabel(f"{x} per class")
    ax.set_ylabel("isolate macro-F1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out, csv
