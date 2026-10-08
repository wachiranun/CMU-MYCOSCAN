"""mycoscan command line: env | make-synthetic | build-openfungi | partition | seal | train | sweep | eval | explain |
predict | compare | results | learning-curve."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .splits import MIN_ISOLATES_TO_SEAL


def _env(_args) -> None:
    import torch

    cuda = torch.cuda.is_available()
    print(json.dumps({"torch": torch.__version__, "cuda_available": cuda,
                      "device": torch.cuda.get_device_name(0) if cuda else "cpu"}, indent=2))


def _make_synthetic(args) -> None:
    from .synthetic import make_synthetic

    print(make_synthetic(args.out, size=args.size, fovs_per_device=args.fovs, seed=args.seed))


def _build_openfungi(args) -> None:
    from .openfungi import build_openfungi_manifest, load_grouping_config

    manifest, summary = build_openfungi_manifest(args.root, args.out, load_grouping_config(args.config, args.set))
    table = pd.DataFrame(summary["classes"])
    table["under_powered"] = table["under_powered"].map({True: "UNDER-POWERED", False: ""})
    print(table.to_string(index=False))
    if summary["under_powered"]:
        min_images = summary["config"]["min_images"]
        print(f"under-powered (< {min_images} images): {', '.join(summary['under_powered'])}")
    print(manifest)


def _partition(args) -> None:
    from .splits import write_splits_file

    print(write_splits_file(args.manifest, args.out, args.b_fraction, args.n_folds, args.seed))


def _seal(args) -> None:
    from .splits import write_sealed_file

    print(write_sealed_file(args.manifest, args.out, args.test_fraction, args.n_folds, args.seed, args.force))


def _train(args) -> None:
    from .config import load_config
    from .pipeline import run_training

    print(run_training(load_config(args.config, args.set)))


METRIC_OPTIONS = {"label_map", "genus_map", "order_map", "subgroups"}


def _metric_options(path: str | None) -> dict:
    if not path:
        return {}
    import tomllib

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - METRIC_OPTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown metric options {unknown}; expected some of {sorted(METRIC_OPTIONS)}")
    return raw


def _eval(args) -> None:
    from .pipeline import evaluate_checkpoint

    m = evaluate_checkpoint(args.checkpoint, args.manifest, args.out, args.modality, args.source, args.bootstrap,
                            args.device, **_metric_options(args.metric_config))
    print(json.dumps({level: {k: m[level][k] for k in ("role", "accuracy", "top2_accuracy", "kappa", "macro")}
                      for level in ("isolate_level", "image_level")}, indent=2))


def _explain(args) -> None:
    from .explain import explain_images
    from .manifest import load_manifest

    labels = modalities = None
    images = args.images
    if args.manifest:
        df = load_manifest(args.manifest)
        if args.modality:
            df = df[df["modality"] == args.modality]
        df = df.groupby("species", group_keys=False).head(args.per_class)
        images, labels, modalities = df["image_path"].tolist(), df["species"].tolist(), df["modality"].tolist()
    sheet = explain_images(args.checkpoint, images, args.out, args.device, labels, modalities)
    print(f"{len(sheet)} panels and review_sheet.csv written to {args.out}")


def _predict(args) -> None:
    from .predict import Predictor

    predictor = Predictor(args.checkpoint, args.device)
    print(json.dumps({img: predictor.predict(img) for img in args.images}, indent=2))


def _with_ci(value: float, ci: dict, key: str) -> str:
    return f"{value:.3f}" + (f" [{ci[key][0]:.2f}-{ci[key][1]:.2f}]" if key in ci else "")


def _compare(args) -> None:
    rows = []
    for run in args.runs:
        m = json.loads((Path(run) / "metrics.json").read_text(encoding="utf-8"))
        for level in ("isolate_level", "image_level"):
            r = m[level]
            ci = r.get("ci95_isolate_bootstrap", {})
            rows.append({"run": Path(run).name, "level": level.split("_")[0], "n": r["n"],
                         "accuracy": _with_ci(r["accuracy"], ci, "accuracy"),
                         "top2_accuracy": _with_ci(r.get("top2_accuracy", float("nan")), ci, "top2_accuracy"),
                         "kappa": _with_ci(r.get("kappa", float("nan")), ci, "kappa"),
                         **{f"macro_{k}": _with_ci(r["macro"][k], ci, f"macro_{k}")
                            for k in ("sensitivity", "specificity", "ppv", "npv", "f1", "auc")}})
    print(pd.DataFrame(rows).to_string(index=False))


def _sweep(args) -> None:
    from .sweep import run_sweep

    summary = run_sweep(args.sweep)
    for cell in summary["cells"]:
        print(f"{cell['status']:6}  {cell['run_name']}" + (f"  {cell['error']}" if cell["status"] == "failed" else ""))
    if summary["failed"]:
        sys.exit(1)


def _results(args) -> None:
    from .results import write_results_table

    for path in write_results_table(args.root, args.out):
        print(path)


def _learning_curve(args) -> None:
    from .results import learning_curve

    for path in learning_curve(args.root, args.out, args.x, args.classifier):
        print(path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(prog="mycoscan")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="report torch version and GPU availability").set_defaults(fn=_env)

    p = sub.add_parser("make-synthetic", help="write a synthetic placeholder dataset and manifest")
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--fovs", type=int, default=6, help="microscopic FOVs per device per isolate")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_make_synthetic)

    p = sub.add_parser("build-openfungi", help="OpenFungi manifest with pHash + embedding pseudo-groups and contact sheets")
    p.add_argument("--root", required=True, help="folder holding macro/<class>/ and micro/<class>/")
    p.add_argument("--out", required=True, help="manifest to write; contact sheets go to contact_sheets/ beside it")
    p.add_argument("--config", help="grouping TOML, e.g. configs/openfungi_manifest.toml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a grouping key")
    p.set_defaults(fn=_build_openfungi)

    p = sub.add_parser("partition", help="freeze OpenFungi groups into Pool A (development) and Pool B (external test)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="splits file to create, e.g. splits_v1.csv; never overwritten")
    p.add_argument("--b-fraction", type=float, default=0.3, help="share of each class's groups in Pool B")
    p.add_argument("--n-folds", type=int, default=5, help="cross-validation folds within Pool A")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_partition)

    p = sub.add_parser("seal", help="seal the locked CMU test set; re-run to add new isolates, sealed ones never move")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="sealed split file, created or extended, e.g. data/cmu/splits_cmu.csv")
    p.add_argument("--test-fraction", type=float, default=0.15, help="share of each class's isolates in the test set")
    p.add_argument("--n-folds", type=int, default=5, help="cross-validation folds among development isolates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help=f"seal even when a class has fewer than {MIN_ISOLATES_TO_SEAL} isolates")
    p.set_defaults(fn=_seal)

    p = sub.add_parser("train", help="train with isolate-level validation")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    p.set_defaults(fn=_train)

    p = sub.add_parser("sweep", help="run every cell of a sweep file; exits 1 if any cell failed")
    p.add_argument("sweep")
    p.set_defaults(fn=_sweep)

    p = sub.add_parser("eval", help="evaluate a checkpoint on a manifest")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modality", choices=["colony", "microscopic", "all"])
    p.add_argument("--source", default="all", choices=["cmu", "openfungi", "all"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--metric-config", help="TOML with label_map, genus_map, order_map and subgroups")
    p.set_defaults(fn=_eval)

    p = sub.add_parser("explain", help="Grad-CAM and SmoothGrad panels plus an expert review sheet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+")
    group.add_argument("--manifest")
    p.add_argument("--modality", choices=["colony", "microscopic"])
    p.add_argument("--per-class", type=int, default=2)
    p.set_defaults(fn=_explain)

    p = sub.add_parser("predict", help="class probabilities for one or more images")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("images", nargs="+")
    p.set_defaults(fn=_predict)

    p = sub.add_parser("compare", help="side-by-side metrics of finished runs")
    p.add_argument("runs", nargs="+")
    p.set_defaults(fn=_compare)

    p = sub.add_parser("results", help="one CSV row per run and one per config cell, from every metrics.json under a directory")
    p.add_argument("root")
    p.add_argument("--out", required=True, help="run table; the cell table is written beside it as <stem>_cells.csv")
    p.set_defaults(fn=_results)

    p = sub.add_parser("learning-curve", help="isolate macro-F1 against images or groups per class, with per-seed bands")
    p.add_argument("root", help="a sweep directory, e.g. runs/<sweep name>")
    p.add_argument("--x", choices=["images", "groups"], default="images")
    p.add_argument("--out", required=True, help="plot (.png); the points are written beside it as .csv")
    p.add_argument("--classifier", default="", help="for linear-probe runs: logreg (default) or knn")
    p.set_defaults(fn=_learning_curve)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
