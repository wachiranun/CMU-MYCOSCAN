"""The model card written beside a final model, for the web app and its readers.

model_card.json says what the model is for, what it calls (classes, modality, pooling, the reject
threshold tau), what it was trained on (data hashes from the run's provenance, the checkpoint's own
hash) and how well it did where it was scored (validation or out-of-fold development predictions;
never the sealed test set, which training never sees).
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import read_metadata
from .provenance import sha256_file

INTENDED_USE = ("Research decision support for clinical mycologists identifying fungal isolates from colony or "
                "microscopy images of the classes listed. Not a diagnostic device: every call is reviewed by a "
                "mycologist, and a no-call (top probability below tau) is referred for conventional identification.")
OUT_OF_SCOPE = ("Species outside the class list, which the model can only misname; images of another modality than "
                "the one stated; use without review by a mycologist.")


def _headline(level: dict) -> dict:
    return {"n": level["n"], "accuracy": level["accuracy"], "top2_accuracy": level.get("top2_accuracy"),
            "macro_f1": level["macro"]["f1"], "accuracy_wilson95": level.get("wilson95", {}).get("accuracy")}


def write_model_card(checkpoint: Path, metrics: dict, summary: dict | None = None) -> Path:
    """model_card.json beside `checkpoint`, with headline metrics from `metrics` (the report of the predictions
    the run scored) and, from `summary`, the spread of isolate scores over folds."""
    meta = read_metadata(checkpoint)
    prov = meta.get("provenance") or {}
    bag = meta.get("bag", "none")
    card = {
        "intended_use": INTENDED_USE,
        "out_of_scope": OUT_OF_SCOPE,
        "classes": meta["classes"],
        "modality": meta.get("modality"),
        "source": meta.get("source"),
        "arch": meta["arch"],
        "image_size": meta["image_size"],
        "pooling": {"bag": bag, "type": meta.get("pooling") if bag != "none" else None,
                    "hierarchy": meta.get("pooling_hierarchy", "none")},
        "tau": meta.get("tau"),
        "tau_selection": meta.get("tau_selection"),
        "leaky": meta.get("leaky", False),
        "checkpoint": {"file": checkpoint.name, "fold": meta.get("fold"), "sha256": sha256_file(checkpoint)},
        "training_data": {"manifest": prov.get("manifest"), "manifest_sha256": prov.get("manifest_sha256"),
                          "splits_sha256": prov.get("splits_sha256"), "held_out": prov.get("held_out"),
                          "fold_membership_sha256": (prov.get("folds") or {}).get("membership_sha256")},
        "code": {"commit": prov.get("commit"), "dirty": prov.get("dirty")},
        "headline_metrics": {"scored_on": metrics.get("note"), "fold": metrics.get("fold"),
                             "isolate_level": _headline(metrics["isolate_level"]),
                             "image_level": _headline(metrics["image_level"])},
    }
    if summary is not None:
        card["headline_metrics"]["across_folds"] = {k: {s: summary[k][s] for s in ("mean", "sd", "n")}
                                                    for k in ("isolate_macro_f1", "isolate_accuracy")}
    path = checkpoint.parent / "model_card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path
