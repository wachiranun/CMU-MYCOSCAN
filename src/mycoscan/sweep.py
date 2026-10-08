"""Sweeps: one base config, a list of override cells, one run per cell.

    base = "cmu_microscopic.toml"     # relative to the sweep file
    name = "p7_learning_curve"        # optional, default the sweep file's stem
    [[cells]]
    train_fraction = 0.1
    [[cells]]
    train_fraction = 0.5

Each cell runs as `<base run_name>__<key>-<value>_...` (keys sorted) under
`<base output_dir>/<name>/`, so its name says what it changed and a rerun lands
in the same place. A cell that fails is recorded and the rest still run.
"""
from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path

from .config import config_with

log = logging.getLogger("mycoscan")


def _token(value) -> str:
    if isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, (list, tuple)):
        text = "+".join(_token(v) for v in value)
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9.+-]", "-", text)


def cell_name(base_run_name: str, overrides: dict) -> str:
    return base_run_name + "__" + "_".join(f"{k}-{_token(overrides[k])}" for k in sorted(overrides))


def run_sweep(sweep_file: str | Path) -> dict:
    from .pipeline import run_training

    sweep_file = Path(sweep_file)
    spec = tomllib.loads(sweep_file.read_text(encoding="utf-8"))
    unknown = sorted(set(spec) - {"base", "name", "cells"})
    if unknown or not spec.get("cells"):
        raise ValueError(f"{sweep_file}: a sweep has `base`, an optional `name` and a list of [[cells]]; "
                         f"unknown keys {unknown}")
    base_path = sweep_file.parent / spec["base"]
    base = config_with(base_path, {})
    name = spec.get("name", sweep_file.stem)
    sweep_dir = Path(base.output_dir) / name
    reserved = sorted({k for overrides in spec["cells"] for k in overrides} & {"run_name", "output_dir"})
    if reserved:
        raise ValueError(f"{sweep_file}: cells may not set {reserved}; the sweep names and places every run")
    names = [cell_name(base.run_name, overrides) for overrides in spec["cells"]]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"{sweep_file}: cells would write the same run {repeated}; make their overrides differ")
    cells = []
    for run_name, overrides in zip(names, spec["cells"]):
        cell = {"run_name": run_name, "overrides": overrides}
        try:
            cfg = config_with(base_path, {**overrides, "run_name": run_name, "output_dir": str(sweep_dir)})
            cell |= {"status": "ok", "run_dir": str(run_training(cfg))}
        except Exception as e:  # one failed cell must not stop the sweep; it is named in the summary
            log.exception("sweep cell %s failed", run_name)
            cell |= {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        cells.append(cell)
    summary = {"sweep": str(sweep_file), "base": str(base_path), "cells": cells,
               "failed": [c["run_name"] for c in cells if c["status"] == "failed"]}
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
