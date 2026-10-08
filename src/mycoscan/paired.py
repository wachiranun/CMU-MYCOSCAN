"""The pre-registered primary analysis: sequential against direct transfer, paired by fold and seed.

A run is sequential when its `weights` is a checkpoint (an earlier stage), and direct
when it starts from ImageNet, DINO or nothing. Each sequential run is paired with each
direct run of the same arch. For a pair, the unit is one fold of one seed: the
difference in isolate macro-F1 (sequential minus direct) on the same validation
isolates. The output is their mean, a bootstrap CI of the mean, a Wilcoxon
signed-rank p, and the verdict fixed in advance: sequential is superior only when the
CI excludes 0 and the mean gain is at least `min_gain` (2 points of macro-F1). A
sequential run that is worse on average is named negative transfer, with the
recommendation to use direct transfer: the Stage-1 guard-rail as a reported number.

Runs are paired only when they are genuinely paired: the same seeds, the same sealed
test (or Pool B) isolates, and the same validation membership in every fold. Anything
else is refused, naming what differs.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .models import is_checkpoint


MIXED_TRANSFER = "mixed"  # the `weights` of a fused run whose branches differ in transfer role


@dataclass(frozen=True)
class PairedConfig:
    min_gain: float = 0.02  # macro-F1, so 0.02 is 2 points
    ci_level: float = 0.95
    require_ci_excludes_zero: bool = True
    bootstrap: int = 2000
    seed: int = 0


def load_paired_config(path: str | Path | None = None, overrides: dict | None = None) -> PairedConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    raw |= overrides or {}
    unknown = sorted(set(raw) - {f.name for f in fields(PairedConfig)})
    if unknown:
        raise ValueError(f"unknown paired comparison keys: {unknown}")
    return PairedConfig(**raw)


@dataclass(frozen=True)
class Run:
    name: str
    arch: str
    weights: str
    seeds: list
    folds: list[str]
    membership: dict[str, str]
    test_isolates: tuple
    f1: dict[tuple, float]  # (seed, fold) -> isolate macro-F1

    @property
    def sequential(self) -> bool:
        return is_checkpoint(self.weights)


def load_run(run_dir: str | Path) -> Run:
    """A finished run directory: its config.json and summary.json (single- or multi-seed alike)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    prov = summary["provenance"]
    if cfg["weights"] == MIXED_TRANSFER:
        raise ValueError(f"{run_dir} fuses a sequential and a direct branch, so it is neither sequential nor direct")
    return Run(name=summary.get("run_name", run_dir.name), arch=cfg["arch"], weights=cfg["weights"],
               seeds=list(prov["seeds"]), folds=list(prov["folds"]["names"]),
               membership=prov["folds"].get("per_fold_sha256", {}),
               test_isolates=(prov.get("splits_sha256"), (prov.get("held_out") or {}).get("sha256")),
               f1={(v["seed"], v["fold"]): v["value"] for v in summary["isolate_macro_f1"]["values"]})


def check_paired(a: Run, b: Run) -> None:
    """Refuse two runs that are not paired by seed, test isolates and fold membership."""
    if sorted(a.seeds) != sorted(b.seeds):
        raise ValueError(f"{a.name} and {b.name} have different seeds: {a.seeds} and {b.seeds}")
    if a.test_isolates != b.test_isolates:
        raise ValueError(f"{a.name} and {b.name} have different test isolates (splits file or held-out groups)")
    if a.folds != b.folds:
        raise ValueError(f"{a.name} and {b.name} have different folds: {a.folds} and {b.folds}")
    for fold in a.folds:
        if a.membership.get(fold) is None or a.membership.get(fold) != b.membership.get(fold):
            raise ValueError(f"{a.name} and {b.name} validate different images in {fold} (first mismatched fold)")
    unmatched = sorted(set(a.f1) ^ set(b.f1), key=str)
    if unmatched:
        raise ValueError(f"{a.name} and {b.name} do not both score (seed, fold) {unmatched[:5]}")


def bootstrap_ci(differences: np.ndarray, level: float, n_boot: int, seed: int) -> list[float]:
    """Percentile CI of the mean difference, resampling the paired (seed, fold) units."""
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (n_boot, len(differences)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))]


def compare_pair(seq: Run, direct: Run, cfg: PairedConfig) -> dict:
    check_paired(seq, direct)
    keys = sorted(seq.f1, key=str)
    diffs = np.array([seq.f1[k] - direct.f1[k] for k in keys])
    mean = float(diffs.mean())
    ci = bootstrap_ci(diffs, cfg.ci_level, cfg.bootstrap, cfg.seed)
    p = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else float("nan")
    excludes_zero = ci[0] > 0
    superior = mean >= cfg.min_gain and (excludes_zero or not cfg.require_ci_excludes_zero)
    negative = mean < 0
    if superior:
        verdict = f"sequential superior: mean gain {100 * mean:.1f} points >= {100 * cfg.min_gain:.1f}" + (
            f", {cfg.ci_level:.0%} CI excludes 0" if cfg.require_ci_excludes_zero else "")
    elif negative:
        verdict = (f"negative transfer: sequential is {-100 * mean:.1f} points below direct"
                   + (f", {cfg.ci_level:.0%} CI excludes 0" if ci[1] < 0 else ", CI includes 0"))
    else:
        verdict = (f"sequential not superior: mean gain {100 * mean:.1f} points, {cfg.ci_level:.0%} CI "
                   f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]")
    return {"sequential": seq.name, "direct": direct.name, "arch": seq.arch, "metric": "isolate macro-F1",
            "n_pairs": len(keys), "mean_difference": mean, "ci": ci, "wilcoxon_p": p,
            "superior": superior, "negative_transfer": negative, "verdict": verdict,
            "recommendation": "use sequential" if superior else "use direct",
            "thresholds": {"min_gain": cfg.min_gain, "ci_level": cfg.ci_level,
                           "require_ci_excludes_zero": cfg.require_ci_excludes_zero},
            "differences": [{"seed": k[0], "fold": k[1], "sequential": seq.f1[k], "direct": direct.f1[k],
                             "difference": float(d)} for k, d in zip(keys, diffs)]}


def compare_runs(run_dirs: list, cfg: PairedConfig) -> list[dict]:
    """Every sequential run against every direct run of the same arch."""
    runs = [load_run(d) for d in run_dirs]
    pairs = [(s, d) for s in runs if s.sequential for d in runs if not d.sequential and d.arch == s.arch]
    if not pairs:
        roles = ", ".join(f"{r.name} ({'sequential' if r.sequential else 'direct'}, {r.arch})" for r in runs)
        raise ValueError(f"no sequential and direct run of the same arch to pair: {roles}")
    return [compare_pair(s, d, cfg) for s, d in pairs]


def write_comparison(pairs: list[dict], cfg: PairedConfig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": asdict(cfg), "pairs": pairs}, indent=2), encoding="utf-8")
    return out
