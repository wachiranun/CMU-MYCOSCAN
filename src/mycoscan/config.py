"""Run configuration. TOML in, frozen dataclass out; validated once here."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .models import PRETRAINED, resolve_weights

CHOICES = {
    "modality": {"colony", "microscopic", "all"},
    "source": {"cmu", "openfungi", "all"},
    "finetune": {"head", "partial", "full"},
    "imbalance": {"sampler", "loss", "none"},
    "split": {"holdout", "kfold", "loio"},
}


@dataclass(frozen=True)
class Config:
    run_name: str
    manifest: str
    output_dir: str = "runs"
    classes: tuple[str, ...] = ()
    modality: str = "microscopic"
    source: str = "cmu"
    arch: str = "densenet121"
    weights: str = "imagenet"
    finetune: str = "head"
    image_size: int = 224
    autocontrast: bool = True
    imbalance: str = "sampler"
    split: str = "kfold"
    n_folds: int = 5
    val_fraction: float = 0.2
    fit_final: bool = True
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    bootstrap: int = 1000
    amp: bool = False
    grad_clip: float = 0.0

    def __post_init__(self) -> None:
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"config {name}={value!r}; expected one of {sorted(allowed)}")
        if self.weights not in {*PRETRAINED, "none"} and not Path(self.weights).is_file():
            raise ValueError(f"config weights={self.weights!r} is none of {[*PRETRAINED, 'none']} nor an existing checkpoint")
        resolve_weights(self.arch, self.weights)
        if self.grad_clip < 0:
            raise ValueError("config grad_clip must be >= 0 (0 turns clipping off)")
        if not 0 < self.val_fraction < 1:
            raise ValueError("config val_fraction must be in (0, 1)")


def _coerce(raw: dict) -> dict:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "classes" in raw:
        raw = {**raw, "classes": tuple(raw["classes"])}
    return raw


def parse_override(text: str) -> tuple[str, object]:
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"override {text!r} must look like key=value")
    try:
        parsed = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value
    return key.strip(), parsed


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw.update(dict(parse_override(o) for o in overrides))
    return Config(**_coerce(raw))
