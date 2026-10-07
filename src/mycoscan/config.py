"""Run configuration. TOML in, frozen dataclass out; validated once here."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .models import PRETRAINED, resolve_weights

CHOICES = {
    "modality": {"colony", "microscopic", "all"},
    "source": {"cmu", "openfungi", "all"},
    "finetune": {"head", "partial", "full"},
    "imbalance": {"sampler", "none"},
    "loss": {"ce", "weighted_ce", "focal"},
    "augmentation": {"none", "standard", "trivial_wide"},
    "tracking": {"none", "mlflow"},
    "split": {"holdout", "kfold", "loio", "image_random"},
}
SUBGROUP_COLUMNS = {"modality", "view", "device", "day", "source", "genus", "temperature", "phase", "z_index"}


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
    loss: str = "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    augmentation: str = "standard"
    plate_crop: bool = False
    split: str = "kfold"
    splits_file: str = ""
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
    bootstrap: int = 2000
    genus_map: dict = field(default_factory=dict)
    order_map: dict = field(default_factory=dict)
    subgroups: tuple[str, ...] = ()
    amp: bool = False
    grad_clip: float = 0.0
    tracking: str = "none"

    def __post_init__(self) -> None:
        if self.imbalance == "loss":
            raise ValueError("config imbalance='loss' is now loss='weighted_ce' with imbalance='none'")
        for name, allowed in CHOICES.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"config {name}={value!r}; expected one of {sorted(allowed)}")
        if self.weights not in {*PRETRAINED, "none"} and not Path(self.weights).is_file():
            raise ValueError(f"config weights={self.weights!r} is none of {[*PRETRAINED, 'none']} nor an existing checkpoint")
        resolve_weights(self.arch, self.weights)
        if self.imbalance == "sampler" and self.loss == "weighted_ce":
            raise ValueError("config imbalance='sampler' and loss='weighted_ce' both correct class imbalance; "
                             "set imbalance='none' or loss='ce'")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("config label_smoothing must be in [0, 1)")
        if self.focal_gamma < 0:
            raise ValueError("config focal_gamma must be >= 0")
        if self.grad_clip < 0:
            raise ValueError("config grad_clip must be >= 0 (0 turns clipping off)")
        if self.splits_file and not Path(self.splits_file).is_file():
            raise ValueError(f"config splits_file={self.splits_file!r} does not exist")
        bad = sorted(set(self.subgroups) - SUBGROUP_COLUMNS)
        if bad:
            raise ValueError(f"config subgroups {bad} are not manifest columns; expected some of {sorted(SUBGROUP_COLUMNS)}")
        if not 0 < self.val_fraction < 1:
            raise ValueError("config val_fraction must be in (0, 1)")


def _coerce(raw: dict) -> dict:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key in ("classes", "subgroups"):
        if key in raw:
            raw = {**raw, key: tuple(raw[key])}
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
