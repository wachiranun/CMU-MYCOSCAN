"""Run configuration. TOML in, frozen dataclass out; validated once here."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .bags import BAG_MODES, POOLINGS
from .metrics import TAU_RULES
from .mil import ATTENTION_POOLINGS, HIERARCHIES
from .models import PRETRAINED, resolve_weights

CHOICES = {
    "modality": {"colony", "microscopic", "all"},
    "source": {"cmu", "openfungi", "all"},
    "finetune": {"head", "partial", "full", "lora", "linear_probe"},
    "imbalance": {"sampler", "none"},
    "loss": {"ce", "weighted_ce", "focal"},
    "augmentation": {"none", "standard", "trivial_wide"},
    "tracking": {"none", "mlflow"},
    "split": {"holdout", "kfold", "loio", "image_random"},
    "tau_rule": set(TAU_RULES),
    "bag": set(BAG_MODES),
    "pooling": {*POOLINGS, *ATTENTION_POOLINGS},
    "pooling_hierarchy": set(HIERARCHIES),
}
SUBGROUP_COLUMNS = {"modality", "view", "device", "day", "source", "genus", "temperature", "phase", "z_index"}


@dataclass(frozen=True)
class Config:
    run_name: str
    manifest: str
    output_dir: str = "runs"
    classes: tuple[str, ...] = ()
    select_classes: tuple[str, ...] = ()
    modality: str = "microscopic"
    source: str = "cmu"
    arch: str = "densenet121"
    weights: str = "imagenet"
    finetune: str = "head"
    partial_blocks: int = 1
    lora_rank: int = 8
    layer_decay: float = 0.8
    ema: bool = False
    ema_decay: float = 0.999
    image_size: int = 224
    autocontrast: bool = True
    imbalance: str = "sampler"
    loss: str = "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    augmentation: str = "standard"
    plate_crop: bool = False
    bag: str = "none"
    pooling: str = "mean"
    pooling_hierarchy: str = "none"
    attention_heads: int = 4
    attention_dim: int = 128
    tile_grid: tuple[int, ...] = (3, 2)
    tile_size: int = 640
    split: str = "kfold"
    splits_file: str = ""
    n_folds: int = 5
    val_fraction: float = 0.2
    train_fraction: float = 1.0
    fit_final: bool = True
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    seeds: tuple[int, ...] = ()
    bootstrap: int = 2000
    genus_map: dict = field(default_factory=dict)
    order_map: dict = field(default_factory=dict)
    subgroups: tuple[str, ...] = ()
    calibration_bins: int = 10
    tau_rule: str = "min_accuracy"
    tau_target: float = 0.9
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
        if len(self.tile_grid) != 2 or min(self.tile_grid) < 1 or self.tile_size < 1:
            raise ValueError("config tile_grid must be [columns, rows] of positive integers and tile_size positive")
        if self.bag != "none" and self.split == "image_random":
            raise ValueError("config split='image_random' scatters a group's images over folds, so bags "
                             "(always inside one group) would straddle them; use bag='none'")
        if self.bag != "none" and self.finetune == "linear_probe":
            raise ValueError("config bag pools a network's instance predictions; finetune='linear_probe' "
                             "scores isolates by the mean of its images, so use bag='none'")
        if self.pooling in ATTENTION_POOLINGS and self.bag == "none":
            raise ValueError(f"config pooling={self.pooling!r} trains on bags; set bag to one of "
                             f"{[b for b in BAG_MODES if b != 'none']}")
        if self.pooling_hierarchy != "none" and (self.pooling not in ATTENTION_POOLINGS or self.bag != "isolate"):
            raise ValueError("config pooling_hierarchy='device_then_isolate' pools an isolate bag's devices with "
                             f"attention, so it needs bag='isolate' and pooling in {list(ATTENTION_POOLINGS)}; "
                             "for mean or max, bag='isolate_device' already pools per device, then across devices")
        if self.attention_heads < 1 or self.attention_dim < 1:
            raise ValueError("config attention_heads and attention_dim must be >= 1")
        if self.calibration_bins < 1:
            raise ValueError("config calibration_bins must be >= 1")
        if not 0 < self.tau_target <= 1:
            raise ValueError("config tau_target must be in (0, 1]")
        if not 0 < self.val_fraction < 1:
            raise ValueError("config val_fraction must be in (0, 1)")
        if not 0 < self.train_fraction <= 1:
            raise ValueError("config train_fraction must be in (0, 1]")
        if self.partial_blocks < 1:
            raise ValueError("config partial_blocks must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("config lora_rank must be >= 1")
        if not 0 < self.layer_decay <= 1:
            raise ValueError("config layer_decay must be in (0, 1]; 1 turns layer-wise decay off")
        if not 0 < self.ema_decay < 1:
            raise ValueError("config ema_decay must be in (0, 1)")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"config seeds {list(self.seeds)} repeat a seed")
        if self.finetune == "linear_probe" and len(self.seeds) > 1:
            raise ValueError("config finetune='linear_probe' is deterministic on fixed features; "
                             "more than one of seeds would repeat one result")


def _coerce(raw: dict) -> dict:
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for key in ("classes", "select_classes", "subgroups", "seeds", "tile_grid"):
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


def config_with(path: str | Path, values: dict) -> Config:
    """The config at `path` with `values` replacing its keys."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config(**_coerce({**raw, **values}))


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    return config_with(path, dict(parse_override(o) for o in overrides))
