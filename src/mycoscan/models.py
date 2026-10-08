"""Backbones, staged transfer, fine-tuning policy and checkpoints.

Any timm model name is a valid `arch`. A small adapter reads what the rest of
the code needs from the timm model itself (head, blocks, explain layer), so a
new backbone needs no registry entry.

Staged transfer: ImageNet weights -> fine-tune on OpenFungi (5 genera) ->
fine-tune on CMU isolates. A later stage loads the earlier checkpoint's
backbone and replaces the classification head, since the class sets differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import timm
import torch
from torch import nn

from . import smallcnn  # noqa: F401  registers arch "small_cnn" with timm

PRETRAINED = ("imagenet", "imagenet22k", "dino")
# The two archs of the torchvision era keep the exact ImageNet weights they were trained from.
TORCHVISION_TAGS = {"densenet121": "tv_in1k", "resnet50": "tv2_in1k"}


@dataclass(frozen=True)
class Adapter:
    head: str
    blocks: tuple[str, ...]  # timm feature_info taps, input to output; a block ends at its tap
    cam_layer: str


def adapter(model: nn.Module) -> Adapter:
    """`model` is a timm model: it carries `feature_info` and `pretrained_cfg`."""
    blocks = tuple(f["module"] for f in getattr(model, "feature_info"))
    return Adapter(head=getattr(model, "pretrained_cfg")["classifier"], blocks=blocks, cam_layer=blocks[-1])


LARGER_THAN_1K = ("in12k", "in21k", "in22k", "_ft_")


def _tag_matches(weights: str, arch: str, tag: str) -> bool:
    if weights == "imagenet":
        return "in1k" in tag and not any(marker in tag for marker in LARGER_THAN_1K)
    if weights == "imagenet22k":
        return "in22k" in tag or "in21k" in tag
    return "dino" in tag or "dino" in arch


def resolve_weights(arch: str, weights: str) -> str:
    """The timm `name.tag` that `weights` (imagenet | imagenet22k | dino) means for `arch`;
    `arch` itself for any other weights.

    An arch that already names a tag keeps it. Otherwise imagenet takes a tag trained on ImageNet-1k
    alone (timm's default if it is one), never one pretrained on a larger set and fine-tuned on 1k,
    so imagenet and imagenet22k stay distinct; imagenet22k prefers the tag not fine-tuned on 1k;
    dino takes a DINO tag or a DINO arch.
    """
    if not timm.is_model(arch):
        raise ValueError(f"config arch={arch!r} is not a timm model name")
    base, _, tag = arch.partition(".")
    if tag or weights not in PRETRAINED:
        return arch
    if weights == "imagenet" and base in TORCHVISION_TAGS:
        return f"{base}.{TORCHVISION_TAGS[base]}"
    tags = [t.partition(".")[2] for t in timm.list_pretrained(base + ".*")]
    default_cfg = timm.models.get_pretrained_cfg(base)
    default = default_cfg.tag if default_cfg else None
    ranked = sorted(tags, key=lambda t: (t != default if weights != "imagenet22k" else "_ft_" in t, t))
    for t in ranked:
        if _tag_matches(weights, base, t):
            return f"{base}.{t}"
    raise ValueError(f"arch {base!r} has no {weights} weights; its pretrained tags are {tags}")


def _create(name: str, pretrained: bool, n_classes: int, image_size: int) -> nn.Module:
    # Transformers bake the input size into their position embeddings; CNNs reject the argument.
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=n_classes, img_size=image_size)
    except TypeError as e:
        if "img_size" not in str(e):
            raise
        return timm.create_model(name, pretrained=pretrained, num_classes=n_classes)


def is_checkpoint(weights: str) -> bool:
    """True when `weights` names an earlier stage's checkpoint rather than an initialisation."""
    return weights not in (*PRETRAINED, "none")


def build_model(arch: str, n_classes: int, weights: str, image_size: int = 224) -> nn.Module:
    """weights: imagenet, imagenet22k, dino, none, or a path to an earlier-stage checkpoint (backbone is reused)."""
    pretrained = weights in PRETRAINED
    model = _create(resolve_weights(arch, weights), pretrained, n_classes, image_size)
    head = adapter(model).head
    in_features = cast(nn.Linear, model.get_submodule(head)).in_features
    model.set_submodule(head, nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, n_classes)))
    if is_checkpoint(weights):
        load_backbone(model, weights, arch)
    return model


def load_backbone(model: nn.Module, checkpoint: str | Path, arch: str) -> dict:
    """Load an earlier stage's backbone into `model`, its head left newly initialised. Refuses a checkpoint of
    another arch, and one whose loading would leave anything but head parameters new. Returns the parameters
    newly initialised and the earlier stage's provenance (None for a checkpoint that carries none)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt["arch"] != arch:
        raise ValueError(f"checkpoint {checkpoint} is {ckpt['arch']}, config asks for {arch}")
    head = adapter(model).head
    backbone = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(head + ".")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    if unexpected or any(not k.startswith(head + ".") for k in missing):
        raise ValueError(f"checkpoint {checkpoint} does not match {arch}: missing={missing[:3]} unexpected={unexpected[:3]}")
    return {"newly_initialised": list(missing), "provenance": ckpt.get("provenance")}


def _block_ends(a: Adapter, names: list[str]) -> list[int]:
    """Index in `names` (the model's modules in order) of the last module of each block."""
    return [max(i for i, n in enumerate(names) if n == tap or n.startswith(tap + ".")) for tap in a.blocks]


def lr_scales(model: nn.Module, layer_decay: float) -> dict[str, float]:
    """Learning-rate multiplier per parameter: 1 for the head and anything after the last block,
    layer_decay for the last block, layer_decay**2 for the one before, and so on, and one more step for
    the stem: everything before the first block, such as a ViT's patch and position embeddings.
    A tap without parameters (ResNet's stem activation) is not a block of its own."""
    names = [n for n, _ in model.named_modules()]
    index = {n: i for i, n in enumerate(names)}
    a = adapter(model)
    taps = tuple(t for t in a.blocks if any(True for _ in model.get_submodule(t).parameters()))
    starts = [index[t] for t in taps]
    ends = _block_ends(Adapter(a.head, taps, a.cam_layer), names)
    scales = {}
    for name, _ in model.named_parameters():
        i = index[name.rpartition(".")[0]]
        depth = len(taps) + 1 if i < starts[0] else len(taps) - next((k for k, end in enumerate(ends) if i <= end),
                                                                       len(taps))
        scales[name] = layer_decay ** depth
    return scales


def apply_finetune(model: nn.Module, mode: str, blocks: int = 1) -> None:
    """head: train only the new head (spec 2.2). partial: also the last `blocks` blocks and
    everything after them (final norms). full: everything."""
    a = adapter(model)
    if mode == "full" or blocks >= len(a.blocks):
        for param in model.parameters():
            param.requires_grad = True
        return
    names = [n for n, _ in model.named_modules()]
    if mode == "partial":
        end = _block_ends(a, names)[-blocks - 1]
        trainable = set(names[end + 1:])
    else:
        trainable = {n for n in names if n == a.head or n.startswith(a.head + ".")}
    for name, param in model.named_parameters():
        param.requires_grad = name.rpartition(".")[0] in trainable


def require_peft() -> ModuleType:
    try:
        import peft
    except ImportError as e:
        raise ImportError('finetune = "lora" needs the optional extra: pip install "mycoscan[lora]"') from e
    return peft


def add_lora(model: nn.Module, rank: int) -> None:
    """LoRA adapters of `rank` on every Linear layer outside the head; only the adapters and the head train."""
    peft = require_peft()
    head = adapter(model).head
    targets = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n != head and not n.startswith(head + ".")]
    if not targets:
        raise ValueError(f"finetune='lora' adapts Linear layers, and {type(model).__name__} has none outside its head")
    peft.inject_adapter_in_model(peft.LoraConfig(r=rank, lora_alpha=rank, target_modules=targets), model)
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name or name.startswith(head + ".")


def merge_lora(model: nn.Module) -> None:
    """Fold each adapter into the weight it adapts, leaving a plain model that loads and predicts without peft."""
    from peft.tuners.lora import LoraLayer

    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLayer):
            module.merge()
            model.set_submodule(name, module.get_base_layer())


def train_mode(model: nn.Module) -> None:
    """model.train(), but frozen BatchNorm layers keep their pretrained running statistics.
    LayerNorm has no running statistics, so freezing its parameters is enough."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def save_checkpoint(path: Path, model: nn.Module, arch_name: str, classes: list[str], image_size: int,
                    autocontrast: bool, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch_name, "classes": classes,
                "image_size": image_size, "autocontrast": autocontrast, **extra}, path)


def add_to_checkpoint(path: Path, **fields) -> None:
    """Record fields known only after a checkpoint was saved (the reject threshold) in it."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    torch.save({**ckpt, **fields}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """The saved model and its metadata; an attention-MIL checkpoint comes back as its MILModel."""
    from .mil import ATTENTION_POOLINGS, MILModel

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), "none", ckpt["image_size"])
    if ckpt.get("pooling") in ATTENTION_POOLINGS:
        model = MILModel(model, len(ckpt["classes"]), ckpt["pooling"], ckpt["attention_heads"],
                         ckpt["attention_dim"], ckpt.get("pooling_hierarchy", "none"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
