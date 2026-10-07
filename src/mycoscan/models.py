"""Backbones, staged transfer, fine-tuning policy and checkpoints.

Staged transfer: ImageNet weights -> fine-tune on OpenFungi (5 genera) ->
fine-tune on CMU isolates. A later stage loads the earlier checkpoint's
backbone and replaces the classification head, since the class sets differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torchvision import models as tvm


@dataclass(frozen=True)
class Arch:
    build: Callable[[bool], nn.Module]
    head: str
    last_block: tuple[str, ...]
    cam_layer: str


ARCHS: dict[str, Arch] = {
    "densenet121": Arch(
        build=lambda imagenet: tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if imagenet else None),
        head="classifier",
        last_block=("features.denseblock4", "features.norm5"),
        cam_layer="features.denseblock4",
    ),
    "resnet50": Arch(
        build=lambda imagenet: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if imagenet else None),
        head="fc",
        last_block=("layer4",),
        cam_layer="layer4",
    ),
}


def _replace_head(model: nn.Module, arch: Arch, n_classes: int) -> None:
    in_features = getattr(model, arch.head).in_features
    setattr(model, arch.head, nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, n_classes)))


def build_model(arch_name: str, n_classes: int, weights: str) -> nn.Module:
    """weights: 'imagenet', 'none', or a path to an earlier-stage checkpoint (backbone is reused)."""
    arch = ARCHS[arch_name]
    model = arch.build(weights == "imagenet")
    _replace_head(model, arch, n_classes)
    if weights not in {"imagenet", "none"}:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if ckpt["arch"] != arch_name:
            raise ValueError(f"checkpoint {weights} is {ckpt['arch']}, config asks for {arch_name}")
        backbone = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(arch.head + ".")}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        if unexpected or any(not k.startswith(arch.head + ".") for k in missing):
            raise ValueError(f"checkpoint {weights} does not match {arch_name}: missing={missing[:3]} unexpected={unexpected[:3]}")
    return model


def apply_finetune(model: nn.Module, arch_name: str, mode: str) -> None:
    """head: train only the new FC head (spec 2.2). partial: also the last block. full: everything."""
    arch = ARCHS[arch_name]
    trainable = {"head": (arch.head,), "partial": (arch.head, *arch.last_block), "full": ("",)}[mode]
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable)


def train_mode(model: nn.Module) -> None:
    """model.train(), but frozen BatchNorm layers keep their ImageNet/OpenFungi running statistics."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def save_checkpoint(path: Path, model: nn.Module, arch_name: str, classes: list[str], image_size: int,
                    autocontrast: bool, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch_name, "classes": classes,
                "image_size": image_size, "autocontrast": autocontrast, **extra}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), "none")
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
