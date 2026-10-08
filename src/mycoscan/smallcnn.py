"""`small_cnn`: a small CNN trained from scratch, for the P0 reproduction of the OpenFungi paper.

Four blocks of 3x3 convolution, batch norm, ReLU and 2x2 max pooling (32, 64, 128
and 256 channels), global average pooling and a linear head, in the spirit of the
paper's Keras CNN at 128 px; matching that network layer for layer is a pilot-report
note, not a goal. It is registered with timm, so `arch = "small_cnn"` goes through the
same registry and adapter as every other backbone. It has no pretrained weights:
`weights = "none"`.
"""
from __future__ import annotations

import torch
from timm.models import register_model
from torch import nn

CHANNELS = (32, 64, 128, 256)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 1000, in_chans: int = 3):
        super().__init__()
        stages, width = [], in_chans
        for out in CHANNELS:
            stages.append(nn.Sequential(nn.Conv2d(width, out, 3, padding=1, bias=False), nn.BatchNorm2d(out),
                                        nn.ReLU(inplace=True), nn.MaxPool2d(2)))
            width = out
        self.stages = nn.Sequential(*stages)
        self.num_features = width
        self.head = nn.Linear(width, num_classes)
        # What the timm adapter reads: the head's name and each block's output tap.
        self.pretrained_cfg = {"classifier": "head"}
        self.feature_info = [{"module": f"stages.{i}", "num_chs": c, "reduction": 2 ** (i + 1)}
                             for i, c in enumerate(CHANNELS)]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        return pooled if pre_logits else self.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


@register_model
def small_cnn(pretrained: bool = False, num_classes: int = 1000, in_chans: int = 3, **_: object) -> SmallCNN:
    """timm passes pretrained_cfg and friends, and build_model an img_size; a CNN needs none of them."""
    if pretrained:
        raise ValueError("small_cnn has no pretrained weights; set weights = \"none\"")
    return SmallCNN(num_classes, in_chans)
