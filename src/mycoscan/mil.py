"""Attention-MIL: the bag is the training unit.

With `pooling = "gated_attention"` or `"mh_attention"` a shared encoder (the run's backbone,
its head unused) embeds every instance of a bag, an attention module weighs the instances and
pools their embeddings, and a linear head classifies the pooled embedding. The bag-level loss
trains all three. Gated attention is that of Ilse et al. (2018): a score w^T (tanh(V h) * sigmoid(U h))
per instance, softmaxed over the bag's real instances. The multi-head variant gives `heads` score
vectors over one shared gate, pools once per head and concatenates the pooled embeddings.

`pooling_hierarchy = "device_then_isolate"` composes two pooling calls inside an isolate bag: the
instances of each device are pooled into a device embedding, classified into a device prediction,
and a single-head attention over the device embeddings weighs the device predictions into the
isolate's. The isolate's probabilities are therefore exactly the attention-weighted mean of its
device probabilities, and one model reports both.

Padded instances (and empty device slots) get zero attention, so padding contributes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .bags import BagDataset, collate_bags
from .data import balanced_sample_weights, refuse_held_out
from .transforms import TileSpec

ATTENTION_POOLINGS = ("gated_attention", "mh_attention")
HIERARCHIES = ("none", "device_then_isolate")


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """scores (bags, instances, heads) -> weights over each bag's real instances; masked ones exactly 0,
    and a bag with no real instance all 0."""
    real = mask.unsqueeze(-1)
    weights = scores.masked_fill(~real, -1e9).softmax(dim=1) * real
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


class AttentionPool(nn.Module):
    """Gated attention pooling with `heads` score vectors: (bags, instances, dim) and a mask ->
    pooled (bags, heads * dim) and weights (bags, instances, heads)."""

    def __init__(self, dim: int, hidden: int, heads: int = 1):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, heads)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = masked_softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), mask)
        pooled = torch.einsum("bnk,bnd->bkd", weights, torch.where(mask.unsqueeze(-1), h, 0))
        return pooled.flatten(1), weights


@dataclass
class MILOutput:
    logits: torch.Tensor  # (bags, classes); with a hierarchy the log of the pooled device probabilities
    weights: torch.Tensor  # (bags, instances, heads); with a hierarchy summing to 1 within each device
    device_logits: torch.Tensor | None = None  # (bags, devices, classes)
    device_weights: torch.Tensor | None = None  # (bags, devices)
    device_mask: torch.Tensor | None = None  # (bags, devices): the device slots the bag has


class MILModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int, pooling: str, heads: int, hidden: int,
                 hierarchy: str = "none"):
        super().__init__()
        if pooling not in ATTENTION_POOLINGS:
            raise ValueError(f"pooling {pooling!r}; attention-MIL takes one of {list(ATTENTION_POOLINGS)}")
        heads = heads if pooling == "mh_attention" else 1
        dim = int(getattr(encoder, "num_features"))
        self.encoder = encoder
        self.pool = AttentionPool(dim, hidden, heads)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(heads * dim, n_classes))
        self.device_pool = AttentionPool(heads * dim, hidden, 1) if hierarchy == "device_then_isolate" else None

    @property
    def hierarchical(self) -> bool:
        return self.device_pool is not None

    def embed(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(bags, instances, C, H, W) -> (bags, instances, dim), zero at padding; padding is never encoded."""
        forward_features, forward_head = getattr(self.encoder, "forward_features"), getattr(self.encoder, "forward_head")
        features = forward_head(forward_features(x[mask]), pre_logits=True)
        h = features.new_zeros((*mask.shape, features.shape[-1]))
        h[mask] = features
        return h

    def attend_embeddings(self, h: torch.Tensor, mask: torch.Tensor, devices: torch.Tensor | None = None) -> MILOutput:
        if self.device_pool is None:
            pooled, weights = self.pool(h, mask)
            return MILOutput(self.head(pooled), weights)
        if devices is None:
            raise ValueError("hierarchical pooling needs each instance's device index")
        n_devices = max(int(devices.max()) + 1, 1)
        bags, n = mask.shape
        in_device = torch.stack([mask & (devices == d) for d in range(n_devices)], dim=1)  # (bags, devices, n)
        pooled, weights = self.pool(h.unsqueeze(1).expand(-1, n_devices, -1, -1).reshape(bags * n_devices, n, -1),
                                    in_device.reshape(bags * n_devices, n))
        device_embeddings = pooled.reshape(bags, n_devices, -1)
        device_logits = self.head(device_embeddings)
        device_mask = in_device.any(dim=-1)
        device_weights = self.device_pool(device_embeddings, device_mask)[1][..., 0]
        probs = (device_weights.unsqueeze(-1) * device_logits.softmax(dim=-1)).sum(dim=1)
        # each instance is in one device, so summing over devices gives its weight within its device
        instance_weights = weights.reshape(bags, n_devices, n, -1).sum(dim=1)
        return MILOutput(probs.clamp_min(1e-12).log(), instance_weights, device_logits, device_weights, device_mask)

    def attend(self, x: torch.Tensor, mask: torch.Tensor, devices: torch.Tensor | None = None) -> MILOutput:
        return self.attend_embeddings(self.embed(x, mask), mask, devices)

    def instance_logits(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Each real instance scored as a bag of one, in mask order."""
        flat = h[mask].unsqueeze(1)
        ones = torch.ones(flat.shape[:2], dtype=torch.bool, device=flat.device)
        return self.attend_embeddings(flat, ones, torch.zeros_like(ones, dtype=torch.long)).logits

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                devices: torch.Tensor | None = None) -> torch.Tensor:
        """Bag logits; a plain image batch (images, C, H, W) is scored image by image, each a bag of one."""
        if mask is None:
            ones = torch.ones(len(x), 1, dtype=torch.bool, device=x.device)
            return self.instance_logits(self.embed(x.unsqueeze(1), ones), ones)
        return self.attend(x, mask, devices).logits


def device_index(rows: pd.DataFrame, metas: list[dict], shape: torch.Size) -> torch.Tensor:
    """Each instance's device as an index within its bag (bags, instances), -1 at padding."""
    index = torch.full(tuple(shape), -1, dtype=torch.long)
    for b, meta in enumerate(metas):
        codes = pd.factorize(rows["device"].astype(str).iloc[meta["rows"]])[0]
        index[b, :len(codes)] = torch.from_numpy(codes)
    return index


def bags_per_batch(dataset: BagDataset, batch_size: int) -> int:
    """As many bags as keep a batch near `batch_size` instances, at least one."""
    return max(1, batch_size // max(len(positions) for _, positions in dataset.bags))


def training_batches(rows: pd.DataFrame, class_to_idx: dict[str, int], transform, batch_size: int, imbalance: str,
                     seed: int, hierarchical: bool, plate_crop: bool = False, tiles: TileSpec | None = None,
                     num_workers: int = 0):
    """An epoch iterator of ((instances, mask, devices or None), labels) over the bags of `rows`, and every
    bag's label. With imbalance = "sampler" bags are drawn so every class, and every group within a class,
    is drawn equally often."""
    refuse_held_out(rows)
    dataset = BagDataset(rows, class_to_idx, transform, plate_crop, tiles)
    per_batch = bags_per_batch(dataset, batch_size)
    first = rows.groupby("bag", sort=False).first()
    generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset) > per_batch
    if imbalance == "sampler":
        sampler = WeightedRandomSampler(balanced_sample_weights(first["species"], first["group"]).tolist(), len(dataset),
                                        replacement=True, generator=generator)
        loader = DataLoader(dataset, per_batch, sampler=sampler, collate_fn=collate_bags, num_workers=num_workers,
                            drop_last=drop_last)
    else:
        loader = DataLoader(dataset, per_batch, shuffle=True, generator=generator, collate_fn=collate_bags,
                            num_workers=num_workers, drop_last=drop_last)

    def epoch():
        for x, mask, y, metas in loader:
            yield (x, mask, device_index(rows, metas, mask.shape) if hierarchical else None), y

    return epoch, torch.tensor(first["species"].map(class_to_idx).to_numpy())


@dataclass
class MILPredictions:
    instance_probs: np.ndarray  # one row per row of `rows`
    instance_weights: np.ndarray  # (rows, heads)
    names: list[str]  # bag names
    bag_probs: np.ndarray
    device_names: list[str] | None = None  # "<bag>|<device>", with a hierarchy
    device_parents: list[str] | None = None  # the bag each device belongs to
    device_probs: np.ndarray | None = None
    device_weights: np.ndarray | None = None


@torch.no_grad()
def predict_mil(model: MILModel, rows: pd.DataFrame, transform, device: str, batch_size: int,
                plate_crop: bool = False, tiles: TileSpec | None = None) -> MILPredictions:
    dataset = BagDataset(rows, {}, transform, plate_crop, tiles)
    loader = DataLoader(dataset, bags_per_batch(dataset, batch_size), collate_fn=collate_bags)
    model.eval()
    devices_of = rows["device"].astype(str)
    instance_probs: np.ndarray | None = None
    instance_weights: np.ndarray | None = None
    names, bag_probs, device_names, device_parents, device_probs, device_weights = [], [], [], [], [], []
    for x, mask, _, metas in loader:
        devices = device_index(rows, metas, mask.shape) if model.hierarchical else None
        h = model.embed(x.to(device), mask.to(device))
        out = model.attend_embeddings(h, mask.to(device), devices.to(device) if devices is not None else None)
        alone = model.instance_logits(h, mask.to(device)).softmax(dim=-1).cpu()
        if instance_probs is None or instance_weights is None:
            instance_probs = np.zeros((len(rows), alone.shape[1]), dtype=np.float32)
            instance_weights = np.zeros((len(rows), out.weights.shape[-1]), dtype=np.float32)
        instance_probs[np.concatenate([m["rows"] for m in metas])] = alone.numpy()
        weights, probs = out.weights.cpu(), out.logits.softmax(dim=-1).cpu()
        for b, meta in enumerate(metas):
            instance_weights[meta["rows"]] = weights[b, :len(meta["rows"])].numpy()
            names.append(meta["bag"])
            if out.device_logits is not None and out.device_weights is not None and out.device_mask is not None:
                for d, name in enumerate(pd.unique(devices_of.iloc[meta["rows"]])):
                    device_names.append(f"{meta['bag']}|{name}")
                    device_parents.append(meta["bag"])
                    device_probs.append(out.device_logits[b, d].softmax(dim=-1).cpu().numpy())
                    device_weights.append(float(out.device_weights[b, d]))
        bag_probs.append(probs.numpy())
    assert instance_probs is not None and instance_weights is not None
    preds = MILPredictions(instance_probs, instance_weights, names, np.concatenate(bag_probs))
    if model.hierarchical:
        preds.device_names, preds.device_parents = device_names, device_parents
        preds.device_probs, preds.device_weights = np.stack(device_probs), np.array(device_weights)
    return preds


def weight_columns(heads: int) -> list[str]:
    return ["attention"] if heads == 1 else [f"attention_h{k}" for k in range(heads)]


def attention_sidecars(rows: pd.DataFrame, preds: MILPredictions) -> dict[str, pd.DataFrame]:
    """`attention`: one row per instance, keyed by its bag (with a hierarchy, its device bag) and the instance
    (its image, and tile), with one weight column per head. `attention_devices`, with a hierarchy: one row per
    device bag, keyed by its isolate bag and the device bag, with the device's weight."""
    bag = rows["bag"].astype(str)
    if preds.device_names is not None:
        bag = bag + "|" + rows["device"].astype(str)
    keys = {"bag": bag.to_numpy(), "instance": rows["image_path"].to_numpy(),
            **({"tile": rows["tile"].to_numpy()} if "tile" in rows else {})}
    weights = dict(zip(weight_columns(preds.instance_weights.shape[1]), preds.instance_weights.T))
    sidecars = {"attention": pd.DataFrame({**keys, **weights})}
    if preds.device_names is not None:
        sidecars["attention_devices"] = pd.DataFrame({"bag": preds.device_parents, "instance": preds.device_names,
                                                      "attention": preds.device_weights})
    return sidecars


def _mean_entropy(frame: pd.DataFrame) -> tuple[float, int]:
    """Mean over bags (and heads) of the entropy, in nats, of each bag's attention weights."""
    columns = [c for c in frame if c.startswith("attention")]
    keys = [k for k in ("seed", "fold", "bag") if k in frame]
    w = frame[columns].to_numpy(dtype=float)
    terms = pd.DataFrame(np.where(w > 0, -w * np.log(np.where(w > 0, w, 1)), 0.0), columns=columns)
    per_bag = terms.groupby([frame[k].to_numpy() for k in keys]).sum()
    return float(per_bag.to_numpy().mean()), len(per_bag)


def attention_summary(sidecars: dict[str, pd.DataFrame]) -> dict:
    """Mean attention entropy over bags (0: one instance carries the bag; log n: uniform over n instances)."""
    entropy, n_bags = _mean_entropy(sidecars["attention"])
    summary = {"mean_entropy": entropy, "n_bags": n_bags, "unit": "nats"}
    if "attention_devices" in sidecars:
        summary["device_mean_entropy"], summary["n_isolates"] = _mean_entropy(sidecars["attention_devices"])
    return summary
