"""Synthetic stand-in data with the same manifest layout as the real CMU and OpenFungi sets.

It exists only to prove the pipeline end to end. Images are procedurally drawn
colonies and hyphae; any metric computed on them says nothing about real fungi.
Each class has its own colour, ring, hypha and spore parameters, and each isolate
jitters them, so isolate leakage would inflate scores the same way it would on real data.
OpenFungi images come in near-duplicate pairs sharing a group_id, mimicking repeated
shots of one plate, so pseudo-group leakage can be tested the same way.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

PLACEHOLDER_CMU_CLASSES = (
    "Aspergillus_fumigatus", "Aspergillus_flavus", "Aspergillus_niger", "Aspergillus_terreus",
    "Fusarium_spp", "Mucorales", "Talaromyces_marneffei", "Sporothrix_schenckii_complex",
    "Penicillium_spp", "Scedosporium_spp",
)
GENUS_OF = {
    "Aspergillus_fumigatus": "Aspergillus", "Aspergillus_flavus": "Aspergillus", "Aspergillus_niger": "Aspergillus",
    "Aspergillus_terreus": "Aspergillus", "Fusarium_spp": "Fusarium", "Mucorales": "Mucorales",
    "Talaromyces_marneffei": "Talaromyces", "Sporothrix_schenckii_complex": "Sporothrix",
    "Penicillium_spp": "Penicillium", "Scedosporium_spp": "Scedosporium",
}
DIMORPHIC = ("Talaromyces_marneffei", "Sporothrix_schenckii_complex")
OPENFUNGI_GENERA = ("Aspergillus", "Penicillium", "Rhizopus", "Alternaria", "Fusarium")
DAYS = (3, 7, 14)


@dataclass(frozen=True)
class Morphology:
    obverse: np.ndarray
    reverse: np.ndarray
    rings: int
    hypha_width: int
    branch_angle: float
    spore_radius: int
    spores_per_head: int


def _morphology(rng: np.random.Generator) -> Morphology:
    return Morphology(
        obverse=rng.integers(30, 230, 3).astype(float),
        reverse=rng.integers(30, 230, 3).astype(float),
        rings=int(rng.integers(0, 5)),
        hypha_width=int(rng.integers(1, 4)),
        branch_angle=float(rng.uniform(20, 90)),
        spore_radius=int(rng.integers(1, 5)),
        spores_per_head=int(rng.integers(3, 25)),
    )


def _jitter(m: Morphology, rng: np.random.Generator) -> Morphology:
    return Morphology(
        obverse=np.clip(m.obverse + rng.normal(0, 12, 3), 0, 255),
        reverse=np.clip(m.reverse + rng.normal(0, 12, 3), 0, 255),
        rings=m.rings,
        hypha_width=m.hypha_width,
        branch_angle=m.branch_angle + float(rng.normal(0, 5)),
        spore_radius=m.spore_radius,
        spores_per_head=max(2, m.spores_per_head + int(rng.integers(-2, 3))),
    )


def _noise(size: int, base, rng: np.random.Generator, sd: float) -> Image.Image:
    arr = np.clip(np.asarray(base, float) + rng.normal(0, sd, (size, size, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_colony(m: Morphology, view: str, day: int, size: int, rng: np.random.Generator) -> Image.Image:
    img = _noise(size, (225, 215, 185), rng, 6)
    draw = ImageDraw.Draw(img)
    color = m.obverse if view == "obverse" else m.reverse
    radius = size * (0.12 + 0.025 * day) * rng.uniform(0.9, 1.1)
    cx, cy = size / 2 + rng.normal(0, size * 0.03, 2)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=tuple(int(c) for c in color))
    for k in range(1, m.rings + 1):
        r = radius * k / (m.rings + 1)
        shade = tuple(int(c * 0.7) for c in color)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=shade, width=2)
    speckle = np.asarray(img, float) + rng.normal(0, 8, (size, size, 3))
    return Image.fromarray(np.clip(speckle, 0, 255).astype(np.uint8))


def draw_micro(m: Morphology, device: str, z: int, size: int, rng: np.random.Generator) -> Image.Image:
    img = _noise(size, (205, 215, 235), rng, 5)
    draw = ImageDraw.Draw(img)
    ink = (40, 60, 140)
    for _ in range(int(rng.integers(3, 6))):
        x0, y0 = rng.uniform(0, size, 2)
        angle = rng.uniform(0, 2 * math.pi)
        length = size * rng.uniform(0.3, 0.6)
        x1, y1 = x0 + length * math.cos(angle), y0 + length * math.sin(angle)
        draw.line([x0, y0, x1, y1], fill=ink, width=m.hypha_width)
        branch = angle + math.radians(m.branch_angle)
        bx, by = (x0 + x1) / 2, (y0 + y1) / 2
        draw.line([bx, by, bx + 0.4 * length * math.cos(branch), by + 0.4 * length * math.sin(branch)], fill=ink, width=m.hypha_width)
        for _ in range(m.spores_per_head):
            sx, sy = x1 + rng.normal(0, 3 + 2 * m.spore_radius, 2)
            r = m.spore_radius
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(30, 45, 120))
    if device == "smartphone":
        arr = np.asarray(img, float) * np.array([1.06, 1.0, 0.88])
        yy, xx = np.mgrid[:size, :size]
        vignette = 1 - 0.35 * (((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2) ** 2)
        img = Image.fromarray(np.clip(arr * vignette[..., None], 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=0.6 * z)) if z else img


def draw_yeast(m: Morphology, device: str, z: int, size: int, rng: np.random.Generator) -> Image.Image:
    """37 degree phase of a dimorphic isolate: budding cells, no hyphae."""
    img = _noise(size, (205, 215, 235), rng, 5)
    draw = ImageDraw.Draw(img)
    for _ in range(m.spores_per_head * 3):
        cx, cy = rng.uniform(0, size, 2)
        r = m.spore_radius + 1
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 45, 120))
        draw.ellipse([cx + r, cy, cx + 2 * r, cy + r], fill=(30, 45, 120))
    if device == "smartphone":
        arr = np.asarray(img, float) * np.array([1.06, 1.0, 0.88])
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=0.6 * z)) if z else img


def near_duplicate(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """A second shot of the same plate: slight reframing and exposure change."""
    w, h = img.size
    dx, dy = (int(v) for v in rng.integers(1, max(2, w // 16), 2))
    cropped = img.crop((dx, dy, w - dx, h - dy)).resize((w, h))
    arr = np.asarray(cropped, float) * rng.uniform(0.9, 1.1)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_synthetic(out_dir: str | Path, size: int = 128, fovs_per_device: int = 6, openfungi_per_genus: int = 24,
                   seed: int = 0) -> Path:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    def save(img: Image.Image, name: str, **row) -> None:
        path = out / "images" / name
        img.save(path)
        rows.append({"image_path": f"images/{name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **row})

    for c, species in enumerate(PLACEHOLDER_CMU_CLASSES):
        base = _morphology(rng)
        for i in range(3 if c % 2 == 0 else 2):
            iso = f"CMU{c:02d}{i}"
            m = _jitter(base, rng)
            common = {"species": species, "genus": GENUS_OF[species], "isolate_id": iso, "group_id": "", "source": "cmu"}
            for view in ("obverse", "reverse"):
                for day in DAYS:
                    save(draw_colony(m, view, day, size, rng), f"{iso}_colony_{view}_d{day}.png",
                         modality="colony", view=view, device="smartphone", day=day, temperature=25, phase="mold",
                         fov_id="", z_index="", **common)
            phases = [("mold", 25, draw_micro)] + ([("yeast", 37, draw_yeast)] if species in DIMORPHIC else [])
            for phase, temperature, draw in phases:
                for device in ("microscope_camera", "smartphone"):
                    for f in range(fovs_per_device):
                        save(draw(m, device, f % 3, size, rng), f"{iso}_micro_{phase}_{device}_f{f}.png",
                             modality="microscopic", view="na", device=device, day="", temperature=temperature,
                             phase=phase, fov_id=f"f{f}", z_index=f % 3, **common)

    for genus in OPENFUNGI_GENERA:
        base = _morphology(rng)
        for j in range(openfungi_per_genus):
            m = _jitter(base, rng)
            if j % 2:
                img, modality, view = draw_micro(m, "microscope_camera", j % 3, size, rng), "microscopic", "na"
            else:
                view = ("obverse", "reverse")[j // 2 % 2]
                img, modality = draw_colony(m, view, DAYS[j % 3], size, rng), "colony"
            group = f"OF_{genus}_{j:03d}"
            for shot, image in enumerate((img, near_duplicate(img, rng))):
                save(image, f"{group}_shot{shot}.png", species=genus, genus=genus, isolate_id="", group_id=group,
                     modality=modality, view=view, device="unknown", day="", temperature="", phase="na",
                     fov_id="", z_index="", source="openfungi")

    manifest = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
