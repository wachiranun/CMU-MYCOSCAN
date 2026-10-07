# CMU MycoScan: AI classification model

Component 2 of CMU MycoScan. It trains and validates CNN classifiers (DenseNet-121 or ResNet-50, PyTorch) for fungal colony and microscopy images. It also produces Grad-CAM and saliency maps for expert review and exposes a `predict(image) -> {class: probability}` entry point for the web app.

Design decisions and their reasons are in [docs/design.md](docs/design.md).

## Install

Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate on Linux/HPC
# CPU only:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU (local RTX or HPC): pick the CUDA build that matches the driver
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
mycoscan env                      # prints torch version and whether CUDA is visible
```

`device = "auto"` in a config uses the GPU when one is visible and the CPU otherwise.

## Prepare a manifest

Every dataset is a CSV manifest with one row per image. Image paths are relative to the manifest's folder.

| column | required | values |
|---|---|---|
| `image_path` | yes | relative or absolute path |
| `species` | yes | class label, for example `Talaromyces_marneffei` |
| `isolate_id` | yes, blank for OpenFungi | sequencing-confirmed isolate. Required for every CMU row. |
| `group_id` | OpenFungi only | pseudo-group of repeated shots of one plate. Required for every OpenFungi row. |
| `modality` | yes | `colony` or `microscopic` |
| `view` | no | `obverse`, `reverse`, `na` (default) |
| `device` | no | `microscope_camera`, `smartphone`, `unknown` (default) |
| `day` | no | growth day, for example `3`, `7`, `14` |
| `source` | no | `cmu` (default) or `openfungi` |
| `genus` | no | taxonomic rollup of `species` |
| `temperature` | no | incubation temperature |
| `phase` | no | `mold`, `yeast`, `na` (default); for dimorphic isolates |
| `fov_id`, `z_index` | no | field of view and Z-plane of a microscopic image |
| `sha256` | no | hash of the image file |
| `split`, `fold` | no | written by the splits file, not by hand |

Loading adds a `group` column: `isolate_id` for CMU rows and `group_id` for OpenFungi rows. It is the unit of splitting, sampling and bootstrapping. Loading fails with a clear error for a missing column, an unknown value, a missing image file, a CMU row without an isolate, an OpenFungi row without a group, or an isolate labelled with two species.

## Train with staged transfer

1. Put OpenFungi at `data/openfungi/manifest.csv` with `source=openfungi` and a `group_id` per image (repeated shots of one plate share a group; the OpenFungi manifest builder, a separate ticket, assigns these).
2. Put CMU images at `data/cmu/manifest.csv` with `source=cmu`.
3. Edit the `classes` list in `configs/cmu_*.toml` to the confirmed 10 species/groups.
4. Run the stages in order:

```bash
mycoscan train --config configs/openfungi_pretrain.toml     # ImageNet -> OpenFungi
mycoscan train --config configs/cmu_microscopic.toml        # OpenFungi backbone -> CMU microscopic
mycoscan train --config configs/cmu_colony.toml             # OpenFungi backbone -> CMU colony
```

To override any config key without editing the file, use `--set key=value`. Some examples:

```bash
mycoscan train --config configs/cmu_microscopic.toml --set finetune=partial --set lr=3e-4 --set run_name=cmu_micro_partial
mycoscan train --config configs/cmu_microscopic.toml --set split=holdout   # the protocol's single 80/20 isolate split
mycoscan train --config configs/cmu_microscopic.toml --set split=loio      # leave one isolate out
mycoscan train --config configs/cmu_microscopic.toml --set arch=resnet50 --set weights=imagenet
```

Each run writes `runs/<run_name>/` with these files:

| file | content |
|---|---|
| `config.json` | the resolved config |
| `predictions.csv` | one row per validation image, with out-of-fold probabilities for `kfold` and `loio` |
| `metrics.json` | image-level and isolate-level metrics, per class and macro, 95% isolate-bootstrap CIs, fold composition, training curves |
| `confusion_image_level.png`, `confusion_isolate_level.png` | confusion matrices |
| `folds/<fold>.pt` | one checkpoint per fold (`kfold`, `loio`) |
| `model.pt` | the deployable model. For `holdout` this is the model trained on 80% of isolates. For `kfold` and `loio` it is retrained on all isolates after cross-validation. |

## Evaluate, explain, predict

```bash
# Score a saved model on a new manifest, for example a later external test set
mycoscan eval --checkpoint runs/cmu_microscopic_densenet121_head/model.pt --manifest data/external/manifest.csv --out runs/external_eval

# Grad-CAM and SmoothGrad panels plus review_sheet.csv for the mycologists.
# Explain held-out images: a holdout run's predictions.csv works as a manifest.
mycoscan explain --checkpoint runs/<holdout_run>/model.pt --manifest runs/<holdout_run>/predictions.csv --per-class 3 --out runs/xai_review
mycoscan explain --checkpoint runs/<run>/model.pt --images a.png b.jpg --out runs/xai_adhoc

mycoscan predict --checkpoint runs/<run>/model.pt image.jpg

# Head-only against partial fine-tuning, image and isolate level, with 95% CIs
mycoscan compare runs/cmu_microscopic_densenet121_head runs/cmu_micro_partial
```

`review_sheet.csv` has one row per panel and empty columns for the reviewer to fill in: `reviewer`, `concordant_with_morphology`, `highlighted_structure`, `artifact_suspected`, `notes`.

From Python (the web app):

```python
from mycoscan.predict import Predictor

predictor = Predictor("runs/cmu_microscopic_densenet121_head/model.pt")   # load once at startup
predictor.predict(pil_image)   # {"Talaromyces_marneffei": 0.91, "Penicillium_spp": 0.04, ...}, highest first
predictor.modality             # "microscopic": route colony photos to the colony model
```

## Try it without real data

```bash
mycoscan make-synthetic --out data/synthetic
mycoscan train --config configs/synthetic/openfungi_pretrain.toml
mycoscan train --config configs/synthetic/cmu_microscopic.toml
```

The synthetic images are drawn shapes. Metrics on them prove only that the pipeline runs. They say nothing about real fungi.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The suite runs on CPU in about one minute. It covers isolate leakage for all three split strategies, transforms (validation is never augmented), sampler weights, metrics against hand-computed values, freeze policy, staged-transfer loading, Grad-CAM and saliency maps, and a train, evaluate, explain and predict smoke run.
