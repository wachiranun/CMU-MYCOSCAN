# CMU MycoScan: AI classification model

Component 2 of CMU MycoScan. It trains and validates image classifiers (any [timm](https://github.com/huggingface/pytorch-image-models) backbone, PyTorch) for fungal colony and microscopy images. It also produces Grad-CAM and saliency maps for expert review and exposes a `predict(image) -> {class: probability}` entry point for the web app.

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
| `split` | no | reserved for the sealed CMU test set; a training loader refuses `test` rows |
| `fold` | no | leave blank; a run with a `splits_file` fills it in |

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
mycoscan train --config configs/cmu_microscopic.toml --set arch=convnext_tiny --set weights=imagenet22k --set amp=true --set grad_clip=1.0
```

### Loss, augmentation and plate crop

| key | values |
|---|---|
| `loss` | `ce` (default), `weighted_ce` (class-weighted; needs `imbalance = "none"`), `focal` |
| `focal_gamma` | focusing exponent of the focal loss, default 2.0 |
| `label_smoothing` | 0.0 (default) to below 1 |
| `augmentation` | `standard` (default), `trivial_wide` (TrivialAugment-Wide without its hue-changing operations), `none` |
| `plate_crop` | `true` crops colony images to the Petri plate circle before any other step, also at evaluation and prediction |
| `imbalance` | `sampler` (default, balances classes and groups within a class) or `none` |

### Backbones and initialisation

`arch` is any timm model name, and `weights` picks its initialisation:

| `weights` | meaning |
|---|---|
| `imagenet` | weights trained on ImageNet-1k alone (timm's default tag when it is one), never a larger pretraining fine-tuned on 1k, so `imagenet` and `imagenet22k` stay separate arms. `densenet121` and `resnet50` keep the torchvision weights they always used. |
| `imagenet22k` | ImageNet-22k (or 21k) pretraining, preferring the tag not fine-tuned on 1k, for example `convnext_tiny.fb_in22k` |
| `dino` | self-supervised DINO weights: a DINO tag (`convnext_small.dinov3_lvd1689m`) or a DINO arch (`vit_small_patch14_dinov2`) |
| `none` | random initialisation |
| a path | an earlier-stage checkpoint of the same arch; its head is replaced |

To pin an exact timm tag, put it in `arch` (`arch = "convnext_tiny.fb_in22k_ft_in1k"`). A config whose arch has no weights of the requested kind fails at load time and lists the tags it does have. Verified to build and train: `convnext_tiny`, `convnext_small`, `tf_efficientnetv2_s`, `densenet121`, `resnet50`, `vit_small_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_224`, `convnext_small.dinov3_lvd1689m`. Transformers are built for `image_size`. `amp = true` turns on mixed precision (float16 on GPU, bfloat16 on CPU) and `grad_clip` caps the gradient norm (0, the default, turns it off).

### Frozen OpenFungi partition (Pool A / Pool B)

```bash
mycoscan partition --manifest data/openfungi/manifest.csv --out data/openfungi/splits_v1.csv
```

This assigns every OpenFungi group to Pool A (70%, development, with 5 cross-validation folds) or Pool B (30%, external test), stratified by class and modality. It writes `splits_v1.csv` and a `splits_v1.csv.sha256` sidecar, and refuses to overwrite either. A run uses it with `splits_file = "data/openfungi/splits_v1.csv"`. The run refuses to start if the manifest's hash differs from the one recorded in the file. It trains and validates on Pool A only, using the frozen folds when `split = "kfold"`. A training loader that receives a Pool B row raises an error naming the row.

`split = "image_random"` splits images at random and ignores groups, which reproduces the leaky OpenFungi paper number. Its outputs are marked `"leaky": true` and its log says "leaky, comparison only".

Each run writes `runs/<run_name>/` with these files:

| file | content |
|---|---|
| `config.json` | the resolved config |
| `predictions.csv` | one row per validation image, with out-of-fold probabilities for `kfold` and `loio` |
| `metrics.json` | isolate-level (primary) and image-level (secondary) metrics: accuracy and Top-2 accuracy with Wilson 95% intervals, Cohen's kappa, per-class and macro sensitivity, specificity, PPV, NPV, F1 and AUC, 95% isolate-bootstrap CIs for every one of them, genus and order rollups, and subgroup blocks; plus fold composition, training curves and a `provenance` block |
| `confusion_image_level.png`, `confusion_isolate_level.png` | confusion matrices |
| `folds/<fold>.pt` | one checkpoint per fold (`kfold`, `loio`) |
| `model.pt` | the deployable model. For `holdout` this is the model trained on 80% of isolates. For `kfold` and `loio` it is retrained on all isolates after cross-validation. |

### Provenance and MLflow

The `provenance` block in every `metrics.json`, from training or `mycoscan eval`, records what the numbers came from:

- the git commit, and `dirty: true` if any file differed from it when the run started
- the SHA-256 of the manifest, of the splits file when one is used, and of the checkpoint for an evaluation
- the resolved config
- the versions of Python and the key packages

To log the same params, metrics and report files to MLflow as well, install the extra and set `tracking = "mlflow"`:

```bash
pip install -e ".[mlflow]"
MLFLOW_TRACKING_URI=http://tracker:5000 mycoscan train --config configs/cmu_microscopic.toml --set tracking=mlflow
```

Without `MLFLOW_TRACKING_URI`, MLflow writes to `mlflow.db` in the working directory. Checkpoints are not uploaded. A run with `tracking = "mlflow"` but without MLflow installed fails before training and names the extra.

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

Three config keys shape the metric block. `genus_map` is a `[genus_map]` table of species to genus; when it is absent, the manifest's `genus` column is used, and a class with no genus skips the rollup with a reason in `taxonomic_rollup_skipped`. `order_map` (genus to order) adds an order-level rollup. `subgroups = ["device", "phase"]` repeats the whole report for each value of each column, under `subgroups` in `metrics.json`. `bootstrap` sets the number of resamples (2,000 by default).

`mycoscan eval --metric-config metrics.toml` takes the same `genus_map`, `order_map` and `subgroups`, and also a `label_map` for an external set whose labels are coarser than the model's classes:

```toml
[label_map]
Flavi = ["Aspergillus_flavus"]
Nigri = ["Aspergillus_niger"]
Fusarium = ["FSSC", "FOSC"]
Rhizopus = ["Rhizopus"]
```

The model's probabilities are summed into each reference label. The model classes that no label covers go into an `unmapped` column, so predicting one of them counts as wrong. Rows whose reference label is not in the map are dropped. `metrics.json` names both under `label_mapping`.

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
pytest -m network     # also downloads pretrained weights and trains on them
```

The suite runs on CPU in about one minute. It covers isolate leakage for all three split strategies, transforms (validation is never augmented), sampler weights, metrics against hand-computed values, freeze policy, staged-transfer loading, Grad-CAM and saliency maps, and a train, evaluate, explain and predict smoke run.
