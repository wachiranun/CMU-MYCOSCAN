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
pip install -e ".[lora]"          # optional: peft, for finetune = "lora"
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
| `batch`, `year` | no | imaging batch and year; `mycoscan seal` spreads them across folds |
| `split` | no | reserved for the sealed CMU test set; a training loader refuses `test` rows |
| `fold` | no | leave blank; a run with a `splits_file` fills it in |

Loading adds a `group` column: `isolate_id` for CMU rows and `group_id` for OpenFungi rows. It is the unit of splitting, sampling and bootstrapping. Loading fails with a clear error for a missing column, an unknown value, a missing image file, a CMU row without an isolate, an OpenFungi row without a group, or an isolate labelled with two species.

## Train with staged transfer

1. Build the OpenFungi manifest with `mycoscan build-openfungi` (see below). It assigns a `group_id` per image, so repeated shots of one plate share a group.
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

### Fine-tuning policies

| key | values |
|---|---|
| `finetune` | `head` (default), `partial`, `full`, `lora`, `linear_probe` |
| `partial_blocks` | blocks `partial` unfreezes, counted back from the head (default 1). For ConvNeXt a block is a stage, for a ViT a transformer block. |
| `lora_rank` | rank of the LoRA adapters (default 8). `lora` adds adapters to every Linear layer outside the head and trains only them and the head. Needs `pip install -e ".[lora]"`; without it the run fails before starting and names the extra. The saved checkpoint has the adapters merged into the weights, so it loads without peft. |
| `layer_decay` | layer-wise learning-rate decay (default 0.8): the head trains at `lr`, the last block at `lr * layer_decay`, the one before at `lr * layer_decay**2`, and the stem (a ViT's patch and position embeddings) one step below the first block. 1.0 turns it off. |
| `ema`, `ema_decay` | `ema = true` keeps an exponential moving average of the weights (`ema_decay`, default 0.999, reached after a warm-up of `(1 + n) / (10 + n)` over the first steps). Validation and the saved checkpoints use the averaged weights. |

`finetune = "linear_probe"` trains nothing. It extracts frozen features once with the evaluation transform and caches them in `<output_dir>/feature_cache/`. The cache key covers the manifest hash, the arch, the weights, the image size, autocontrast and the plate crop, so a second run on the same data reuses them and a changed manifest does not. Each fold fits an L2-regularised logistic regression and a cosine k-NN (k = 5) on the training groups' features. `predictions.csv` has a row per image for each classifier, told apart by the `classifier` column. `metrics.json` holds the logistic regression's block at the top and both under `classifiers`. The checkpoints are the frozen backbone with the logistic regression as its head, so `eval`, `predict` and `explain` work on them as usual. Both probes are deterministic, so a probe config refuses more than one entry in `seeds`. Its `resources` include the feature extraction.

### OpenFungi manifest and pseudo-groups

```bash
mycoscan build-openfungi --root openfungi --out data/openfungi/manifest.csv --config configs/openfungi_manifest.toml
mycoscan build-openfungi --root openfungi --out data/openfungi/manifest.csv --set hamming_threshold=6   # tighter
```

This walks `openfungi/macro/<class>/` (colony photos) and `openfungi/micro/<class>/` (micrographs) and writes a manifest with class, modality, image size, SHA-256, perceptual hash (`phash`) and `group_id`. Within each modality and class, two images are joined into one group when their pHash Hamming distance is at most `hamming_threshold` (8) and the cosine distance of their frozen DINO embeddings is below `cosine_threshold` (0.15). The embedding model (`embed_arch`, `embed_weights`, `embed_image_size`) comes from the same backbone registry as training. The `Mixed` macro class is left out unless `include_mixed = true`. `contact_sheets/<group_id>.png`, beside the manifest, shows every group with more than one image, so a mycologist can confirm the groups. The printed summary lists images and groups per class and flags classes under `min_images` (20) as under-powered.

To report the four well-supported macro classes alone, list them in `select_classes`, as `configs/openfungi_macro_well_supported.toml` does. Rows of other classes are dropped before splitting.

### Frozen OpenFungi partition (Pool A / Pool B)

```bash
mycoscan partition --manifest data/openfungi/manifest.csv --out data/openfungi/splits_v1.csv
```

This assigns every OpenFungi group to Pool A (70%, development, with 5 cross-validation folds) or Pool B (30%, external test), stratified by class and modality. It writes `splits_v1.csv` and a `splits_v1.csv.sha256` sidecar, and refuses to overwrite either. A run uses it with `splits_file = "data/openfungi/splits_v1.csv"`. The run refuses to start if the manifest's hash differs from the one recorded in the file. It trains and validates on Pool A only, using the frozen folds when `split = "kfold"`. A training loader that receives a Pool B row raises an error naming the row.

`split = "image_random"` splits images at random and ignores groups, which reproduces the leaky OpenFungi paper number. Its outputs are marked `"leaky": true` and its log says "leaky, comparison only".

### Sealed CMU test set

```bash
mycoscan seal --manifest data/cmu/manifest.csv --out data/cmu/splits_cmu.csv
```

Run by someone other than the modeller, this seals 15% of each class's isolates as the locked test set (`split = test`). The rest become development isolates (`split = dev`) with one of 5 folds. Within a class, isolates are ordered by `batch`, `year` and devices, and test picks and fold assignments are spread over that order, so the test set and the folds mix all three. Dev isolates go to the class's least-filled fold, so each class is spread over the folds as evenly as its count allows. The command refuses while any class has fewer than 8 isolates, and names those classes, unless `--force` is passed. Re-run it after new isolates are imaged. Only the new ones are assigned, topping each class's test share back up to 15%, and an isolate already sealed never moves. The file gets a `.sha256` sidecar like the OpenFungi splits file. A run with `splits_file = "data/cmu/splits_cmu.csv"` trains and validates on development isolates only, with the sealed folds for `split = "kfold"`, and the training-loader guard refuses any test isolate.

### Seeds, sweeps and results tables

`seeds = [0, 1, 2]` runs every fold once per seed. Folds come from `seed` alone, so every seed validates the same groups in the same folds; seeds change initialisation and sampling only. `train_fraction = 0.5` keeps half of each class's groups, chosen at random, before folds are made, so a learning curve never splits a group. With a splits file's frozen folds, it keeps half of each class within each fold, so no fold loses a class. `metrics.json` records the kept images and groups per class and the removed groups under `subsample`.

A sweep file names a base config and a list of override cells:

```bash
mycoscan sweep configs/sweeps/p7_learning_curve.toml          # one run per cell, under runs/p7_learning_curve/
mycoscan results runs/p7_learning_curve --out runs/p7_learning_curve/table.csv
mycoscan learning-curve runs/p7_learning_curve --x images --out runs/p7_learning_curve/curve.png
```

Each cell runs as `<run_name>__<key>-<value>_...`, so its directory says what it changed. A sweep whose cells set `run_name` or `output_dir`, or whose cells would land in the same directory, is refused before anything runs. A cell that fails is logged, the rest still run, and `sweep_summary.json` names it; the command then exits with status 1. `mycoscan results` reads every `metrics.json` under a directory. It writes one row per run (each fold of each seed, and each seed's pooled predictions, with run name, cell, seed, fold, classifier, isolate macro-F1, accuracy, train fraction, cost and the provenance commit), and in `<stem>_cells.csv` one row per cell with the mean and SD over its folds and seeds. `mycoscan learning-curve` plots each seed's pooled isolate macro-F1 against images (or groups, `--x groups`) per class, with the mean and a ±SD band, and writes the points beside the plot as CSV.

### Run outputs

Each run writes `runs/<run_name>/` with these files:

| file | content |
|---|---|
| `config.json` | the resolved config |
| `predictions.csv` | one row per validation image, with out-of-fold probabilities for `kfold` and `loio`, and the `classifier` that produced it (`network`, or `logreg` and `knn` for a linear probe) |
| `metrics.json` | isolate-level (primary) and image-level (secondary) metrics: accuracy and Top-2 accuracy with Wilson 95% intervals, Cohen's kappa, per-class and macro sensitivity, specificity, PPV, NPV, F1 and AUC, 95% isolate-bootstrap CIs for every one of them, genus and order rollups, and subgroup blocks; plus fold composition, training curves, `subsample`, `resources` (wall seconds, device, GPU-minutes and peak GPU memory, 0 on CPU) and a `provenance` block |
| `confusion_image_level.png`, `confusion_isolate_level.png` | confusion matrices |
| `folds/<fold>/metrics.json`, `folds/<fold>/model.pt` | each fold's metrics (without bootstrap) and checkpoint (`kfold`, `loio`) |
| `summary.json` | mean and SD of isolate macro-F1 and accuracy over all folds and seeds, and a provenance block that adds the seeds, the fold names and membership hash, and the hash of each seed's pooled predictions |
| `model.pt` | the deployable model. For `holdout` this is the model trained on 80% of isolates. For `kfold` and `loio` it is retrained on all isolates after cross-validation, with `seed`. |

With `seeds` set, everything but `config.json`, `summary.json` and the final `model.pt` moves into `seed<seed>/`, one directory per seed.

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

The suite runs on CPU in a few minutes. It covers isolate leakage for all three split strategies, the Pool A / Pool B partition and CMU sealing, transforms (validation is never augmented), sampler weights, metrics against hand-computed values, freeze, LoRA, layer-decay and EMA policies, staged-transfer loading, the linear probe and its feature cache, OpenFungi pseudo-grouping, multi-seed runs, sweeps and results tables, Grad-CAM and saliency maps, and a train, evaluate, explain and predict smoke run. `pytest -m network` also builds a manifest from the real `openfungi/` folders when they are present.
