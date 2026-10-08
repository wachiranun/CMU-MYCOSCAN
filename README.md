# CMU MycoScan: AI classification model

Component 2 of CMU MycoScan. It trains and validates image classifiers (any [timm](https://github.com/huggingface/pytorch-image-models) backbone, PyTorch) for fungal colony and microscopy images. It also produces Grad-CAM (CNNs) or attention-rollout (ViTs) and saliency maps for a blinded two-rater expert review, and exposes a `predict(image) -> {class: probability}` entry point for the web app.

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

To run on the CMU HPC ERAWAN cluster with Slurm, see [docs/runbook-hpc-slurm.md](docs/runbook-hpc-slurm.md) (also as [HTML](docs/runbook-hpc-slurm.html)).

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

To pin an exact timm tag, put it in `arch` (`arch = "convnext_tiny.fb_in22k_ft_in1k"`). A config whose arch has no weights of the requested kind fails at load time and lists the tags it does have. Verified to build and train: `convnext_tiny`, `convnext_small`, `tf_efficientnetv2_s`, `densenet121`, `resnet50`, `vit_small_patch14_dinov2`, `vit_small_patch16_dinov3`, `vit_base_patch16_224`, `convnext_small.dinov3_lvd1689m`. `arch = "small_cnn"` is a small CNN trained from scratch (four conv-BN-ReLU-max-pool blocks of 32 to 256 channels, then global pooling and a linear head) for the P0 reproduction; it has no pretrained weights, so it needs `weights = "none"`. Transformers are built for `image_size`. `amp = true` turns on mixed precision (float16 on GPU, bfloat16 on CPU) and `grad_clip` caps the gradient norm (0, the default, turns it off).

### Fine-tuning policies

| key | values |
|---|---|
| `finetune` | `head` (default), `partial`, `full`, `lora`, `linear_probe` |
| `partial_blocks` | blocks `partial` unfreezes, counted back from the head (default 1). For ConvNeXt a block is a stage, for a ViT a transformer block. |
| `lora_rank` | rank of the LoRA adapters (default 8). `lora` adds adapters to every Linear layer outside the head and trains only them and the head. Needs `pip install -e ".[lora]"`; without it the run fails before starting and names the extra. The saved checkpoint has the adapters merged into the weights, so it loads without peft. |
| `layer_decay` | layer-wise learning-rate decay (default 0.8): the head trains at `lr`, the last block at `lr * layer_decay`, the one before at `lr * layer_decay**2`, and the stem (a ViT's patch and position embeddings) one step below the first block. 1.0 turns it off. |
| `ema`, `ema_decay` | `ema = true` keeps an exponential moving average of the weights (`ema_decay`, default 0.999, reached after a warm-up of `(1 + n) / (10 + n)` over the first steps). Validation and the saved checkpoints use the averaged weights. |

`finetune = "linear_probe"` trains nothing. It extracts frozen features once with the evaluation transform and caches them in `<output_dir>/feature_cache/`. The cache key covers the manifest hash, the arch, the weights, the image size, autocontrast and the plate crop, so a second run on the same data reuses them and a changed manifest does not. Each fold fits an L2-regularised logistic regression and a cosine k-NN (k = 5) on the training groups' features. `predictions.csv` has a row per image for each classifier, told apart by the `classifier` column. `metrics.json` holds the logistic regression's block at the top and both under `classifiers`. The checkpoints are the frozen backbone with the logistic regression as its head, so `eval`, `predict` and `explain` work on them as usual. Both probes are deterministic, so a probe config refuses more than one entry in `seeds`. Its `resources` include the feature extraction.

### Bags and pooling

| key | values |
|---|---|
| `bag` | `none` (default), `isolate` (all images of an isolate), `isolate_device` (its images from one device), `tiles` (the tiles of one image) |
| `pooling` | `mean` (default) or `max`: how a bag's instance predictions become the bag prediction. `max` takes each class's highest probability over the instances and renormalises. `gated_attention` or `mh_attention`: attention-MIL, which trains on bags (below). |
| `attention_heads`, `attention_dim` | for `mh_attention`, the number of attention heads (default 4); for both attention poolings, the hidden size of the attention gate (default 128) |
| `pooling_hierarchy` | `none` (default) or `device_then_isolate`: with attention pooling and `bag = "isolate"`, pool each device's instances, then the devices |
| `tile_grid`, `tile_size` | for `bag = "tiles"`: the image is resized to `columns * tile_size` by `rows * tile_size` and cut into that grid (default `[3, 2]` at 640 px, which fits a 3:2 camera frame) |

`mean` and `max` are parameter-free, so with them training stays single-instance: single images, or single tiles each labelled with its image's class. Pooling is applied when predicting, at validation and at `mycoscan eval` alike, from the `bag` and `pooling` stored in the checkpoint. `bag = "isolate"` with `pooling = "mean"` gives the same isolate-level metrics as `bag = "none"`, the mean-of-images vote. A bag batch is a padded tensor with a mask, so bags of 10 and 30 instances batch together and padding contributes nothing.

With a bag, `predictions.csv` holds a row per image (`level = "image"`; per tile, `level = "tile"`, for tile bags) and a row per bag (`level = "bag"`, with its `bag` id and `n_instances`). Isolate-level metrics pool the bag rows of each isolate, so `isolate_device` bags are pooled per device, then across devices. Image-level metrics score the image rows (for tile bags, the bag rows, since each bag is one image). Subgroup blocks pool each subgroup's own images. A bag is always built inside one group, and every fold checks that no bag straddles train and validation. Bags need a network: `finetune = "linear_probe"` and `split = "image_random"` refuse them.

With `pooling = "gated_attention"` or `"mh_attention"` the bag becomes the training unit. The backbone embeds every instance of a bag, a gated attention module weighs the instances and pools their embeddings, and a new linear head classifies the pooled embedding; the bag-level loss trains all three, and the fine-tuning policy applies to the backbone as usual. `mh_attention` pools once per head and concatenates. Batches hold as many bags as keep them near `batch_size` instances, and the sampler draws bags so every class and every isolate within it is drawn equally often. The checkpoint holds the attention and head weights, and `mycoscan eval` and `Predictor` rebuild the model from it (`Predictor` scores one image as a bag of one). `attention.csv`, beside `predictions.csv`, has one row per instance: its `fold`, `bag`, `instance` (image path, and `tile`), and its weight in `attention`, or `attention_h0`, `attention_h1`, ... per head. Padded instances get zero weight, and each bag's weights sum to 1. `metrics.json` gains `attention.mean_entropy`, the mean over bags (and heads) of the entropy of the weights, in nats: 0 when one instance carries the bag, log n when n instances share it evenly. The image rows of an attention run score each image as a bag of one.

`pooling_hierarchy = "device_then_isolate"` pools each device's instances of an isolate bag into a device prediction, then a single-head attention over the devices weighs those into the isolate prediction. `predictions.csv` then has a row per device bag (`level = "bag"`, `bag` = `<isolate>|<device>`) and a row per isolate (`level = "isolate"`) whose probabilities are the attention-weighted mean of its device rows, with the device weights in `attention_devices.csv`; isolate-level metrics read the isolate rows. In `attention.csv` the instance weights sum to 1 within each device. Subgroup blocks of an attention run average their images' bag-of-one predictions per bag, since the attention cannot be recomputed on part of a bag.

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

#### P0: the paper's number, leaky and grouped

```bash
mycoscan train --config configs/p0_small_cnn_micro_leaky.toml     # image-level random split, as the paper did
mycoscan train --config configs/p0_small_cnn_micro_grouped.toml   # grouped 5-fold CV inside Pool A
mycoscan results runs --out runs/p0_table.csv
```

The two configs run the same `small_cnn` at 128 px on the micrographs of Pool A and differ only in `split` (and `run_name`); `--set modality=colony` runs the colony photos. In the results table the two pooled rows sit side by side, and the leaky one carries the marker `LEAKY, comparison only`. The gap between them is the leakage inflation. The test suite shows it on a planted-duplicate set whose labels are unrelated to image content: the grouped split stays near chance and the leaky split does not.

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

Each cell runs as `<run_name>__<key>-<value>_...`, so its directory says what it changed. A sweep whose cells set `run_name` or `output_dir`, or whose cells would land in the same directory, is refused before anything runs. A cell that fails is logged, the rest still run, and `sweep_summary.json` names it; the command then exits with status 1. `mycoscan results` reads every `metrics.json` under a directory. It writes one row per run (each fold of each seed, and each seed's pooled predictions, with run name, cell, seed, fold, classifier, isolate macro-F1, accuracy, train fraction, cost and the provenance commit), and in `<stem>_cells.csv` one row per cell with the mean and SD over its folds and seeds. `mycoscan learning-curve` plots each seed's pooled isolate macro-F1 against images (or groups, `--x groups`) per class, with the mean and a ±SD band, and writes the points beside the plot as CSV. The run table lists run-level rows (pooled, holdout, evaluation) first, in run-name order, then the fold rows; it also has `tau`, `isolate_ece`, `isolate_coverage_at_tau`, `isolate_accuracy_at_tau` and the `marker` of leaky runs.

### Paired comparison: sequential against direct

```bash
mycoscan paired runs/cmu_micro_convnext_sequential runs/cmu_micro_convnext_direct --out runs/m1_paired.json
mycoscan paired runs/seq runs/direct --set min_gain=0.03 --set bootstrap=5000
```

A run whose `weights` is a checkpoint is sequential; one starting from `imagenet`, `imagenet22k`, `dino` or `none` is direct. Each sequential run is paired with each direct run of the same arch, fold by fold and seed by seed, from their `summary.json` (single- or multi-seed runs alike). For each pair the command prints the mean difference in isolate macro-F1 (sequential minus direct) in points, its bootstrap CI over the (fold, seed) pairs, the Wilcoxon signed-rank p and the pre-registered verdict: sequential is superior only when the CI excludes 0 and the mean gain is at least 2 points. A sequential run that is worse on average is reported as negative transfer with the recommendation "use direct". The thresholds are config values (`min_gain`, `ci_level`, `require_ci_excludes_zero`, `bootstrap`, `seed`, from `--config` TOML or `--set`) and are written into the `--out` JSON with every pair's differences. The command refuses runs with different seeds, different test isolates (splits file or held-out groups), or different validation images in any fold, and names the first fold that differs.

### Late fusion of a colony run and a microscopy run

```bash
mycoscan fuse --colony runs/cmu_colony --micro runs/cmu_micro --out runs/cmu_fused
mycoscan fuse --colony runs/cmu_colony --micro runs/cmu_micro --out runs/cmu_fused_mlp --method mlp
mycoscan fuse --colony runs/cmu_colony --micro runs/cmu_micro --out runs/cmu_fused \
    --colony-test runs/eval_colony_test --micro-test runs/eval_micro_test
```

The command pools each run's out-of-fold predictions into isolates (by the run's own pooling) and joins them on the isolate. It refuses runs whose classes, seeds, folds, test isolates (splits file or held-out groups) or isolate folds differ, naming the difference; isolates imaged in one modality only are left out and listed under `fusion.unmatched_isolates`. `weighted` (the default) fuses `w * p_colony + (1 - w) * p_micro`, with the colony weight `w` from 0 to 1 in steps of 0.05 chosen by isolate macro-F1, ties going to the weight nearest 0.5. `mlp` trains a one-hidden-layer MLP on the concatenated isolate embeddings of both branches: each image embedded by the branch's frozen initial backbone through the linear probe's feature cache, and averaged per isolate. Both are cross-fitted: each fold's isolates are fused by a weight (or MLP) fitted on the other folds. The final weight is fitted on every development isolate of every seed; with `--colony-test` and `--micro-test`, the `mycoscan eval` directories of both branches' final models on the test set, it is applied once to the test isolates, which never reach a fit, and the result goes to `<out>/test/`. The fused directory has the files of a run (`config.json`, `predictions.csv` with one row per isolate, `metrics.json` with a `fusion` block holding the weight, each fold's weight and the grid, `folds/<fold>/metrics.json` and `summary.json`), so it appears in `mycoscan results` and pairs in `mycoscan paired` like any run. Its `weights` is the branches' when both are sequential or both direct, and `mixed` otherwise, which `mycoscan paired` refuses. Two `holdout` runs have a single fold, so there is nothing to cross-fit: the weight is fitted on the fold it scores, `fusion.cross_fitted` is false, and the fused holdout numbers are optimistic. `--bootstrap` (default 2000) and `--seed` set the isolate bootstrap and the MLP's seed, and `--device` the device that embeds images not yet in the feature cache.

### Calibration and the reject option

Every metric level reports `calibration` (expected calibration error over `calibration_bins` equal-width confidence bins, default 10, with the reliability table: count, mean confidence and accuracy per bin) and `reject_option` (the accuracy-coverage curve: a call is made when the top probability is at least tau, and "no call" otherwise). `reliability_<level>.png` draws the reliability diagram beside the accuracy-coverage curve.

| key | values |
|---|---|
| `tau_rule` | `min_accuracy` (default): the lowest tau whose accepted isolates are at least `tau_target` accurate, so the most coverage. `min_coverage`: the highest tau that still calls at least `tau_target` of isolates. `none`: no reject option. |
| `tau_target` | 0.9 by default |

tau is chosen once per run on the out-of-fold isolate predictions of every seed, from development rows only; asking it to be tuned on a table with sealed test or Pool B rows raises an error. It is the confidence of the least confident accepted development isolate. It goes into `metrics.json` under `tau` (with the selection record) and into the deployable `model.pt`. The pooled isolate-level metrics report accuracy and coverage at it (`isolate_level.reject_option.at_tau`; tau is an isolate threshold, so the image level keeps only its curve), which is optimistic since tau was chosen on those rows. `mycoscan eval` applies the checkpoint's tau as it is, once, and never re-tunes it, so evaluating the sealed test set gives the honest accuracy and coverage. When no threshold reaches the target, tau is `null` and the log says why.

### Stage-1 checkpoints for Plan 2

```bash
mycoscan export-stage1 --config configs/openfungi_stage1_micro.toml                        # runs/stage1/of_micro_<arch>.pt
mycoscan export-stage1 --config configs/openfungi_stage1_micro.toml --set modality=colony   # runs/stage1/of_macro_<arch>.pt
```

This trains the config's backbone and recipe on every Pool A image of one modality (micro has 5 classes, macro 6) and writes a checkpoint without its head, named by modality and arch, with a JSON of its provenance beside it: commit, manifest and splits hashes, resolved config, classes and the groups it trained on. It refuses to run without a `splits_file`, since that is what holds Pool B out, and the training loader refuses any Pool B row. A Stage-2 config names the file as `weights`. The run then checks that the arch matches, records under `provenance.stage1` that only head parameters were newly initialised, and copies the Stage-1 provenance into its own.

### Run outputs

Each run writes `runs/<run_name>/` with these files:

| file | content |
|---|---|
| `config.json` | the resolved config |
| `predictions.csv` | one row per validation image, with out-of-fold probabilities for `kfold` and `loio`, the `classifier` that produced it (`network`, or `logreg` and `knn` for a linear probe) and its `level`; with a `bag`, also a row per bag (see Bags and pooling) |
| `metrics.json` | isolate-level (primary) and image-level (secondary) metrics: accuracy and Top-2 accuracy with Wilson 95% intervals, Cohen's kappa, per-class and macro sensitivity, specificity, PPV, NPV, F1 and AUC, 95% isolate-bootstrap CIs for every one of them, ECE and reliability bins, the accuracy-coverage curve and the metrics at tau, genus and order rollups, and subgroup blocks; plus the chosen `tau`, fold composition, training curves, `subsample`, `resources` (wall seconds, device, GPU-minutes and peak GPU memory, 0 on CPU) and a `provenance` block |
| `confusion_image_level.png`, `confusion_isolate_level.png` | confusion matrices |
| `reliability_image_level.png`, `reliability_isolate_level.png` | reliability diagram and accuracy-coverage curve, tau marked |
| `attention.csv`, `attention_devices.csv` | attention-MIL runs only: each instance's attention weights, and with a device-then-isolate hierarchy each device's (see Bags and pooling) |
| `folds/<fold>/metrics.json`, `folds/<fold>/model.pt` | each fold's metrics (without bootstrap) and checkpoint (`kfold`, `loio`) |
| `summary.json` | mean and SD of isolate macro-F1 and accuracy over all folds and seeds, and a provenance block that adds the seeds, the fold names and membership hashes (all folds, and each fold), the held-out groups, and the hash of each seed's pooled predictions |
| `model.pt` | the deployable model, with its classes, `modality`, `tau`, `bag`, `pooling` (and attention weights) and the run's `provenance`. For `holdout` this is the model trained on 80% of isolates. For `kfold` and `loio` it is retrained on all isolates after cross-validation, with `seed`. |
| `model_card.json` | beside each deployable `model.pt`: intended use and what is out of scope, classes, modality, pooling, tau, the checkpoint's hash, the training data hashes (manifest, splits file, held-out groups, fold membership), the code commit, and headline image- and isolate-level metrics (n, accuracy, Top-2 accuracy, macro-F1) from the predictions the run scored (the holdout validation set, or the out-of-fold development predictions of the final model's `seed`, with the mean and SD over folds) |

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

# Grad-CAM (CNN) or attention rollout (ViT) and SmoothGrad panels plus a blinded review sheet for the mycologists.
# Sample 5 images per class from Pool B (or --pool test with a sealed CMU file), reproducibly:
mycoscan explain --checkpoint runs/<run>/model.pt --manifest data/openfungi/manifest.csv --splits-file data/openfungi/splits_v1.csv --pool B --per-class 5 --seed 0 --out runs/xai_review
# A holdout run's predictions.csv works as a manifest of its held-out images:
mycoscan explain --checkpoint runs/<holdout_run>/model.pt --manifest runs/<holdout_run>/predictions.csv --per-class 3 --out runs/xai_holdout
mycoscan explain --checkpoint runs/<run>/model.pt --images a.png b.jpg --out runs/xai_adhoc --reveal
mycoscan score-review runs/xai_review/review_sheet_rater1.csv runs/xai_review/review_sheet_rater2.csv --out runs/xai_review/scores.json

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

The map is picked from the checkpoint: attention rollout for a vision transformer (attention averaged over heads, with the residual added, multiplied through the layers and read from the class token), Grad-CAM for anything else. Both write the same `review_sheet.csv`: one row per panel with `panel`, `level`, `method`, `predicted`, `probability`, `true_species`, `image_path`, and for each of two raters `rater<k>_focus` and `rater<k>_notes`. A rater judges where the map points on a 3-point scale: `structure` (the diagnostic morphology), `partial` or `background`. The sheet and the panels are blinded: they show the model's call but not the true label or the image path (OpenFungi paths name the class), unless `--reveal` is passed. `review_key.csv` maps each panel to its image and true label; keep it from the raters. `--per-class` draws that many images per class at random with `--seed` (all of a class's images when it has fewer).

An attention-MIL checkpoint gets one panel per bag instead (`level = "bag"` rows, same rater columns): the bag's instances ordered by attention weight with their weights (heads averaged; with a device-then-isolate hierarchy, each instance's share of the isolate), and under the top three the map of that field of view, scored alone, for the bag's call. The question for the rater is the same: does the model look at diagnostic structure. With `--manifest`, `--per-class` draws whole bags (all of an isolate's images, by the checkpoint's `bag`); with `--images` all the images form one bag. `review_key.csv` gives each bag panel its `bag`, its instances in panel order and their `attention`. A `mean` or `max` pooling checkpoint has no learned weights to show, so it gets image panels and the log says so. `mycoscan score-review` reads one sheet with both raters, or one sheet per rater merged by panel, refuses values off the scale, and reports per rater the share of panels judged structure, partial and background, and the raters' agreement (percent and Cohen's kappa, from the same kappa code the metric block uses).

From Python (the web app):

```python
from mycoscan.predict import Predictor

predictor = Predictor("runs/cmu_microscopic_densenet121_head/model.pt")   # load once at startup
predictor.predict(pil_image)   # {"Talaromyces_marneffei": 0.91, "Penicillium_spp": 0.04, ...}, highest first
predictor.modality             # "microscopic": route colony photos to the colony model
predictor.predict_bag([fov1, fov2, fov3])
# {"probabilities": {...}, highest first, "top2": ["Talaromyces_marneffei", "Penicillium_spp"], "no_call": False}
```

`predict_bag` pools one isolate's images the way the checkpoint was trained to: with its learned attention for an attention-MIL model (`devices=` gives each image's device for a device-then-isolate one), otherwise by its `pooling` of the per-image probabilities, so with `mean` it is the mean of `predict` over the images. A tile checkpoint cuts each image into its tiles first, as in training. `no_call` is true when the top probability is below the checkpoint's tau (never when it has none); the Top-2 is given either way. `predict` is unchanged. `model_card.json` beside the model says what it is for and how it scored.

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

The suite runs on CPU in a few minutes. It covers isolate leakage for all three split strategies, the Pool A / Pool B partition and CMU sealing, transforms (validation is never augmented), sampler weights, metrics against hand-computed values (including ECE, the accuracy-coverage curve and tau selection), freeze, LoRA, layer-decay and EMA policies, staged-transfer loading and Stage-1 export, the linear probe and its feature cache, OpenFungi pseudo-grouping, bags, tiles and pooling, gated and multi-head attention-MIL and its hierarchy, late fusion, the P0 leakage inflation on planted duplicates, multi-seed runs, sweeps, results tables and the paired comparison, Grad-CAM, attention rollout, saliency maps and review scoring, and a train, evaluate, explain and predict smoke run. `pytest -m network` also builds a manifest from the real `openfungi/` folders when they are present.
