# About the classification model design

This document explains the choices behind `src/mycoscan`. Each choice says what the protocol (sections 2.1 to 2.4) asks for, what the code does, and why.

## The group is the unit, not the image

The CMU study collects 25 to 30 sequence-confirmed isolates per class, 250 to 300 in all. Each isolate has many images: colony obverse and reverse at several growth days, and 10 to 30 microscope fields or Z-planes per slide from two devices. Two images of the same isolate are close to duplicates. If one sits in training and the other in validation, the model can score well by recognising the isolate instead of the species. OpenFungi has the same problem in a different form: it has no isolate identifiers, but many of its photos are repeated shots of one plate.

So the manifest loader derives one `group` column, which is the isolate for CMU rows and a pseudo-group of repeated shots for OpenFungi rows, and everything downstream uses it. `splits.py` assigns whole groups to folds, never single images, and `assert_no_group_leakage` runs before every fold trains. The same rule drives two other places:

- The weighted sampler in `data.py` gives every class the same total weight, and every group within a class the same share. An isolate photographed 30 times does not outweigh one photographed 5 times.
- The bootstrap confidence intervals in `metrics.py` resample groups, not images. Resampling images would treat 30 near-copies as 30 independent cases and make the intervals far too narrow. Groups are resampled within each species, so every replicate contains every class and the macro averages always cover the same classes.
- A CMU row without an `isolate_id`, or an OpenFungi row without a `group_id`, fails at load time. A blank value would otherwise make the image its own group and let it land on the other side of a split from its sibling images. The loader has an explicit opt-out that loads ungrouped OpenFungi rows as one group per image. Only the deliberately leaky `image_random` split uses it (see below).

## Pool A and Pool B: a partition made once

Plan 2 scores its models on OpenFungi images that no model in either plan was trained on (Pool B). That only holds if the partition never changes, so `mycoscan partition` writes it once, to a splits file such as `splits_v1.csv`, and refuses to overwrite it. Every OpenFungi group goes to Pool A (development, 70%) or Pool B (external test, 30%). Within each class, groups are ordered by modality and Pool B takes evenly spaced ones, so every class gives up the same share and the colony and microscopic groups of a class are split in proportion. Pool A groups also get one of 5 folds for grouped cross-validation.

The file records the SHA-256 of the manifest it was built from, and a sidecar (`splits_v1.csv.sha256`) records the file's own hash. A run that names the file in `splits_file` checks both before it starts. It refuses if the manifest has changed since the partition, because a changed manifest can move images between groups. It then merges each group's pool and fold into the manifest, drops Pool B, and with `split = "kfold"` uses the frozen folds. A second defence sits in the data loader: a training loader raises, naming the rows, if any Pool B (or sealed test) image reaches it. That guard catches a wrong config or a future code path that skips the merge.

## The leaky split, kept for comparison

The OpenFungi paper's 99.8% comes from a random image-level split, where near-duplicate shots of one plate sit on both sides. `split = "image_random"` reproduces it: stratified folds of images, ignoring groups, and the group leakage check is skipped. Its `metrics.json` and `config.json` carry `"leaky": true` and its log lines say "leaky, comparison only". It exists only so the pilot report can show the leaky number next to the grouped one.

## Why k-fold is the headline estimate and 80/20 is kept for the protocol

A single 80/20 isolate split of 25 to 30 isolates per class holds out 5 or 6 isolates per class. That is enough for a go/no-go on overall accuracy but gives per-class estimates with very wide intervals, and in the interim runs on partial data (12 or 20 isolates per class) it leaves some classes with one or two validation isolates. The synthetic set, which has only 2 to 3 isolates per class, shows the extreme case: its holdout fold had no validation isolate for 5 of 10 classes.

The code implements the protocol's 80/20 split (`split = "holdout"`) and records which classes it leaves without validation groups. The default is 5-fold group-level cross-validation (`split = "kfold"`). Every group is validated exactly once, by a model that never saw it, and metrics are computed on the pooled out-of-fold predictions over every development isolate, which is the highest-precision estimate available. Leave-one-group-out (`split = "loio"`) is the most data-efficient option and costs one training run per group. With head-only fine-tuning on a GPU that cost is small.

sklearn's `StratifiedKFold` refuses a fold count larger than every class's member count, which small OpenFungi classes and interim CMU counts can hit. `splits.py` instead deals each class's groups across folds in turn. Each class is spread over as many folds as it has groups, and fold sizes differ by at most one group.

## Staged transfer

OpenFungi has 1,249 images, which is too few to pretrain a CNN from scratch. "Pre-trained weights from OpenFungi" is implemented as a chain:

1. ImageNet weights (or ImageNet-22k, or self-supervised DINO weights) from timm.
2. Fine-tune on OpenFungi (5 genera, colony and microscopic images) with the last block unfrozen (`configs/openfungi_pretrain.toml`).
3. Load that backbone, attach a new 10-class head, and fine-tune on CMU data (`weights = "runs/openfungi_densenet121/model.pt"`).

`finetune = "head"` trains only the new fully connected head, as the protocol says. `finetune = "partial"` also trains the last block and everything after it, such as final norms. `finetune = "full"` trains everything. Frozen BatchNorm layers stay in eval mode during training, so their running statistics keep the pretrained values. LayerNorm has no running statistics, so a frozen LayerNorm only needs its parameters frozen.

Backbones come from timm, so any timm model name is a valid `arch`, and the code reads what it needs from the model itself instead of from a hand-kept registry. The head is the module timm names as the classifier. The blocks are the taps in timm's `feature_info`, and a block runs from just after the previous tap up to and including its own. For ResNet-50 the last block is `layer4`. For DenseNet-121 it is the last transition, `denseblock4` and `norm5`. For ConvNeXt it is the last stage, and for a ViT it is the last transformer block. The explain layer is the last tap. `weights = "imagenet"` takes only weights trained on ImageNet-1k alone. timm's default tag is often a larger pretraining (ImageNet-12k or 21k) fine-tuned on 1k, and taking it would blur the comparison with `imagenet22k`.

The plate crop is decided in one place, `transforms.load_image`, from each image's own modality. Training, evaluation, `Predictor` and `explain` all load images through it, so a `modality = "all"` model crops its colony photos and leaves its microscope fields alone. `densenet121` and `resnet50` were torchvision models before timm, and timm's versions have identical parameter names, so checkpoints from that time still load and predict identically.

Head-only was the safer default when the study was expected to have only a few isolates per class. With 25 to 30 isolates per class, partial and full fine-tuning become realistic, and which depth wins is an empirical question that the pilot on OpenFungi and the k-fold estimate on CMU data settle.

## One model per modality

Colony photographs and microscope fields share almost nothing visually. The scale, background, colour cues and diagnostic structures all differ. A single model would also be dominated by the microscopic set, which has about 10 times more images. So each config trains one modality (`modality = "colony"` or `"microscopic"`). The checkpoint records its modality so the web app can route an image to the right model. `modality = "all"` still exists for experiments.

Because predictions are stored per image with their `group`, combining the two models for one isolate is a mean over both prediction tables (`metrics.aggregate_by_group`). This matches how a mycologist combines the colony and the slide.

## Preprocessing and augmentation

Every image is first converted to 8-bit RGB. 16-bit microscope frames are min-max scaled, and transparent pixels are placed on white. Training, validation, `Predictor` and `explain` all use the same conversion, `transforms.to_rgb`.

Colony photographs can then be cropped to the Petri plate (`plate_crop = true`), before any other step and for training, validation, `Predictor` and `explain` alike. The plate is the largest region on the other side of an Otsu threshold from the image border, with the colony filled in. The crop is the plate's bounding square, and everything outside the plate circle is set to black. This removes the bench, labels and plate edges that a model could otherwise learn instead of the colony. A photo with no plate-sized region is left as it is. Crops are always regenerated from the raw image by code, so derived images never diverge from the source.

Both training and validation images then get autocontrast, a resize, and ImageNet normalisation. Autocontrast reduces the brightness and contrast differences between microscope cameras and smartphones (protocol 2.1). It applies one luminance stretch to all three channels (`preserve_tone=True`). A per-channel stretch would turn a red pigment patch grey or black.

Only training images are augmented, and `augmentation` picks the recipe. `standard` (the default) uses random crops, flips, right-angle rotations, brightness, contrast and saturation jitter, and occasional blur that mimics an out-of-focus Z-plane. `trivial_wide` uses random crops, flips and TrivialAugment-Wide, which applies one random operation per image at a random strength. `none` gives training images the validation transform. Validation and test images are real images, centre-cropped. Hue is never changed, because pigment colour is diagnostic. The red diffusible pigment of *Talaromyces marneffei* on the colony reverse is one example. TrivialAugment-Wide's Solarize, Posterize, Equalize and AutoContrast operations are dropped because each one changes hue. Equalize and AutoContrast stretch each channel separately. Brightness, saturation and contrast keep hue because they blend towards black or grey. A test checks every recipe against pure pigment patches.

Class imbalance is handled by the group-aware weighted sampler (`imbalance = "sampler"`, the default) or by a class-weighted loss (`loss = "weighted_ce"` with `imbalance = "none"`). The code never applies both, because that would correct the imbalance twice: the config refuses that combination. `loss = "focal"` (with `focal_gamma`) down-weights images the model already gets right, and `label_smoothing` applies to every loss.

## Metrics

For each class (one versus rest) and as a macro average over classes present in the validation data: sensitivity, specificity, PPV, NPV, F1, accuracy and AUC-ROC. A class that is present but never predicted gets PPV 0. Leaving it undefined would drop it from the macro PPV and inflate the average. Overall accuracy is also reported. Everything is computed twice:

- **Image level**: each image is scored on its own.
- **Isolate level**: the mean of the class probabilities over all of an isolate's images. This is the clinically meaningful number, since a laboratory identifies an isolate, not a photograph.

Training runs a fixed number of epochs and does not pick the best epoch on validation. The validation fold is also the fold the reported score comes from, so picking an epoch on it would make that score optimistic, and with few validation isolates per class in interim runs the optimism would be large.

## Provenance

A number in the pilot or main-study report is only worth quoting if it can be traced to the exact code, data and settings that produced it. So every training and evaluation run writes a `provenance` block into `metrics.json`. The block holds the git commit and a dirty flag, the SHA-256 of the manifest (and of the splits file or checkpoint), the resolved config, and package versions. The git state is read when the run starts, so editing code during a long run does not change what is recorded. Untracked files count as dirty, because a new module can change what runs. `data/` and `runs/` are git-ignored, so they never do. MLflow logging is optional (`tracking = "mlflow"`), because the suite and the Colab, Kaggle and cluster environments must work without a tracking server. The file is always the record, and MLflow is a convenience view of it.

## Explainability

`explain.py` computes Grad-CAM on the backbone's last feature tap (`features.norm5` for DenseNet-121, `layer4` for ResNet-50) and SmoothGrad saliency, which is the mean absolute input gradient over 25 noisy copies. Each image gets a three-panel PNG (original, Grad-CAM overlay, saliency) and a row in `review_sheet.csv` with blank columns for the expert's judgement. That sheet is the record for the protocol's concordance review (2.4). The reviewer checks whether the highlighted regions are conidiophores, hyphae, spores and colony texture, or artifacts such as plate edges, labels, scale bars and dust.

Gradients flow from the input image, so Grad-CAM also works when the backbone is frozen. A ViT's last tap outputs tokens, not a feature map, so Grad-CAM does not apply to it. Attention rollout for ViT backbones is a separate ticket.
