# About the classification model design

This document explains the choices behind `src/mycoscan`. Each choice says what the protocol (sections 2.1 to 2.4) asks for, what the code does, and why.

## The group is the unit, not the image

The CMU study collects 25 to 30 sequence-confirmed isolates per class, 250 to 300 in all. Each isolate has many images: colony obverse and reverse at several growth days, and 10 to 30 microscope fields or Z-planes per slide from two devices. Two images of the same isolate are close to duplicates. If one sits in training and the other in validation, the model can score well by recognising the isolate instead of the species. OpenFungi has the same problem in a different form: it has no isolate identifiers, but many of its photos are repeated shots of one plate.

So the manifest loader derives one `group` column, which is the isolate for CMU rows and a pseudo-group of repeated shots for OpenFungi rows, and everything downstream uses it. `splits.py` assigns whole groups to folds, never single images, and `assert_no_group_leakage` runs before every fold trains. The same rule drives two other places:

- The weighted sampler in `data.py` gives every class the same total weight, and every group within a class the same share. An isolate photographed 30 times does not outweigh one photographed 5 times.
- The bootstrap confidence intervals in `metrics.py` resample groups, not images. Resampling images would treat 30 near-copies as 30 independent cases and make the intervals far too narrow. Groups are resampled within each species, so every replicate contains every class and the macro averages always cover the same classes.
- A CMU row without an `isolate_id`, or an OpenFungi row without a `group_id`, fails at load time. A blank value would otherwise make the image its own group and let it land on the other side of a split from its sibling images. The loader has an explicit opt-out that loads ungrouped OpenFungi rows as one group per image; it is reserved for the deliberately leaky image-level split that a later ticket adds to reproduce the OpenFungi paper's number, and nothing in the training pipeline uses it.

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

Backbones come from timm, so any timm model name is a valid `arch`, and the code reads what it needs from the model itself instead of from a hand-kept registry. The head is the module timm names as the classifier. The blocks are the taps in timm's `feature_info`, and a block runs from just after the previous tap up to and including its own. For ResNet-50 the last block is `layer4`. For DenseNet-121 it is the last transition, `denseblock4` and `norm5`. For ConvNeXt it is the last stage, and for a ViT it is the last transformer block. The explain layer is the last tap. `densenet121` and `resnet50` were torchvision models before timm, and timm's versions have identical parameter names, so checkpoints from that time still load and predict identically.

Head-only was the safer default when the study was expected to have only a few isolates per class. With 25 to 30 isolates per class, partial and full fine-tuning become realistic, and which depth wins is an empirical question that the pilot on OpenFungi and the k-fold estimate on CMU data settle.

## One model per modality

Colony photographs and microscope fields share almost nothing visually. The scale, background, colour cues and diagnostic structures all differ. A single model would also be dominated by the microscopic set, which has about 10 times more images. So each config trains one modality (`modality = "colony"` or `"microscopic"`). The checkpoint records its modality so the web app can route an image to the right model. `modality = "all"` still exists for experiments.

Because predictions are stored per image with their `group`, combining the two models for one isolate is a mean over both prediction tables (`metrics.aggregate_by_group`). This matches how a mycologist combines the colony and the slide.

## Preprocessing and augmentation

Every image is first converted to 8-bit RGB. 16-bit microscope frames are min-max scaled, and transparent pixels are placed on white. Training, validation, `Predictor` and `explain` all use the same conversion, `transforms.to_rgb`.

Both training and validation images then get autocontrast, a resize, and ImageNet normalisation. Autocontrast reduces the brightness and contrast differences between microscope cameras and smartphones (protocol 2.1). It applies one luminance stretch to all three channels (`preserve_tone=True`). A per-channel stretch would turn a red pigment patch grey or black.

Only training images are augmented, with random crops, flips, right-angle rotations, brightness, contrast and saturation jitter, and occasional blur that mimics an out-of-focus Z-plane. Validation and test images are real images, centre-cropped. Hue is never jittered, because pigment colour is diagnostic. The red diffusible pigment of *Talaromyces marneffei* on the colony reverse is one example.

Class imbalance is handled by the group-aware weighted sampler (`imbalance = "sampler"`, the default) or by a class-weighted loss (`imbalance = "loss"`). The code never applies both, because that would correct the imbalance twice.

## Metrics

For each class (one versus rest) and as a macro average over classes present in the validation data: sensitivity, specificity, PPV, NPV, F1, accuracy and AUC-ROC. A class that is present but never predicted gets PPV 0. Leaving it undefined would drop it from the macro PPV and inflate the average. Overall accuracy is also reported. Everything is computed twice:

- **Image level**: each image is scored on its own.
- **Isolate level**: the mean of the class probabilities over all of an isolate's images. This is the clinically meaningful number, since a laboratory identifies an isolate, not a photograph.

Training runs a fixed number of epochs and does not pick the best epoch on validation. The validation fold is also the fold the reported score comes from, so picking an epoch on it would make that score optimistic, and with few validation isolates per class in interim runs the optimism would be large.

## Explainability

`explain.py` computes Grad-CAM on the backbone's last feature tap (`features.norm5` for DenseNet-121, `layer4` for ResNet-50) and SmoothGrad saliency, which is the mean absolute input gradient over 25 noisy copies. Each image gets a three-panel PNG (original, Grad-CAM overlay, saliency) and a row in `review_sheet.csv` with blank columns for the expert's judgement. That sheet is the record for the protocol's concordance review (2.4). The reviewer checks whether the highlighted regions are conidiophores, hyphae, spores and colony texture, or artifacts such as plate edges, labels, scale bars and dust.

Gradients flow from the input image, so Grad-CAM also works when the backbone is frozen. A ViT's last tap outputs tokens, not a feature map, so Grad-CAM does not apply to it. Attention rollout for ViT backbones is a separate ticket.
