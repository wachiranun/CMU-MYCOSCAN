# About the classification model design

This document explains the choices behind `src/mycoscan`. Each choice says what the protocol (sections 2.1 to 2.4) asks for, what the code does, and why.

## The isolate is the unit, not the image

The CMU data has 25 to 30 isolates. Each isolate has many images: colony obverse and reverse at several growth days, and about 30 microscope fields or Z-planes per slide from two devices. Two images of the same isolate are close to duplicates. If one sits in training and the other in validation, the model can score well by recognising the isolate instead of the species.

So `splits.py` assigns whole isolates to folds, never single images, and `assert_no_isolate_leakage` runs before every fold trains. The same rule drives two other places:

- The weighted sampler in `data.py` gives every class the same total weight, and every isolate within a class the same share. An isolate photographed 30 times does not outweigh one photographed 5 times.
- The bootstrap confidence intervals in `metrics.py` resample isolates, not images. Resampling images would treat 30 near-copies as 30 independent cases and make the intervals far too narrow. Isolates are resampled within each species, so every replicate contains every class and the macro averages always cover the same classes.
- A CMU row without an `isolate_id` fails at load time. A blank ID would otherwise make the image its own group and let it land on the other side of a split from its sibling images. Only OpenFungi rows may leave it blank.

## Why k-fold is the headline estimate and 80/20 is kept for the protocol

With 2 to 3 isolates per class, a single 80/20 isolate split holds out about 5 isolates in total. That leaves about half of the 10 classes with no validation isolate at all. Their sensitivity cannot be computed, and the remaining classes rest on one isolate each. The synthetic run shows this directly. Its holdout fold had no validation isolate for 5 of 10 classes.

The code implements the protocol's 80/20 split (`split = "holdout"`) and records which classes it leaves without validation isolates. The default is 5-fold isolate-level cross-validation (`split = "kfold"`). Every isolate is validated exactly once, by a model that never saw it, and metrics are computed on the pooled out-of-fold predictions. Leave-one-isolate-out (`split = "loio"`) is the most data-efficient option and costs one training run per isolate. With head-only fine-tuning on a GPU that cost is small.

sklearn's `StratifiedKFold` refuses a fold count larger than every class's member count, which is the normal case here. `splits.py` instead deals each class's isolates across folds in turn. Each class is spread over as many folds as it has isolates, and fold sizes differ by at most one isolate.

## Staged transfer

OpenFungi has 1,249 images, which is too few to pretrain a CNN from scratch. "Pre-trained weights from OpenFungi" is implemented as a chain:

1. ImageNet weights from torchvision.
2. Fine-tune on OpenFungi (5 genera, colony and microscopic images) with the last block unfrozen (`configs/openfungi_pretrain.toml`).
3. Load that backbone, attach a new 10-class head, and fine-tune on CMU data (`weights = "runs/openfungi_densenet121/model.pt"`).

`finetune = "head"` trains only the new fully connected head, as the protocol says. `finetune = "partial"` also trains the last dense block (DenseNet-121) or `layer4` (ResNet-50). `finetune = "full"` trains everything. Frozen BatchNorm layers stay in eval mode during training, so their running statistics keep the pretrained values.

Head-only is the safer default for 25 to 30 isolates. Partial unfreezing may fit fungal texture better, but it has far more parameters to overfit with. Which one wins on the real data is an empirical question, and the k-fold estimate is the way to settle it.

## One model per modality

Colony photographs and microscope fields share almost nothing visually. The scale, background, colour cues and diagnostic structures all differ. A single model would also be dominated by the microscopic set, which has about 10 times more images. So each config trains one modality (`modality = "colony"` or `"microscopic"`). The checkpoint records its modality so the web app can route an image to the right model. `modality = "all"` still exists for experiments.

Because predictions are stored per image with `isolate_id`, combining the two models for one isolate is a mean over both prediction tables (`metrics.aggregate_by_isolate`). This matches how a mycologist combines the colony and the slide.

## Preprocessing and augmentation

Every image is first converted to 8-bit RGB. 16-bit microscope frames are min-max scaled, and transparent pixels are placed on white. Training, validation, `Predictor` and `explain` all use the same conversion, `transforms.to_rgb`.

Both training and validation images then get autocontrast, a resize, and ImageNet normalisation. Autocontrast reduces the brightness and contrast differences between microscope cameras and smartphones (protocol 2.1). It applies one luminance stretch to all three channels (`preserve_tone=True`). A per-channel stretch would turn a red pigment patch grey or black.

Only training images are augmented, with random crops, flips, right-angle rotations, brightness, contrast and saturation jitter, and occasional blur that mimics an out-of-focus Z-plane. Validation and test images are real images, centre-cropped. Hue is never jittered, because pigment colour is diagnostic. The red diffusible pigment of *Talaromyces marneffei* on the colony reverse is one example.

Class imbalance is handled by the isolate-aware weighted sampler (`imbalance = "sampler"`, the default) or by a class-weighted loss (`imbalance = "loss"`). The code never applies both, because that would correct the imbalance twice.

## Metrics

For each class (one versus rest) and as a macro average over classes present in the validation data: sensitivity, specificity, PPV, NPV, F1, accuracy and AUC-ROC. A class that is present but never predicted gets PPV 0. Leaving it undefined would drop it from the macro PPV and inflate the average. Overall accuracy is also reported. Everything is computed twice:

- **Image level**: each image is scored on its own.
- **Isolate level**: the mean of the class probabilities over all of an isolate's images. This is the clinically meaningful number, since a laboratory identifies an isolate, not a photograph.

Training runs a fixed number of epochs and does not pick the best epoch on validation. With 5 validation isolates, picking an epoch on them would make the reported score optimistic.

## Explainability

`explain.py` computes Grad-CAM on the last convolutional block (`features.denseblock4` or `layer4`) and SmoothGrad saliency, which is the mean absolute input gradient over 25 noisy copies. Each image gets a three-panel PNG (original, Grad-CAM overlay, saliency) and a row in `review_sheet.csv` with blank columns for the expert's judgement. That sheet is the record for the protocol's concordance review (2.4). The reviewer checks whether the highlighted regions are conidiophores, hyphae, spores and colony texture, or artifacts such as plate edges, labels, scale bars and dust.

DenseNet's classifier applies an in-place ReLU to the output of `norm5`, which breaks gradient hooks there. That is why Grad-CAM hooks `denseblock4` instead. Gradients flow from the input image, so Grad-CAM also works when the backbone is frozen.
