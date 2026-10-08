"""Linear probe and k-NN on frozen features: the cheap baseline every backbone gets.

Per fold, an L2-regularised logistic regression (on standardised features) and a
cosine k-NN (k = 5) are fitted on the training groups' features and scored on the
validation groups'. The logistic regression is also exported as a linear head on
the frozen backbone, so a probe run leaves a checkpoint that `eval`, `predict`
and `explain` use like any other.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn

from .models import adapter

PROBE_C = 1.0  # inverse L2 strength of the logistic regression
KNN_K = 5
ABSENT_LOGIT = -1e4  # a class missing from a fold's training set is never predicted


@dataclass(frozen=True)
class LinearHead:
    weight: np.ndarray  # (n_classes, n_features), acting on raw features
    bias: np.ndarray


def _all_classes(probs: np.ndarray, seen: np.ndarray, n_classes: int) -> np.ndarray:
    full = np.zeros((len(probs), n_classes))
    full[:, seen] = probs
    return full


def _as_linear(scaler: StandardScaler, logreg: LogisticRegression, n_features: int, n_classes: int) -> LinearHead:
    """The logistic regression as softmax(weight @ x + bias) on unstandardised features."""
    coef, intercept = logreg.coef_, logreg.intercept_
    if coef.shape[0] == 1:  # two classes: sklearn keeps one row, and sigmoid(z) is softmax([-z/2, z/2])
        coef, intercept = np.vstack([-coef / 2, coef / 2]), np.array([-intercept[0] / 2, intercept[0] / 2])
    weight = coef / scaler.scale_
    bias = intercept - weight @ scaler.mean_
    full_weight, full_bias = np.zeros((n_classes, n_features)), np.full(n_classes, ABSENT_LOGIT)
    full_weight[logreg.classes_], full_bias[logreg.classes_] = weight, bias
    return LinearHead(full_weight, full_bias)


@dataclass(frozen=True)
class Probes:
    scaler: StandardScaler
    logreg: LogisticRegression
    knn: KNeighborsClassifier
    n_features: int
    n_classes: int

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Class probabilities of each probe, logreg first: it is the run's primary classifier."""
        return {"logreg": _all_classes(self.logreg.predict_proba(self.scaler.transform(x)), self.logreg.classes_,
                                       self.n_classes),
                "knn": _all_classes(self.knn.predict_proba(x), self.knn.classes_, self.n_classes)}

    def linear_head(self) -> LinearHead:
        return _as_linear(self.scaler, self.logreg, self.n_features, self.n_classes)


def fit_probes(train_x: np.ndarray, train_y: np.ndarray, n_classes: int) -> Probes:
    scaler = StandardScaler().fit(train_x)
    logreg = LogisticRegression(C=PROBE_C, max_iter=5000).fit(scaler.transform(train_x), train_y)
    knn = KNeighborsClassifier(n_neighbors=min(KNN_K, len(train_y)), metric="cosine").fit(train_x, train_y)
    return Probes(scaler, logreg, knn, train_x.shape[1], n_classes)


def with_linear_head(backbone: nn.Module, head: LinearHead) -> nn.Module:
    """A copy of `backbone` (built by models.build_model) whose head computes the probe."""
    model = copy.deepcopy(backbone)
    linear = cast(nn.Linear, cast(nn.Sequential, model.get_submodule(adapter(model).head))[1])
    with torch.no_grad():
        linear.weight.copy_(torch.as_tensor(head.weight))
        linear.bias.copy_(torch.as_tensor(head.bias))
    return model.eval()
