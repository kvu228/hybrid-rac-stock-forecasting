"""SVM classifier utilities for RAC.

The classifier is intentionally simple: it consumes an engineered feature vector
and outputs a 3-class label with a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC


@dataclass(frozen=True)
class SvmPrediction:
    label: int
    confidence: float
    probabilities: list[float]


def train_svm(x_train: np.ndarray, y_train: np.ndarray) -> SVC:
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")
    clf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    clf.fit(x_train, y_train)
    return clf


def save_svm(model: SVC, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_svm(path: Path) -> SVC:
    return joblib.load(path)


def predict(model: SVC, x: np.ndarray) -> SvmPrediction:
    if x.ndim != 1:
        raise ValueError("x must be 1D feature vector")
    probs = model.predict_proba(x.reshape(1, -1))[0]
    label = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return SvmPrediction(label=label, confidence=confidence, probabilities=[float(p) for p in probs.tolist()])

