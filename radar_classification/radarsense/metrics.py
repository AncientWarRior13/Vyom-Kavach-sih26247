"""Shared metric computation for all pipeline stages."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from radarsense import config as C


def full_metrics(y_true, y_pred, y_score=None, classes=C.CLASSES):
    """y_true/y_pred are string labels; y_score is (n, n_classes) proba or None."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for i, cls in enumerate(classes):
        out[f"precision_{cls}"] = float(prec[i])
        out[f"recall_{cls}"] = float(rec[i])
        out[f"f1_{cls}"] = float(f1[i])
    if y_score is not None:
        y_drone = (np.asarray(y_true) == C.MINORITY_CLASS).astype(int)
        out["pr_auc_drone"] = float(
            average_precision_score(y_drone, y_score[:, C.MINORITY_IDX]))
    return out


def confusion(y_true, y_pred, classes=C.CLASSES):
    return confusion_matrix(y_true, y_pred, labels=classes).tolist()
