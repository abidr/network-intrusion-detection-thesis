"""Section 3.7 -- Evaluation metrics. Implements Eq 12-16.

    Accuracy  = (TP+TN)/(TP+TN+FP+FN)              (Eq 12)
    Precision = TP/(TP+FP)                          (Eq 13)
    Recall    = TP/(TP+FN)                          (Eq 14)
    F1        = 2*P*R/(P+R)                         (Eq 15)
    AUC       -- "based on the true positive and false positive rates at various thresholds"
    latency   = (1/n_t) * sum_i delta_t_i           (Eq 16)

The paper does not state an averaging scheme for multi-class precision/recall/F1. Both weighted
and macro are computed here. On this data macro is the meaningful one -- with ~80% Benign,
weighted averages are dominated by the majority class and track accuracy almost exactly.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    classes: Optional[List[int]] = None,
    latency_s: Optional[float] = None,
) -> Dict:
    """Eq 12-16 for one batch or one full evaluation."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),                                # Eq 12
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }

    # Binary task: the paper's Eq 13-15 are stated in bare TP/FP/FN form and Table 4 reports a
    # single P/R/F1 triple, which is the attack-class (positive) score. Report it explicitly so
    # the comparison to Table 4 is unambiguous.
    present = np.unique(np.concatenate([y_true, y_pred]))
    if len(present) <= 2 and set(present.tolist()) <= {0, 1}:
        out["precision_binary"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        out["recall_binary"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        out["f1_binary"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    out["auc_ovr_macro"] = _safe_auc(y_true, y_proba, classes)

    if latency_s is not None:
        out["latency_total_s"] = float(latency_s)
        out["latency_per_instance_ms"] = float(latency_s / max(1, len(y_true)) * 1000.0)  # Eq 16

    return out


def _safe_auc(y_true, y_proba, classes) -> Optional[float]:
    """One-vs-rest macro AUC, or None when it is undefined for this batch.

    A mini-batch frequently contains only a subset of classes, which makes OvR AUC undefined.
    Returning None (rather than 0.0) keeps those batches out of the average instead of silently
    dragging it down.
    """
    if y_proba is None or classes is None:
        return None
    try:
        present = np.unique(y_true)
        if len(present) < 2:
            return None
        classes = np.asarray(classes)
        keep = np.isin(classes, present)
        proba = np.asarray(y_proba)[:, keep]
        row_sums = proba.sum(axis=1, keepdims=True)
        proba = proba / np.where(row_sums == 0, 1.0, row_sums)
        if len(present) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=present))
    except Exception:
        return None


def aggregate(batch_metrics: List[Dict]) -> Dict:
    """Section 3.7: metrics are 'summed up along the time axis'.

    Aggregated by instance count, so large and small batches are weighted correctly. None-valued
    AUC entries are skipped rather than treated as zero.
    """
    if not batch_metrics:
        return {}

    total_n = sum(m["n"] for m in batch_metrics)
    keys = [
        "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_binary", "recall_binary", "f1_binary",
    ]
    out = {
        k: float(sum(m[k] * m["n"] for m in batch_metrics) / total_n)
        for k in keys
        if all(k in m for m in batch_metrics)
    }

    aucs = [(m["auc_ovr_macro"], m["n"]) for m in batch_metrics if m.get("auc_ovr_macro") is not None]
    out["auc_ovr_macro"] = (
        float(sum(a * n for a, n in aucs) / sum(n for _, n in aucs)) if aucs else None
    )
    out["auc_batches_defined"] = len(aucs)

    lat = [m["latency_per_instance_ms"] for m in batch_metrics if "latency_per_instance_ms" in m]
    if lat:
        out["latency_per_instance_ms"] = float(np.mean(lat))

    out["n_batches"] = len(batch_metrics)
    out["n_instances"] = total_n
    return out


def per_class_report(y_true, y_pred, classes: List[int]) -> Dict[int, Dict]:
    """Per-class precision/recall/F1/support -- the view accuracy hides on imbalanced data."""
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    support = [int(np.sum(np.asarray(y_true) == c)) for c in classes]
    return {
        int(c): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": support[i]}
        for i, c in enumerate(classes)
    }


def confusion(y_true, y_pred, classes: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=classes)
