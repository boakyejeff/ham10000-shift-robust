"""Evaluation: per-source metrics, pooled-vs-cross-source gap, triage curve."""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_auc_score, f1_score)

from .data import CLASSES


def metrics_for(y_true, proba) -> dict:
    pred = proba.argmax(axis=1)
    out = {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }
    # macro AUROC (one-vs-rest); undefined if a class has no pos/neg samples
    try:
        out["macro_auroc"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                                average="macro"))
    except ValueError:
        out["macro_auroc"] = float("nan")
    return out


def per_source_metrics(df_test, proba, source_col="dataset") -> dict:
    """Metrics grouped by acquisition source of the test set."""
    out = {}
    for src, idx in df_test.groupby(source_col).indices.items():
        out[src] = metrics_for(df_test.iloc[idx]["y"].to_numpy(), proba[idx])
    return out


def triage_curve(proba, y_true, thresholds=None):
    """Uncertainty-gated triage: refer if max-softmax < tau.

    Returns list of (tau, referral_rate, accuracy_on_retained)."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.95, 20)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    rows = []
    for tau in thresholds:
        keep = conf >= tau
        rate = float(1.0 - keep.mean())
        acc = float((pred[keep] == y_true[keep]).mean()) if keep.any() else float("nan")
        rows.append({"tau": float(tau), "referral_rate": rate,
                     "accuracy_retained": acc, "n_retained": int(keep.sum())})
    return rows
