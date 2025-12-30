"""Evaluation metrics.

Headline metrics: AUPRC (average precision) and F1 / decision-cost at a
threshold tuned on VALIDATION predictions only. ROC-AUC is diagnostic.
Accuracy is deliberately absent -- the task is imbalanced.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def auprc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float | None:
    """Diagnostic only; None when a split has a single class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, np.asarray(scores)))


def f1_at_threshold(y_true, scores, threshold: float) -> float:
    preds = (np.asarray(scores) >= threshold).astype(int)
    return float(f1_score(y_true, preds, zero_division=0))


def _threshold_candidates(scores_val, grid=None) -> np.ndarray:
    scores_val = np.asarray(scores_val, dtype=float)
    if grid is not None:
        return np.asarray(grid, dtype=float)
    if len(scores_val) == 0:
        return np.asarray([0.5], dtype=float)
    lo, hi = float(scores_val.min()), float(scores_val.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.asarray([0.5], dtype=float)
    base = np.linspace(lo, hi, 199)
    order = np.sort(scores_val)
    mids = (order[:-1] + order[1:]) / 2 if len(order) > 1 else np.array([], dtype=float)
    return np.unique(np.concatenate([base, mids]))


def tune_threshold_on_val(y_val, scores_val, grid=None) -> tuple[float, float]:
    """Search thresholds maximizing F1 on VALIDATION predictions only.

    Returns (best_threshold, best_f1).
    """
    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val)
    candidates = _threshold_candidates(scores_val, grid=grid)

    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_at_threshold(y_val, scores_val, float(t))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def decision_cost(y_true, scores, threshold: float, cost_fp: float = 1.0, cost_fn: float = 2.0) -> float:
    """Total decision cost = cost_fp * FP + cost_fn * FN (lower is better)."""
    y_true = np.asarray(y_true).astype(int)
    preds = (np.asarray(scores) >= threshold).astype(int)
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return float(cost_fp) * fp + float(cost_fn) * fn


def tune_threshold_cost(
    y_val,
    scores_val,
    cost_fp: float = 1.0,
    cost_fn: float = 2.0,
    grid=None,
) -> tuple[float, float]:
    """Search thresholds minimizing expected decision cost on VAL only.

    Returns (best_threshold, best_cost).
    """
    y_val = np.asarray(y_val)
    scores_val = np.asarray(scores_val)
    candidates = _threshold_candidates(scores_val, grid=grid)
    best_t, best_cost = 0.5, float("inf")
    for t in candidates:
        c = decision_cost(y_val, scores_val, float(t), cost_fp=cost_fp, cost_fn=cost_fn)
        if c < best_cost:
            best_cost, best_t = c, float(t)
    return best_t, best_cost


def tune_threshold_for_precision(
    y_val,
    scores_val,
    min_precision: float = 0.80,
    grid=None,
) -> tuple[float, float]:
    """Lowest threshold on VAL that achieves precision >= min_precision.

    Falls back to the max-precision threshold if the target is unreachable.
    Returns (threshold, achieved_precision).
    """
    y_val = np.asarray(y_val).astype(int)
    scores_val = np.asarray(scores_val)
    candidates = np.sort(_threshold_candidates(scores_val, grid=grid))
    best_t, best_p, best_f1 = 0.5, -1.0, -1.0
    chosen = None
    for t in candidates:
        preds = (scores_val >= float(t)).astype(int)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if prec >= min_precision:
            # among feasible, take lowest thr (highest recall)
            chosen = (float(t), float(prec))
            break
        if prec > best_p or (prec == best_p and f1 > best_f1):
            best_p, best_f1, best_t = prec, f1, float(t)
    if chosen is not None:
        return chosen
    return best_t, best_p


def classification_report(y_true, scores, threshold: float) -> dict:
    """Full metric bundle at a given operating point."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    report = {
        "auprc": auprc(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "threshold": float(threshold),
        "f1_at_threshold": f1_at_threshold(y_true, scores, threshold),
    }
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    report.update(
        {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "base_rate": base_rate,
            "auprc_lift_over_base": report["auprc"] - base_rate,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )
    return report


def tune_threshold_f1(y_val, scores_val) -> float:
    """Return threshold maximizing validation F1."""
    t, _ = tune_threshold_on_val(y_val, scores_val)
    return t


def format_report(name: str, report: dict) -> str:
    auc = "n/a" if report["roc_auc"] is None else f"{report['roc_auc']:.4f}"
    return (
        f"{name}: AUPRC={report['auprc']:.4f} | F1@{report['threshold']:.3f}="
        f"{report['f1_at_threshold']:.4f} (P={report['precision']:.3f} R={report['recall']:.3f}) "
        f"| ROC-AUC(diag)={auc} | TP/FP/FN={report['tp']}/{report['fp']}/{report['fn']}"
    )
