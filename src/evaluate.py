"""
evaluate.py - metrics + comparison helpers, implemented FROM SCRATCH in NumPy.

scikit-learn is used only in the tests (to sanity-check these numbers) and for
the baseline models in run_all.py.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) ---------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_report(y_true_log, y_pred_log) -> dict:
    """Metrics on log(price) AND on the AZN scale (back-transformed with exp)."""
    yt, yp = np.exp(y_true_log), np.exp(y_pred_log)
    return {"rmse_log": rmse(y_true_log, y_pred_log), "mae_log": mae(y_true_log, y_pred_log),
            "r2_log": r2(y_true_log, y_pred_log),
            "rmse_azn": rmse(yt, yp), "mae_azn": mae(yt, yp), "r2_azn": r2(yt, yp)}


# --- Classification (Task B: price tier) --------------------------------------
def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """2x2 matrix [[TN, FP], [FN, TP]] (positive class = premium = 1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """(precision, recall, f1) of the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(prec), float(rec), float(f1)


def _average_ranks(x):
    """1-based ranks with ties replaced by their mean rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    xs = x[order]
    boundaries = np.flatnonzero(np.diff(xs)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(x)]])
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = 0.5 * (s + 1 + e)
    return ranks


def roc_auc(y_true, scores) -> float:
    """AUC via the Mann-Whitney U statistic (ties handled by mean ranks)."""
    y_true = np.asarray(y_true)
    r = _average_ranks(scores)
    n_pos = int((y_true == 1).sum())
    n_neg = len(y_true) - n_pos
    u = r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """(fpr, tpr) over all distinct thresholds, for plotting."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")
    y = y_true[order]
    tps, fps = np.cumsum(y == 1), np.cumsum(y == 0)
    distinct = np.r_[np.flatnonzero(np.diff(np.asarray(scores)[order])), len(y) - 1]
    tpr = np.r_[0, tps[distinct] / max(tps[-1], 1)]
    fpr = np.r_[0, fps[distinct] / max(fps[-1], 1)]
    return fpr, tpr


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: sum_k (R_k - R_{k-1}) P_k."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")
    y = y_true[order]
    s = np.asarray(scores)[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(y) - 1]
    tps = np.cumsum(y == 1)[distinct]
    prec = tps / (distinct + 1)
    rec = tps / max(int((y_true == 1).sum()), 1)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def classification_report(y_true, y_pred, scores) -> dict:
    p, r, f = precision_recall_f1(y_true, y_pred)
    return {"accuracy": accuracy(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "roc_auc": roc_auc(y_true, scores), "pr_auc": average_precision(y_true, scores)}
