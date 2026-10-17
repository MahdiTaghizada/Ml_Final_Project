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


# --- Resampling (uncertainty) ---------------------------------------------------
def kfold_indices(n, k=5, seed=42, strata=None):
    """List of (train_idx, test_idx) for k-fold CV (stratified if `strata` given)."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(n, dtype=int)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(strata == g) for g in np.unique(strata)]
    for idx in groups:
        idx = rng.permutation(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def cross_validate(fit_eval, n, k=5, seed=42, strata=None) -> dict:
    """
    fit_eval(train_idx, test_idx) -> dict of metrics.  Returns {metric: (mean, std)}.
    Preprocessing statistics must be computed INSIDE fit_eval, from train_idx only.
    """
    rows = [fit_eval(tr, te) for tr, te in kfold_indices(n, k, seed, strata)]
    return {m: (float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows], ddof=1)))
            for m in rows[0]}


def bootstrap_ci(metric_fn, *arrays, n_boot=200, seed=42, alpha=0.05):
    """Percentile bootstrap CI of metric_fn(*arrays) over test-set resamples."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = [metric_fn(*[a[i] for a in arrays]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --- Comparison helper ------------------------------------------------------------
def compare(name_to_metrics: dict, title: str = "") -> str:
    """
    Pretty-print a table: rows = models, columns = metrics.  Values may be floats
    or (mean, std) tuples.  Returns the string as well.
    """
    cols = []
    for m in name_to_metrics.values():
        for c in m:
            if c not in cols:
                cols.append(c)

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, tuple):
            return f"{v[0]:.4g} +/- {v[1]:.2g}"
        return f"{v:.4g}" if isinstance(v, (int, float, np.floating)) else str(v)

    width = max(len(n) for n in name_to_metrics) + 2
    cw = {c: max(len(c), *(len(fmt(m.get(c))) for m in name_to_metrics.values())) + 2 for c in cols}
    lines = [title] if title else []
    lines.append("model".ljust(width) + "".join(c.rjust(cw[c]) for c in cols))
    lines.append("-" * (width + sum(cw.values())))
    for name, m in name_to_metrics.items():
        lines.append(name.ljust(width) + "".join(fmt(m.get(c)).rjust(cw[c]) for c in cols))
    out = "\n".join(lines)
    print(out)
    return out
