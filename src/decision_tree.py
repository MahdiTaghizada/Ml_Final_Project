"""
decision_tree.py - a decision tree built FROM SCRATCH (NumPy only).

Supports both project tasks with one class:
  * classification: criterion = "gini" | "entropy"   (price tier, Task B)
  * regression:     criterion = "mse"                (price / log-price, Task A)

Maths (node t with N_t samples, children L and R):
    Gini     G(t) = 1 - sum_k p_k^2
    Entropy  H(t) = - sum_k p_k log2 p_k
    MSE      I(t) = 1/N_t sum_i (y_i - ybar)^2
    gain     = I(t) - (N_L/N_t) I(L) - (N_R/N_t) I(R)        (information gain)

The split search sorts each feature once per node and evaluates EVERY
midpoint between consecutive distinct values with cumulative sums, so one
node costs O(d * n log n).  We follow scikit-learn's conventions so that the
two implementations are comparable:
  * a split sends  x[f] <= threshold  to the left child, threshold = midpoint;
  * min_impurity_decrease uses the same *weighted* decrease as sklearn:
        (N_t / N) * ( I(t) - N_L/N_t I(L) - N_R/N_t I(R) )  >=  min_impurity_decrease;
  * feature_importances_ = total weighted impurity decrease per feature, normalised.
Differences that remain (tie-breaking, float32 casting) are listed in the report.

scikit-learn is NOT used anywhere in this file.
"""

from __future__ import annotations

import sys

import numpy as np


# --- impurity functions (take class COUNTS or a y vector) --------------------
def gini_from_counts(counts: np.ndarray) -> np.ndarray:
    """Gini impurity; `counts` has shape (..., K)."""
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(tot, 1)
    return 1.0 - (p ** 2).sum(axis=-1)


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits; `counts` has shape (..., K); 0*log(0) := 0."""
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log2(p), 0.0)
    return -plogp.sum(axis=-1)


def mse_of(y: np.ndarray) -> float:
    """Variance (= MSE around the mean) of a vector."""
    return float(np.mean((y - y.mean()) ** 2)) if len(y) else 0.0


class Node:
    """One node of the tree."""

    def __init__(self):
        self.feature = None        # int: feature index to split on (internal nodes)
        self.threshold = None      # float: go left if x[feature] <= threshold
        self.left = None           # Node
        self.right = None          # Node
        self.value = None          # leaf prediction (class label or mean)
        self.proba = None          # class frequencies at this node (classification)
        self.n_samples = 0
        self.impurity = 0.0
        self.node_id = -1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,         # kept for API symmetry; the search is deterministic
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if task == "classification" and criterion not in ("gini", "entropy"):
            raise ValueError("classification criterion must be 'gini' or 'entropy'")
        if task == "regression" and criterion != "mse":
            raise ValueError("regression criterion must be 'mse'")
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity ------------------------------------------------------------
    def _impurity(self, y) -> float:
        """Impurity of a label/target vector under the chosen criterion."""
        y = np.asarray(y)
        if self.task == "regression":
            return mse_of(y.astype(float))
        classes = getattr(self, "classes_", np.unique(y))
        counts = np.array([(y == c).sum() for c in classes], dtype=float)
        fn = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        return float(fn(counts))

    def _node_impurity(self, idx) -> float:
        if self.task == "regression":
            return mse_of(self._y[idx])
        counts = self._Y[idx].sum(axis=0)
        fn = gini_from_counts if self.criterion == "gini" else entropy_from_counts
        return float(fn(counts))

    # --- split search ----------------------------------------------------------
    def _best_split(self, X, y, idx, parent_imp):
        """
        Search all features and ALL candidate thresholds (midpoints of consecutive
        distinct sorted values) for the split with the largest impurity decrease.
        `y` is unused (kept for the starter signature); labels live in self._Y / self._y.
        Returns (feature, threshold, decrease) or None if no valid split exists.
        """
        n = len(idx)
        msl = self.min_samples_leaf
        n_left = np.arange(1, n)                     # size of the left child at each cut
        size_ok = (n_left >= msl) & (n - n_left >= msl)
        if not size_ok.any():
            return None

        if self.task == "classification":
            Yn = self._Y[idx]
            imp_fn = gini_from_counts if self.criterion == "gini" else entropy_from_counts
            total = Yn.sum(axis=0)
        else:
            yn = self._y[idx]
            yc = yn - yn.mean()                      # centre for numerical stability
            yc2 = yc ** 2
            total_s, total_s2 = yc.sum(), yc2.sum()

        best_gain, best = -np.inf, None
        for f in range(X.shape[1]):
            xs = X[idx, f]
            order = np.argsort(xs, kind="stable")
            xs_s = xs[order]
            valid = size_ok & (xs_s[:-1] < xs_s[1:])  # cut only between distinct values
            if not valid.any():
                continue
            if self.task == "classification":
                left = np.cumsum(Yn[order], axis=0)[:-1]            # (n-1, K)
                right = total - left
                i_l, i_r = imp_fn(left), imp_fn(right)
            else:
                cs = np.cumsum(yc[order])[:-1]
                cs2 = np.cumsum(yc2[order])[:-1]
                nl = n_left.astype(float)
                nr = n - nl
                i_l = cs2 / nl - (cs / nl) ** 2
                i_r = (total_s2 - cs2) / nr - ((total_s - cs) / nr) ** 2
            child_imp = (n_left * i_l + (n - n_left) * i_r) / n
            gain = np.where(valid, parent_imp - child_imp, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:                  # strict '>' -> lowest feature index wins ties
                thr = 0.5 * (xs_s[k] + xs_s[k + 1])
                if thr >= xs_s[k + 1]:               # float guard (adjacent floats)
                    thr = xs_s[k]
                best_gain, best = float(gain[k]), (f, float(thr))
        if best is None:
            return None
        return best[0], best[1], best_gain
