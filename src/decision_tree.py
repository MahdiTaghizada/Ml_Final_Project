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

    # --- grow / predict ----------------------------------------------------------
    def _make_leaf(self, node: Node, idx):
        if self.task == "classification":
            counts = self._Y[idx].sum(axis=0)
            node.proba = counts / counts.sum()
            node.value = self.classes_[int(np.argmax(counts))]
        else:
            node.value = float(self._y[idx].mean())

    def _grow(self, X, idx, depth: int) -> Node:
        """Recursively build the tree, respecting every stopping rule."""
        node = Node()
        node.n_samples = len(idx)
        node.impurity = self._node_impurity(idx)
        self._make_leaf(node, idx)          # every node stores a prediction (needed for leaves)

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or node.n_samples < self.min_samples_split
            or node.n_samples < 2 * self.min_samples_leaf
            or node.impurity <= 1e-12
        )
        if stop:
            return node
        found = self._best_split(X, None, idx, node.impurity)
        if found is None:
            return node
        f, thr, gain = found
        weighted_decrease = gain * node.n_samples / self._n_total
        if weighted_decrease < self.min_impurity_decrease:
            return node
        mask = X[idx, f] <= thr
        node.feature, node.threshold = f, thr
        self._importances[f] += weighted_decrease
        node.left = self._grow(X, idx[mask], depth + 1)
        node.right = self._grow(X, idx[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if np.isnan(X).any():
            raise ValueError("X contains NaN - impute first (data_prep.impute)")
        self.n_features_ = X.shape[1]
        self._n_total = len(y)
        self._importances = np.zeros(self.n_features_)
        if self.task == "classification":
            self.classes_, y_enc = np.unique(y, return_inverse=True)
            self._Y = np.eye(len(self.classes_))[y_enc]           # one-hot (N, K)
        else:
            self._y = y.astype(float)
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 20000))
        self.root = self._grow(X, np.arange(len(y)), depth=0)
        sys.setrecursionlimit(old)
        # free training-only buffers, flatten for fast vectorised prediction
        self._Y = self._y = None
        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        self._flatten()
        return self

    def _flatten(self):
        """Store the tree in parallel arrays so predict() can be vectorised."""
        nodes, stack = [], [self.root]
        while stack:
            nd = stack.pop()
            nd.node_id = len(nodes)
            nodes.append(nd)
            if not nd.is_leaf():
                stack.extend([nd.right, nd.left])     # left visited first
        m = len(nodes)
        self._feat = np.full(m, -1, dtype=int)
        self._thr = np.zeros(m)
        self._left = np.full(m, -1, dtype=int)
        self._right = np.full(m, -1, dtype=int)
        self._depth = np.zeros(m, dtype=int)
        for nd in nodes:
            if not nd.is_leaf():
                i = nd.node_id
                self._feat[i], self._thr[i] = nd.feature, nd.threshold
                self._left[i], self._right[i] = nd.left.node_id, nd.right.node_id
                self._depth[nd.left.node_id] = self._depth[nd.right.node_id] = self._depth[i] + 1
        self._nodes = nodes

    # --- prediction --------------------------------------------------------------
    def apply(self, X) -> np.ndarray:
        """Leaf id reached by each row (vectorised level-by-level traversal)."""
        X = np.asarray(X, dtype=float)
        cur = np.zeros(len(X), dtype=int)
        active = self._feat[cur] >= 0
        while active.any():
            rows = np.flatnonzero(active)
            nodes = cur[rows]
            go_left = X[rows, self._feat[nodes]] <= self._thr[nodes]
            cur[rows] = np.where(go_left, self._left[nodes], self._right[nodes])
            active = self._feat[cur] >= 0
        return cur

    def _predict_one(self, x, node: Node):
        """Reference (slow) traversal of a single sample - used in tests."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        leaves = self.apply(X)
        if self.task == "classification":
            vals = np.array([nd.value for nd in self._nodes])
            return vals[leaves]
        vals = np.array([nd.value for nd in self._nodes], dtype=float)
        return vals[leaves]

    def predict_proba(self, X):
        """Classification only: class frequencies of the leaf each row lands in."""
        if self.task != "classification":
            raise AttributeError("predict_proba is only defined for classification")
        P = np.array([nd.proba if nd.proba is not None else np.zeros(len(self.classes_))
                      for nd in self._nodes])
        return P[self.apply(X)]

    def score(self, X, y):
        """Accuracy (classification) or R^2 (regression)."""
        pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(pred == np.asarray(y)))
        y = np.asarray(y, dtype=float)
        return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    # --- introspection -------------------------------------------------------------
    def get_depth(self) -> int:
        return int(self._depth.max())

    def get_n_leaves(self) -> int:
        return int((self._feat < 0).sum())

    def node_count(self) -> int:
        return len(self._nodes)

    def export_text(self, feature_names=None, max_depth: int = 4) -> str:
        """Readable dump of the top of the tree."""
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def rec(nd, depth):
            pad = "|   " * depth
            if nd.is_leaf() or depth > max_depth:
                lines.append(f"{pad}|--- value: {nd.value if self.task == 'regression' else nd.value}"
                             f"  (n={nd.n_samples})")
                return
            lines.append(f"{pad}|--- {names[nd.feature]} <= {nd.threshold:.4g}")
            rec(nd.left, depth + 1)
            lines.append(f"{pad}|--- {names[nd.feature]} >  {nd.threshold:.4g}")
            rec(nd.right, depth + 1)

        rec(self.root, 0)
        return "\n".join(lines)
