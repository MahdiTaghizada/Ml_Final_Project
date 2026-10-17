"""
bonuses.py - Optional bonus implementations (Problems 8, 9, 10).

Implemented FROM SCRATCH in NumPy:
  * Problem 8: Linear / Ridge regression via the normal equations
        w_hat = (X^T X + lambda * I)^(-1) X^T y
    as a Task A baseline, compared against DecisionTree and sklearn Ridge.
  * Problem 9: Ensembles from our own DecisionTree:
        RandomForestRegressor and RandomForestClassifier
    with bootstrap aggregation (bagging) and random feature subsampling.
  * Problem 10: Unsupervised & geospatial EDA:
        PCA (SVD-based) and K-Means clustering on coordinates (lat, lng)
    to discover geographic market segments and sharpen residual error analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.decision_tree import DecisionTree
from src import evaluate as ev


# ============================================================================
# Problem 8: From-Scratch Ridge Regression (Normal Equations)
# ============================================================================
class RidgeRegression:
    """
    Linear Ridge Regression solved via normal equations:
        w_hat = (X_c^T X_c + lambda * I)^(-1) X_c^T y_c
    where X_c and y_c are centered by subtracting column means.
    """
    def __init__(self, lambda_: float = 1.0, fit_intercept: bool = True):
        self.lambda_ = float(lambda_)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.x_mean_ = None
        self.y_mean_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        if self.fit_intercept:
            self.x_mean_ = X.mean(axis=0)
            self.y_mean_ = float(y.mean())
            Xc = X - self.x_mean_
            yc = y - self.y_mean_
            A = Xc.T @ Xc + self.lambda_ * np.eye(d)
            self.coef_ = np.linalg.solve(A, Xc.T @ yc)
            self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        else:
            A = X.T @ X + self.lambda_ * np.eye(d)
            self.coef_ = np.linalg.solve(A, X.T @ y)
            self.intercept_ = 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


# ============================================================================
# Problem 9: From-Scratch Random Forest Ensemble from Own Decision Trees
# ============================================================================
class RandomForest:
    """
    Random Forest ensemble built using our from-scratch DecisionTree as weak learners.
    Features:
      * Bootstrap sampling of instances (with replacement).
      * Random feature subsampling (sqrt(d) for classification, d/3 for regression).
      * Aggregates predictions: average for regression, majority vote for classification.
    """
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 10,
        max_depth: int | None = 10,
        min_samples_leaf: int = 5,
        max_features: str | int | float = "auto",
        random_state: int = 42,
    ):
        self.task = task
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTree] = []
        self.feature_subsets_: list[np.ndarray] = []

    def _resolve_max_features(self, d: int) -> int:
        if self.max_features == "auto" or self.max_features == "sqrt":
            return max(1, int(np.sqrt(d))) if self.task == "classification" else max(1, d // 3)
        elif self.max_features == "third":
            return max(1, d // 3)
        elif isinstance(self.max_features, int):
            return min(d, max(1, self.max_features))
        elif isinstance(self.max_features, float):
            return min(d, max(1, int(self.max_features * d)))
        return d

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        m = self._resolve_max_features(d)
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        self.feature_subsets_ = []
        crit = "mse" if self.task == "regression" else "gini"

        for i in range(self.n_estimators):
            # 1. Bootstrap sample (with replacement)
            boot_idx = rng.choice(n, size=n, replace=True)
            # 2. Random feature subset (without replacement)
            feat_idx = rng.choice(d, size=m, replace=False)

            X_b = X[boot_idx][:, feat_idx]
            y_b = y[boot_idx]

            tree = DecisionTree(
                task=self.task,
                criterion=crit,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_b, y_b)
            self.estimators_.append(tree)
            self.feature_subsets_.append(feat_idx)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = np.zeros((len(self.estimators_), len(X)), dtype=float)
        for i, (tree, feat_idx) in enumerate(zip(self.estimators_, self.feature_subsets_)):
            preds[i] = tree.predict(X[:, feat_idx])

        if self.task == "regression":
            return preds.mean(axis=0)
        else:
            return (preds.mean(axis=0) >= 0.5).astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns empirical probability of class 1."""
        X = np.asarray(X, dtype=float)
        preds = np.zeros((len(self.estimators_), len(X)), dtype=float)
        for i, (tree, feat_idx) in enumerate(zip(self.estimators_, self.feature_subsets_)):
            preds[i] = tree.predict(X[:, feat_idx])
        p1 = preds.mean(axis=0)
        return np.column_stack([1.0 - p1, p1])


# ============================================================================
# Problem 10: Unsupervised & Geospatial EDA (PCA & K-Means on Lat/Lng)
# ============================================================================
class PCA:
    """
    Principal Component Analysis (PCA) via SVD:
        X_c = U * S * V^T
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray) -> PCA:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        n = X.shape[0]

        # SVD on centered matrix
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        var = (S ** 2) / (n - 1)
        self.explained_variance_ratio_ = var[:self.n_components] / var.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class KMeans:
    """
    K-Means clustering in pure NumPy with random initialization.
    """
    def __init__(self, n_clusters: int = 5, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        # Random sample initialization
        init_idx = rng.choice(n, size=self.n_clusters, replace=False)
        self.cluster_centers_ = X[init_idx].copy()

        for _ in range(self.max_iter):
            # Euclidean distance to each centroid
            dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            if np.allclose(self.cluster_centers_, new_centers, atol=1e-5):
                break
            self.cluster_centers_ = new_centers

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)
