"""
test_bonuses.py - sanity tests for the bonus implementations (Problems 8, 9, 10).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge as SkRidge
from sklearn.decomposition import PCA as SkPCA

from src.bonuses import RidgeRegression, RandomForest, PCA, KMeans


def test_ridge_matches_sklearn():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = X @ np.array([1.5, -2.0, 0.5, 0.0, 3.2]) + 0.5 + 0.1 * rng.randn(100)

    lam = 2.5
    ours = RidgeRegression(lambda_=lam, fit_intercept=True).fit(X, y)
    sk = SkRidge(alpha=lam, fit_intercept=True).fit(X, y)

    np.testing.assert_allclose(ours.coef_, sk.coef_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ours.intercept_, sk.intercept_, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ours.predict(X), sk.predict(X), rtol=1e-5, atol=1e-5)


def test_random_forest_regression_and_classification():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 6)
    y_reg = X[:, 0] * 2.0 - X[:, 1] + rng.randn(80) * 0.1
    y_cls = (y_reg > np.median(y_reg)).astype(float)

    rf_reg = RandomForest(task="regression", n_estimators=5, max_depth=4, random_state=42).fit(X, y_reg)
    preds_reg = rf_reg.predict(X)
    assert preds_reg.shape == (80,)
    assert not np.isnan(preds_reg).any()

    rf_cls = RandomForest(task="classification", n_estimators=5, max_depth=4, random_state=42).fit(X, y_cls)
    preds_cls = rf_cls.predict(X)
    assert preds_cls.shape == (80,)
    assert set(np.unique(preds_cls)).issubset({0.0, 1.0})


def test_pca_matches_sklearn():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4) @ rng.randn(4, 4)

    ours = PCA(n_components=2).fit(X)
    sk = SkPCA(n_components=2).fit(X)

    # Explained variance ratio must match
    np.testing.assert_allclose(ours.explained_variance_ratio_, sk.explained_variance_ratio_, rtol=1e-5, atol=1e-5)


def test_kmeans_clustering():
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 2) + np.array([5.0, 5.0])
    c2 = rng.randn(30, 2) + np.array([-5.0, -5.0])
    X = np.vstack([c1, c2])

    km = KMeans(n_clusters=2, random_state=42).fit(X)
    labels = km.predict(X)
    assert len(np.unique(labels)) == 2
    # Samples in c1 should have different label from samples in c2
    assert len(np.unique(labels[:30])) == 1
    assert len(np.unique(labels[30:])) == 1
    assert labels[0] != labels[30]
