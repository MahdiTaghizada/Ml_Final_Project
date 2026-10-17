"""
Tests (run with `python -m tests.run_tests`, or `pytest` if installed).

scikit-learn appears ONLY here, as the reference implementation.
"""
import numpy as np

from src import data_prep as dp
from src import evaluate as ev
from src.decision_tree import DecisionTree
from src.svm import PegasosSVM, RFFPegasosSVM, RandomFourierFeatures


# ---------------------------------------------------------------- tree --------
def _nodes(tree):
    """[(feature, threshold)] of internal nodes in pre-order, from our tree."""
    out, stack = [], [tree.root]
    while stack:
        n = stack.pop()
        if not n.is_leaf():
            out.append((n.feature, n.threshold))
            stack.extend([n.right, n.left])
    return out


def _sk_nodes(sk):
    t = sk.tree_
    out, stack = [], [0]
    while stack:
        i = stack.pop()
        if t.children_left[i] != -1:
            out.append((int(t.feature[i]), float(t.threshold[i])))
            stack.extend([t.children_right[i], t.children_left[i]])
    return out


def test_tree_matches_sklearn_on_tiny_example():
    """Fully-determined 2-D example: every split is unique, so structures must be identical."""
    from sklearn.tree import DecisionTreeClassifier
    X = np.array([[1, 1], [2, 1], [3, 1], [4, 1],
                  [1, 2], [2, 2], [3, 2], [4, 2],
                  [1, 3], [2, 3], [3, 3], [4, 3]], float)
    y = ((X[:, 0] > 2.5) & (X[:, 1] > 1.5)).astype(int)
    for crit in ("gini", "entropy"):
        ours = DecisionTree("classification", crit).fit(X, y)
        ref = DecisionTreeClassifier(criterion=crit, random_state=0).fit(X, y)
        assert _nodes(ours) == _sk_nodes(ref), (crit, _nodes(ours), _sk_nodes(ref))
        assert (ours.predict(X) == y).all()


def test_tree_regression_tiny_example():
    from sklearn.tree import DecisionTreeRegressor
    X = np.array([[1.], [2.], [3.], [10.], [11.], [12.]])
    y = np.array([1., 1.2, 0.8, 5., 5.4, 4.6])
    ours = DecisionTree("regression", "mse", max_depth=1).fit(X, y)
    ref = DecisionTreeRegressor(max_depth=1).fit(X, y)
    assert _nodes(ours) == _sk_nodes(ref) == [(0, 6.5)]
    assert np.allclose(ours.predict(X), ref.predict(X))


def test_tree_matches_sklearn_on_random_data():
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 5))
    yc = ((X[:, 0] + X[:, 1] ** 2 + 0.3 * rng.normal(size=1500)) > 0.4).astype(int)
    yr = 2 * X[:, 0] + np.sin(X[:, 1]) + 0.1 * rng.normal(size=1500)
    for crit in ("gini", "entropy"):
        a = DecisionTree("classification", crit, max_depth=6, min_samples_leaf=5).fit(X, yc)
        b = DecisionTreeClassifier(criterion=crit, max_depth=6, min_samples_leaf=5, random_state=0).fit(X, yc)
        assert (a.predict(X) == b.predict(X)).mean() > 0.995
    a = DecisionTree("regression", "mse", max_depth=7, min_samples_leaf=3).fit(X, yr)
    b = DecisionTreeRegressor(max_depth=7, min_samples_leaf=3, random_state=0).fit(X, yr)
    assert np.allclose(a.predict(X), b.predict(X), atol=1e-8)


def test_tree_hyperparameters_change_the_tree():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 4))
    y = ((X[:, 0] * X[:, 1] + 0.5 * rng.normal(size=600)) > 0).astype(int)
    full = DecisionTree("classification", "gini").fit(X, y)
    assert DecisionTree("classification", "gini", max_depth=3).fit(X, y).get_depth() <= 3 < full.get_depth()
    assert DecisionTree("classification", "gini", min_samples_split=100).fit(X, y).get_n_leaves() < full.get_n_leaves()
    leaf = DecisionTree("classification", "gini", min_samples_leaf=40).fit(X, y)
    assert leaf.get_n_leaves() < full.get_n_leaves()
    assert min(nd.n_samples for nd in leaf._nodes if nd.is_leaf()) >= 40
    assert DecisionTree("classification", "gini", min_impurity_decrease=0.01).fit(X, y).get_n_leaves() < full.get_n_leaves()


def test_tree_predict_proba_and_importances():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 3))
    y = (X[:, 1] > 0).astype(int)
    t = DecisionTree("classification", "gini", max_depth=3).fit(X, y)
    P = t.predict_proba(X)
    assert P.shape == (500, 2) and np.allclose(P.sum(1), 1)
    assert (t.classes_[P.argmax(1)] == t.predict(X)).all()
    assert abs(t.feature_importances_.sum() - 1) < 1e-9 and t.feature_importances_.argmax() == 1
    # vectorised predict == reference single-sample traversal
    assert all(t._predict_one(X[i], t.root).value == t.predict(X[i:i + 1])[0] for i in range(50))


# ----------------------------------------------------------------- SVM ----------
def _blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-1.5, 1, (n // 2, 2)), rng.normal(1.5, 1, (n // 2, 2))])
    return X, np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)


def test_svm_objective_decreases_and_separates():
    X, y = _blobs()
    m = PegasosSVM(lambda_=1e-2, n_epochs=40, batch_size=32).fit(X, y)
    obj = [o for _, o, _ in m.history_]
    assert obj[-1] < 0.1 * 1.0                      # J(w=0) = 1 (every hinge term equals 1)
    assert obj[-1] <= obj[0] + 1e-9                 # no worse than the first logged value
    first, second = np.mean(obj[:len(obj) // 2]), np.mean(obj[len(obj) // 2:])
    assert second < first                           # decreasing trend over training
    assert m.score(X, y) > 0.9


def test_svm_margin_and_support_vectors():
    X, y = _blobs()
    m = PegasosSVM(lambda_=1e-2, n_epochs=40, batch_size=32).fit(X, y)
    assert abs(m.margin_width_ - 2 / np.linalg.norm(m.w)) < 1e-12
    ypm = np.where(y == 1, 1, -1)
    assert set(m.support_) == set(np.flatnonzero(ypm * m.decision_function(X) <= 1))
    # stronger regularisation -> smaller ||w|| -> wider margin
    wide = PegasosSVM(lambda_=1e-1, n_epochs=40, batch_size=32).fit(X, y)
    assert wide.margin_width_ > m.margin_width_


def test_svm_close_to_sklearn_linear_svm():
    from sklearn.svm import LinearSVC
    X, y = _blobs(2000, 3)
    ours = PegasosSVM(lambda_=1e-3, n_epochs=40, batch_size=64).fit(X, y)
    ref = LinearSVC(C=1 / (1e-3 * len(y))).fit(X, y)
    assert abs(ours.score(X, y) - ref.score(X, y)) < 0.02


def test_rff_solves_nonlinear_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 2))
    y = (np.linalg.norm(X, axis=1) > 1.2).astype(int)           # circle: not linearly separable
    lin = PegasosSVM(lambda_=1e-3, n_epochs=30).fit(X, y).score(X, y)
    rbf = RFFPegasosSVM(gamma=0.5, n_components=200, lambda_=1e-4, n_epochs=30).fit(X, y).score(X, y)
    assert rbf > 0.9 and rbf > lin + 0.15
    Z = RandomFourierFeatures(gamma=0.5, n_components=4000).fit(X[:50])
    K_approx = Z.transform(X[:50]) @ Z.transform(X[:50]).T
    K_true = np.exp(-0.5 * ((X[:50, None] - X[None, :50]) ** 2).sum(-1))
    assert np.abs(K_approx - K_true).max() < 0.1


# ------------------------------------------------------------- metrics ----------
def test_metrics_match_sklearn():
    from sklearn import metrics as sm
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 400)
    sc = np.round(rng.normal(size=400) + yt, 1)                  # rounded -> ties
    yp = (sc > 0.5).astype(int)
    assert abs(ev.roc_auc(yt, sc) - sm.roc_auc_score(yt, sc)) < 1e-12
    assert abs(ev.average_precision(yt, sc) - sm.average_precision_score(yt, sc)) < 1e-12
    p, r, f = ev.precision_recall_f1(yt, yp)
    assert np.allclose([p, r, f], [sm.precision_score(yt, yp), sm.recall_score(yt, yp), sm.f1_score(yt, yp)])
    assert (ev.confusion_matrix(yt, yp) == sm.confusion_matrix(yt, yp)).all()
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert np.allclose([ev.rmse(a, b), ev.mae(a, b), ev.r2(a, b)],
                       [sm.root_mean_squared_error(a, b), sm.mean_absolute_error(a, b), sm.r2_score(a, b)])


# ------------------------------------------------------------ data prep ----------
def test_tier_threshold_and_split_are_leak_free():
    y = np.random.default_rng(0).lognormal(12, 0.5, 5000)
    tr, va, te = dp.split_indices(len(y), seed=1, strata=np.digitize(y, np.quantile(y, np.linspace(0, 1, 11)[1:-1])))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(y)
    _, thr = dp.make_tier_label(y[tr])
    assert thr == np.median(y[tr])
    yt, _ = dp.make_tier_label(y[te], thr)                      # test reuses the TRAIN threshold
    assert abs(yt.mean() - 0.5) < 0.05


def test_scaler_and_imputer_use_train_statistics_only():
    rng = np.random.default_rng(0)
    A, B = rng.normal(0, 1, (200, 3)), rng.normal(50, 1, (100, 3))
    a, b = dp.standardize(A, B)
    assert np.allclose(a.mean(0), 0, atol=1e-9) and b.mean() > 10      # B is NOT re-centred
    A2 = A.copy(); A2[0, 0] = np.nan
    B2 = B.copy(); B2[0, 1] = np.nan
    a2, b2 = dp.impute(A2, B2)
    assert not np.isnan(a2).any() and not np.isnan(b2).any()
    assert abs(b2[0, 1] - np.nanmedian(A2[:, 1])) < 1e-12


def test_no_leakage_columns_in_features():
    for c in dp.LEAKAGE_COLUMNS + ["price", "views"]:
        assert c not in dp.FEATURE_NAMES
