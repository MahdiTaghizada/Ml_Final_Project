"""
svm.py - a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

Primal objective (labels y_i in {-1,+1}):

    J(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum_i max(0, 1 - y_i (w.x_i + b))

Pegasos = (mini-batch) sub-gradient descent with eta_t = 1/(lambda * t):

    w <- (1 - eta_t*lambda) w + (eta_t/k) * sum_{i in batch, y_i w.x_i < 1} y_i x_i

Bias: we fold it into the weight vector by appending a constant feature
`bias_scale` to every x (the trick suggested in the brief):  w~ = [w ; b/bias_scale].
This regularises b only weakly when bias_scale is large; the tracked objective
is the one actually minimised,
    J~ = lambda/2 (||w||^2 + (b/bias_scale)^2) + mean hinge,
which equals J as bias_scale -> infinity.  Margin width is reported as 2/||w||.

Non-linearity (option ii of the brief): `RandomFourierFeatures` maps x to
phi(x) = sqrt(2/D) cos(W x + c) with W ~ N(0, 2*gamma*I), c ~ U[0, 2*pi], so that
phi(x).phi(y) ~= exp(-gamma ||x-y||^2) (RBF kernel, Rahimi & Recht 2007);
`RFFPegasosSVM` then runs the same linear solver in that space.  We chose RFF
over kernelised Pegasos because the latter needs an n x n kernel (40k^2 = 1.6e9
entries) and O(n) work per prediction, while RFF keeps O(n D) training.

scikit-learn's SVC / SGDClassifier are BASELINES only - never used in this file.
Features MUST be standardised first (see data_prep.standardize).
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,        # regularisation strength  (lambda ~= 1/(n*C))
        n_epochs: int = 20,           # passes over the training data
        batch_size: int | None = 64,  # None -> full-batch sub-gradient descent
        lr_schedule: str = "pegasos", # "pegasos": 1/(lambda t) | "invscaling": eta0/sqrt(t) | "constant": eta0
        eta0: float = 0.1,            # used by "invscaling" / "constant"
        project: bool = True,         # Pegasos ball projection  ||w~|| <= 1/sqrt(lambda)
        bias_scale: float = 1.0,
        average: bool = True,         # suffix averaging: final w = mean of the iterates of the last half

        n_iters: int | None = None,   # if given, overrides n_epochs (total number of steps)
        n_log: int = 100,             # number of objective evaluations kept in history
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.eta0 = eta0
        self.project = project
        self.bias_scale = bias_scale
        self.average = average
        self.n_iters = n_iters
        self.n_log = n_log
        self.random_state = random_state
        self.w = None              # weight vector (original feature space)
        self.b = 0.0               # bias

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def _to_pm1(y):
        """Map any 2-class labels to {-1,+1}; returns (y_pm1, classes) with classes[1] = +1."""
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("PegasosSVM is a binary classifier")
        return np.where(np.asarray(y) == classes[1], 1.0, -1.0), classes

    def _augment(self, X):
        return np.hstack([X, np.full((len(X), 1), self.bias_scale)])

    def _eta(self, t: int) -> float:
        if self.lr_schedule == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.lr_schedule == "invscaling":
            return self.eta0 / np.sqrt(t)
        if self.lr_schedule == "constant":
            return self.eta0
        raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")

    def _objective_aug(self, Xa, y, wt) -> float:
        hinge = np.maximum(0.0, 1.0 - y * (Xa @ wt)).mean()
        return float(0.5 * self.lambda_ * (wt @ wt) + hinge)

    # --- training ---------------------------------------------------------------
    def fit(self, X, y):
        """Run (mini-batch) Pegasos on the regularised hinge-loss primal. Returns self."""
        X = np.asarray(X, dtype=float)
        ypm, self.classes_ = self._to_pm1(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        Xa = self._augment(X)
        wt = np.zeros(d + 1)

        k = n if self.batch_size is None else min(self.batch_size, n)
        steps_per_epoch = int(np.ceil(n / k))
        T = self.n_iters if self.n_iters is not None else self.n_epochs * steps_per_epoch
        log_every = max(1, T // self.n_log)
        self.history_ = []            # (step, objective, mean hinge loss)
        t = 0
        radius = 1.0 / np.sqrt(self.lambda_)
        w_sum, n_avg = np.zeros(d + 1), 0          # running sum for suffix averaging
        while t < T:
            order = rng.permutation(n) if k < n else np.arange(n)
            for s in range(0, n, k):
                if t >= T:
                    break
                t += 1
                bi = order[s:s + k]
                xb, yb = Xa[bi], ypm[bi]
                eta = self._eta(t)
                viol = yb * (xb @ wt) < 1.0                     # hinge active
                grad_part = (yb[viol, None] * xb[viol]).sum(axis=0) / len(bi)
                wt = (1.0 - eta * self.lambda_) * wt + eta * grad_part
                if self.project:
                    norm = np.linalg.norm(wt)
                    if norm > radius:
                        wt *= radius / norm
                if self.average and t > T // 2:
                    w_sum += wt
                    n_avg += 1
                if t % log_every == 0 or t == T:
                    hinge = np.maximum(0.0, 1.0 - ypm * (Xa @ wt)).mean()
                    self.history_.append((t, float(0.5 * self.lambda_ * wt @ wt + hinge), float(hinge)))
        self.n_steps_ = t
        self.last_iterate_objective_ = self._objective_aug(Xa, ypm, wt)
        if self.average and n_avg > 0:
            wt = w_sum / n_avg          # averaged iterate: far less noisy than the last one
        self.w = wt[:-1].copy()
        self.b = float(wt[-1] * self.bias_scale)
        self._wt = wt
        self.objective_ = self._objective_aug(Xa, ypm, wt)
        # geometry of the learned classifier
        self.w_norm_ = float(np.linalg.norm(self.w))
        self.margin_width_ = 2.0 / self.w_norm_ if self.w_norm_ > 0 else np.inf
        self.support_ = np.flatnonzero(ypm * (X @ self.w + self.b) <= 1.0)   # y(w.x+b) <= 1
        return self

    # --- inference ------------------------------------------------------------------
    def decision_function(self, X):
        """Signed margin  w.x + b  for each row."""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        """Class labels (sign of the margin, mapped back to the original labels)."""
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def objective(self, X, y) -> float:
        """J(w,b) as in the brief (||w||^2 only, b unregularised) on any dataset."""
        ypm, _ = self._to_pm1(y)
        hinge = np.maximum(0.0, 1.0 - ypm * self.decision_function(X)).mean()
        return float(0.5 * self.lambda_ * (self.w @ self.w) + hinge)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class RandomFourierFeatures:
    """phi(x) = sqrt(2/D) cos(Wx + c)  approximating the RBF kernel exp(-gamma ||x-y||^2)."""

    def __init__(self, gamma: float = 0.05, n_components: int = 300, random_state: int = 42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        d = np.asarray(X).shape[1]
        self.W_ = rng.normal(0.0, np.sqrt(2.0 * self.gamma), size=(d, self.n_components))
        self.c_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        return np.sqrt(2.0 / self.n_components) * np.cos(np.asarray(X, dtype=float) @ self.W_ + self.c_)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class RFFPegasosSVM:
    """Non-linear SVM = RandomFourierFeatures  ->  linear PegasosSVM."""

    def __init__(self, gamma: float = 0.05, n_components: int = 300, **svm_kwargs):
        self.gamma = gamma
        self.n_components = n_components
        self.svm_kwargs = svm_kwargs
        self.random_state = svm_kwargs.get("random_state", 42)

    def fit(self, X, y):
        self.rff_ = RandomFourierFeatures(self.gamma, self.n_components, self.random_state).fit(X)
        self.svm_ = PegasosSVM(**self.svm_kwargs).fit(self.rff_.transform(X), y)
        self.classes_ = self.svm_.classes_
        self.history_ = self.svm_.history_
        self.margin_width_ = self.svm_.margin_width_
        self.w_norm_ = self.svm_.w_norm_
        self.support_ = self.svm_.support_
        self.objective_ = self.svm_.objective_
        return self

    def decision_function(self, X):
        return self.svm_.decision_function(self.rff_.transform(X))

    def predict(self, X):
        return self.svm_.predict(self.rff_.transform(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
