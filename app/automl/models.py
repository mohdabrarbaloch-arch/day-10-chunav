"""Learners, implemented from scratch on numpy.

No scikit-learn, no xgboost, no autograd. Each estimator exposes the same
tiny interface so the search loop can treat them interchangeably::

    model.fit(X, y)
    model.predict(X)          -> labels (classification) or values (regression)
    model.predict_proba(X)    -> (n, n_classes)  [classifiers only]

Everything is deterministic given a seed. That matters more than it sounds:
a leaderboard that reshuffles between runs is not a leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "StandardScaler",
    "OneHotEncoder",
    "LogisticRegression",
    "RidgeRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "KNN",
]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


class StandardScaler:
    """Zero-mean, unit-variance scaling. Constant columns map to 0, not NaN."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A column that never varies carries no signal; dividing by its zero
        # std would poison the whole matrix with NaN.
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler: call fit before transform")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class OneHotEncoder:
    """One-hot encode integer-coded categorical columns.

    Unseen categories at transform time become all-zeros rather than raising —
    a new city in the test split should not crash a training run.
    """

    def __init__(self) -> None:
        self.categories_: list[np.ndarray] = []
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> OneHotEncoder:
        X = np.asarray(X)
        self.n_features_ = X.shape[1]
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        blocks = []
        for j in range(self.n_features_):
            cats = self.categories_[j]
            block = np.zeros((X.shape[0], len(cats)), dtype=float)
            index = {c: i for i, c in enumerate(cats)}
            for row, value in enumerate(X[:, j]):
                if value in index:
                    block[row, index[value]] = 1.0
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.zeros((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# Linear models
# --------------------------------------------------------------------------


@dataclass
class LogisticRegression:
    """Multinomial logistic regression trained by gradient descent.

    Softmax over ``W @ x + b``, L2 penalty on the weights (not the bias), and
    a small momentum term so it converges in a few hundred iterations without
    a learning-rate schedule.
    """

    learning_rate: float = 0.35
    epochs: int = 400
    l2: float = 1e-3
    seed: int = 0

    W_: np.ndarray | None = field(default=None, init=False, repr=False)
    b_: np.ndarray | None = field(default=None, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)

        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        index = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((n, k))
        for row, label in enumerate(y):
            Y[row, index[label]] = 1.0

        self.W_ = rng.normal(0.0, 0.01, size=(d, k))
        self.b_ = np.zeros(k)
        vW = np.zeros_like(self.W_)
        vb = np.zeros_like(self.b_)
        momentum = 0.9

        for _ in range(self.epochs):
            logits = X @ self.W_ + self.b_
            logits -= logits.max(axis=1, keepdims=True)  # numerical stability
            exp = np.exp(logits)
            P = exp / exp.sum(axis=1, keepdims=True)

            grad_W = X.T @ (P - Y) / n + self.l2 * self.W_
            grad_b = (P - Y).sum(axis=0) / n

            vW = momentum * vW - self.learning_rate * grad_W
            vb = momentum * vb - self.learning_rate * grad_b
            self.W_ += vW
            self.b_ += vb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("LogisticRegression: call fit first")
        logits = np.asarray(X, dtype=float) @ self.W_ + self.b_
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@dataclass
class RidgeRegression:
    """Closed-form ridge regression: ``(XᵀX + λI)⁻¹ Xᵀy``.

    Solved with ``np.linalg.solve`` rather than an explicit inverse — same
    answer, better conditioned. The intercept is not penalised.
    """

    alpha: float = 1.0

    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    intercept_: float = field(default=0.0, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        self.intercept_ = float(y.mean())
        Xc = X - X.mean(axis=0)
        yc = y - self.intercept_

        A = Xc.T @ Xc + self.alpha * np.eye(d)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeRegression: call fit first")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------


class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "proba")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _TreeNode | None = None
        self.right: _TreeNode | None = None
        self.value: float | None = None
        self.proba: np.ndarray | None = None


@dataclass
class DecisionTree:
    """CART tree with gini impurity (classification) or variance (regression).

    Splits are chosen by scanning candidate thresholds at quantiles of each
    feature — a full scan is O(n·d) per node and gets slow on wide data, and
    the quantile grid loses almost nothing in practice.
    """

    max_depth: int = 6
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    n_thresholds: int = 16
    task: str = "classification"  # or "regression"
    seed: int = 0

    root_: _TreeNode | None = field(default=None, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_features_: int = field(default=0, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTree:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, depth=0)
        return self

    # -- internals ---------------------------------------------------------

    def _leaf(self, y: np.ndarray) -> _TreeNode:
        node = _TreeNode()
        if self.task == "classification":
            counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
            node.proba = counts / counts.sum()
            # Keep the label as-is. Coercing it to float works for 0/1 targets
            # and explodes on string labels like "setosa" — the class label is
            # not a number and must not be treated as one.
            node.value = self.classes_[int(np.argmax(counts))]
        else:
            node.value = float(np.mean(y))
        return node

    def _impurity(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        if self.task == "classification":
            _, counts = np.unique(y, return_counts=True)
            p = counts / counts.sum()
            return float(1.0 - np.sum(p**2))  # gini
        return float(np.var(y))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        parent = self._impurity(y)
        best_gain, best = 0.0, None

        for feature in range(d):
            column = X[:, feature]
            if np.all(column == column[0]):
                continue
            candidates = np.unique(np.quantile(column, np.linspace(0.05, 0.95, self.n_thresholds)))
            for threshold in candidates:
                mask = column <= threshold
                n_left = int(mask.sum())
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                weighted = n_left / n * self._impurity(y[mask]) + n_right / n * self._impurity(
                    y[~mask]
                )
                gain = parent - weighted
                if gain > best_gain:
                    best_gain, best = gain, (feature, float(threshold), mask)
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        if depth >= self.max_depth or y.size < self.min_samples_split or self._impurity(y) <= 1e-12:
            return self._leaf(y)

        split = self._best_split(X, y)
        if split is None:
            return self._leaf(y)

        feature, threshold, mask = split
        node = _TreeNode()
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _descend(self, row: np.ndarray) -> _TreeNode:
        node = self.root_
        while node is not None and node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    # -- public ------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTree: call fit first")
        X = np.asarray(X, dtype=float)
        values = [self._descend(row).value for row in X]
        if self.task == "classification":
            # dtype=object preserves string labels; a float array would
            # silently stringify them back into numbers.
            return np.array(values, dtype=object)
        return np.array(values, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification")
        X = np.asarray(X, dtype=float)
        return np.vstack([self._descend(row).proba for row in X])


@dataclass
class RandomForest:
    """Bagged trees over bootstrap samples and a random feature subset.

    The two sources of randomness are what make the ensemble better than any
    single tree; without them the trees are near-identical and averaging buys
    nothing.
    """

    n_estimators: int = 12
    max_depth: int = 7
    min_samples_leaf: int = 2
    max_features: float = 0.7
    task: str = "classification"
    seed: int = 0

    trees_: list[DecisionTree] = field(default_factory=list, init=False, repr=False)
    feature_sets_: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        if self.task == "classification":
            self.classes_ = np.unique(y)

        k = max(1, int(round(self.max_features * d)))
        self.trees_, self.feature_sets_ = [], []

        for _ in range(self.n_estimators):
            rows = rng.integers(0, n, size=n)
            cols = rng.choice(d, size=k, replace=False)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                task=self.task,
                seed=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[np.ix_(rows, cols)], y[rows])
            self.trees_.append(tree)
            self.feature_sets_.append(cols)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("RandomForest: call fit first")
        X = np.asarray(X, dtype=float)
        total = np.zeros((X.shape[0], len(self.classes_)))
        for tree, cols in zip(self.trees_, self.feature_sets_, strict=False):
            total += tree.predict_proba(X[:, cols])
        return total / len(self.trees_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        X = np.asarray(X, dtype=float)
        preds = np.vstack(
            [
                tree.predict(X[:, cols])
                for tree, cols in zip(self.trees_, self.feature_sets_, strict=False)
            ]
        )
        return preds.mean(axis=0)


@dataclass
class GradientBoosting:
    """Gradient boosting with shallow regression trees as the base learner.

    Classification uses the standard logit trick: fit trees to the negative
    gradient of the log-loss (``y - p``) and add them to the log-odds. The
    learning rate shrinks each tree's contribution, which is what keeps a
    greedy stagewise fit from overfitting in the first ten rounds.
    """

    n_estimators: int = 40
    learning_rate: float = 0.15
    max_depth: int = 3
    task: str = "classification"
    seed: int = 0

    trees_: list[DecisionTree] = field(default_factory=list, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)
    base_: float = field(default=0.0, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoosting:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.trees_ = []

        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError(
                    "GradientBoosting here is binary-only; use LogisticRegression "
                    "or RandomForest for multiclass."
                )
            target = (y == self.classes_[1]).astype(float)
            p = np.clip(target.mean(), 1e-6, 1 - 1e-6)
            self.base_ = float(np.log(p / (1 - p)))
            raw = np.full(y.size, self.base_)

            for _ in range(self.n_estimators):
                prob = 1.0 / (1.0 + np.exp(-raw))
                residual = target - prob
                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=2,
                    task="regression",
                    seed=self.seed,
                )
                tree.fit(X, residual)
                raw += self.learning_rate * tree.predict(X)
                self.trees_.append(tree)
        else:
            target = y.astype(float)
            self.base_ = float(target.mean())
            raw = np.full(y.size, self.base_)

            for _ in range(self.n_estimators):
                residual = target - raw
                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=2,
                    task="regression",
                    seed=self.seed,
                )
                tree.fit(X, residual)
                raw += self.learning_rate * tree.predict(X)
                self.trees_.append(tree)
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.base_)
        for tree in self.trees_:
            out += self.learning_rate * tree.predict(X)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification")
        p1 = 1.0 / (1.0 + np.exp(-self._raw(X)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
        return self._raw(X)


@dataclass
class KNN:
    """k-nearest neighbours with distance-weighted voting.

    Kept in the candidate pool because it is the honest counter-example: on
    small tabular data it is often within a point of the fancy models, and
    when it is, that is worth knowing before shipping a forest.
    """

    k: int = 5
    task: str = "classification"

    X_: np.ndarray | None = field(default=None, init=False, repr=False)
    y_: np.ndarray | None = field(default=None, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNN:
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(self.y_)
        return self

    def _neighbours(self, X: np.ndarray):
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("KNN: call fit first")
        X = np.asarray(X, dtype=float)
        k = min(self.k, self.X_.shape[0])
        out = []
        for row in X:
            dist = np.sqrt(((self.X_ - row) ** 2).sum(axis=1))
            idx = np.argsort(dist, kind="mergesort")[:k]
            weights = 1.0 / (dist[idx] + 1e-9)
            out.append((idx, weights))
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for idx, weights in self._neighbours(X):
            labels = self.y_[idx]
            if self.task == "classification":
                scores = {c: 0.0 for c in self.classes_}
                for label, w in zip(labels, weights, strict=False):
                    scores[label] += w
                preds.append(max(scores, key=scores.get))
            else:
                preds.append(float(np.average(labels.astype(float), weights=weights)))
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification")
        rows = []
        for idx, weights in self._neighbours(X):
            labels = self.y_[idx]
            scores = np.zeros(len(self.classes_))
            for label, w in zip(labels, weights, strict=False):
                scores[int(np.where(self.classes_ == label)[0][0])] += w
            rows.append(scores / scores.sum())
        return np.vstack(rows)
