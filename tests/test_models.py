"""Tests for the estimators.

These assert properties that must hold regardless of the data — a constant
column must not produce NaN, a tree must handle string labels, the same seed
must give the same answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.automl.models import (
    KNN,
    DecisionTree,
    GradientBoosting,
    LogisticRegression,
    RandomForest,
    RidgeRegression,
    StandardScaler,
)


@pytest.fixture()
def blobs():
    """Two well-separated 2-D clusters."""
    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 0.4, size=(40, 2))
    b = rng.normal(4.0, 0.4, size=(40, 2))
    X = np.vstack([a, b])
    y = np.array(["low"] * 40 + ["high"] * 40, dtype=object)
    return X, y


class TestStandardScaler:
    def test_mean_is_zero_after_transform(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Z = StandardScaler().fit_transform(X)
        assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)

    def test_std_is_one_after_transform(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Z = StandardScaler().fit_transform(X)
        assert np.allclose(Z.std(axis=0), 1.0, atol=1e-12)

    def test_constant_column_does_not_produce_nan(self):
        # A column that never varies has std 0; dividing by it would poison
        # the whole matrix.
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        Z = StandardScaler().fit_transform(X)
        assert not np.isnan(Z).any()
        assert np.allclose(Z[:, 0], 0.0)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(np.array([[1.0]]))


class TestLogisticRegression:
    def test_separates_two_blobs(self, blobs):
        X, y = blobs
        model = LogisticRegression(epochs=300).fit(X, y)
        pred = model.predict(X)
        assert (pred == y).mean() > 0.95

    def test_probabilities_sum_to_one(self, blobs):
        X, y = blobs
        model = LogisticRegression(epochs=100).fit(X, y)
        proba = model.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_handles_three_classes(self):
        rng = np.random.default_rng(3)
        X = np.vstack(
            [
                rng.normal(0, 0.3, (20, 2)),
                rng.normal(5, 0.3, (20, 2)),
                rng.normal(10, 0.3, (20, 2)),
            ]
        )
        y = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20, dtype=object)
        model = LogisticRegression(epochs=400).fit(X, y)
        assert (model.predict(X) == y).mean() > 0.9

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LogisticRegression().predict(np.array([[1.0]]))


class TestRidgeRegression:
    def test_recovers_a_linear_relationship(self):
        rng = np.random.default_rng(11)
        X = rng.normal(0, 1, (100, 2))
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 5.0
        model = RidgeRegression(alpha=1e-6).fit(X, y)
        assert np.allclose(model.predict(X), y, atol=1e-3)

    def test_intercept_is_the_mean(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0])
        model = RidgeRegression(alpha=1.0).fit(X, y)
        assert model.intercept_ == pytest.approx(20.0)

    def test_heavy_regularisation_shrinks_coefficients(self):
        rng = np.random.default_rng(5)
        X = rng.normal(0, 1, (50, 3))
        y = X @ np.array([5.0, 5.0, 5.0])
        light = RidgeRegression(alpha=0.01).fit(X, y)
        heavy = RidgeRegression(alpha=1000.0).fit(X, y)
        assert np.linalg.norm(heavy.coef_) < np.linalg.norm(light.coef_)


class TestDecisionTree:
    def test_learns_a_simple_split(self, blobs):
        X, y = blobs
        tree = DecisionTree(max_depth=3, task="classification").fit(X, y)
        assert (tree.predict(X) == y).mean() > 0.95

    def test_string_labels_survive(self, blobs):
        # The leaf used to coerce labels to float, which explodes on "low".
        X, y = blobs
        tree = DecisionTree(max_depth=3, task="classification").fit(X, y)
        pred = tree.predict(X)
        assert set(pred) <= {"low", "high"}

    def test_depth_zero_is_a_single_leaf(self, blobs):
        X, y = blobs
        tree = DecisionTree(max_depth=0, task="classification").fit(X, y)
        pred = tree.predict(X)
        assert len(set(pred)) == 1

    def test_regression_tree_predicts_floats(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        y = 2.0 * X[:, 0]
        tree = DecisionTree(max_depth=4, task="regression").fit(X, y)
        pred = tree.predict(X)
        assert pred.dtype == np.float64
        assert np.corrcoef(pred, y)[0, 1] > 0.95

    def test_probabilities_are_normalised(self, blobs):
        X, y = blobs
        tree = DecisionTree(max_depth=3, task="classification").fit(X, y)
        proba = tree.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestRandomForest:
    def test_beats_a_single_shallow_tree(self, blobs):
        X, y = blobs
        forest = RandomForest(n_estimators=15, max_depth=3, task="classification").fit(X, y)
        assert (forest.predict(X) == y).mean() > 0.95

    def test_same_seed_same_predictions(self, blobs):
        X, y = blobs
        a = RandomForest(n_estimators=8, seed=1, task="classification").fit(X, y)
        b = RandomForest(n_estimators=8, seed=1, task="classification").fit(X, y)
        assert np.array_equal(a.predict(X), b.predict(X))

    def test_regression_forest_averages(self):
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (80, 2))
        y = X[:, 0] * 3 + rng.normal(0, 0.1, 80)
        forest = RandomForest(n_estimators=10, max_depth=5, task="regression").fit(X, y)
        assert np.corrcoef(forest.predict(X), y)[0, 1] > 0.9


class TestGradientBoosting:
    def test_binary_classification(self, blobs):
        X, y = blobs
        model = GradientBoosting(n_estimators=30, task="classification").fit(X, y)
        assert (model.predict(X) == y).mean() > 0.95

    def test_multiclass_raises_clearly(self):
        X = np.random.default_rng(1).normal(0, 1, (30, 2))
        y = np.array(["a", "b", "c"] * 10, dtype=object)
        with pytest.raises(ValueError, match="binary-only"):
            GradientBoosting(task="classification").fit(X, y)

    def test_regression_reduces_error(self):
        rng = np.random.default_rng(4)
        X = rng.normal(0, 1, (100, 2))
        y = X[:, 0] * 2 + rng.normal(0, 0.2, 100)
        model = GradientBoosting(n_estimators=40, task="regression").fit(X, y)
        rmse = np.sqrt(np.mean((model.predict(X) - y) ** 2))
        assert rmse < np.std(y)


class TestKNN:
    def test_classifies_blobs(self, blobs):
        X, y = blobs
        model = KNN(k=5, task="classification").fit(X, y)
        assert (model.predict(X) == y).mean() > 0.95

    def test_k_larger_than_dataset_is_clamped(self):
        X = np.array([[0.0], [1.0]])
        y = np.array(["a", "b"], dtype=object)
        model = KNN(k=50, task="classification").fit(X, y)
        assert len(model.predict(X)) == 2

    def test_regression_averages_neighbours(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        model = KNN(k=2, task="regression").fit(X, y)
        pred = model.predict(np.array([[1.5]]))
        assert 0.0 <= pred[0] <= 3.0
