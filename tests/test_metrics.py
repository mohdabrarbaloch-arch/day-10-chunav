"""Tests for the metrics module.

Every assertion here is a number that can be checked by hand — that is the
point of writing the metrics from scratch.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.automl import metrics


class TestAccuracy:
    def test_perfect(self):
        assert metrics.accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_half(self):
        assert metrics.accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            metrics.accuracy([], [])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics.accuracy([1, 2], [1])


class TestConfusionMatrix:
    def test_known_matrix(self):
        # 3 correct, 1 off-diagonal, 1 correct, 1 off-diagonal
        cm = metrics.confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "a"], ["a", "b"])
        assert cm.tolist() == [[1, 1], [1, 1]]

    def test_rows_are_actual(self):
        cm = metrics.confusion_matrix(["a", "a", "a", "b"], ["a", "a", "a", "b"], ["a", "b"])
        assert cm[0].sum() == 3  # three actual 'a'
        assert cm[1].sum() == 1  # one actual 'b'

    def test_unseen_prediction_ignored(self):
        cm = metrics.confusion_matrix(["a", "b"], ["a", "zzz"], ["a", "b"])
        assert cm.sum() == 1


class TestPerClass:
    def test_precision_recall_by_hand(self):
        # class 0: tp=2, fp=1, fn=0  -> P=2/3, R=1.0
        # class 1: tp=1, fp=0, fn=1  -> P=1.0, R=0.5
        cm = np.array([[2, 0], [1, 1]])
        rows = metrics.per_class_prf(cm)
        assert rows[0]["precision"] == pytest.approx(2 / 3)
        assert rows[0]["recall"] == pytest.approx(1.0)
        assert rows[1]["precision"] == pytest.approx(1.0)
        assert rows[1]["recall"] == pytest.approx(0.5)

    def test_f1_is_harmonic_mean(self):
        cm = np.array([[2, 0], [1, 1]])
        row = metrics.per_class_prf(cm)[0]
        p, r = row["precision"], row["recall"]
        assert row["f1"] == pytest.approx(2 * p * r / (p + r))

    def test_zero_division_is_zero_not_nan(self):
        cm = np.array([[0, 0], [0, 3]])
        rows = metrics.per_class_prf(cm)
        assert rows[0]["precision"] == 0.0
        assert rows[0]["f1"] == 0.0

    def test_macro_f1_averages_classes(self):
        cm = np.array([[2, 0], [1, 1]])
        rows = metrics.per_class_prf(cm)
        assert metrics.macro_f1(cm) == pytest.approx(np.mean([r["f1"] for r in rows]))

    def test_weighted_f1_weights_by_support(self):
        cm = np.array([[8, 0], [1, 1]])
        rows = metrics.per_class_prf(cm)
        expected = sum(r["f1"] * r["support"] for r in rows) / 10
        assert metrics.weighted_f1(cm) == pytest.approx(expected)


class TestRocAuc:
    def test_perfect_separation(self):
        auc = metrics.roc_auc_binary([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1)
        assert auc == pytest.approx(1.0)

    def test_perfect_inversion(self):
        auc = metrics.roc_auc_binary([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 1)
        assert auc == pytest.approx(0.0)

    def test_all_ties_is_half(self):
        auc = metrics.roc_auc_binary([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 1)
        assert auc == pytest.approx(0.5)

    def test_single_class_returns_none(self):
        assert metrics.roc_auc_binary([1, 1, 1], [0.2, 0.5, 0.9], 1) is None


class TestRegression:
    def test_rmse_by_hand(self):
        # errors 1, -1, 2 -> mean square (1+1+4)/3 = 2 -> sqrt = 1.4142
        assert metrics.rmse([1, 2, 3], [2, 1, 5]) == pytest.approx(np.sqrt(2.0))

    def test_mae_by_hand(self):
        assert metrics.mae([1, 2, 3], [2, 1, 5]) == pytest.approx(4 / 3)

    def test_r2_perfect(self):
        assert metrics.r2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_r2_can_be_negative(self):
        # Predicting the mean is R2=0; worse than that is negative.
        assert metrics.r2([1, 2, 3], [10, 10, 10]) < 0

    def test_r2_of_mean_predictor_is_zero(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert metrics.r2(y, [2.5] * 4) == pytest.approx(0.0)


class TestBaselines:
    def test_majority_baseline(self):
        assert metrics.majority_baseline(["a", "a", "a", "b"]) == pytest.approx(0.75)

    def test_majority_baseline_balanced(self):
        assert metrics.majority_baseline(["a", "b", "a", "b"]) == pytest.approx(0.5)

    def test_mean_baseline_is_std(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert metrics.mean_baseline(y) == pytest.approx(np.std(y))
