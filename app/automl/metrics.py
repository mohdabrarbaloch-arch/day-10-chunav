"""Metrics, computed from scratch.

Every number the leaderboard shows is produced here. Nothing is delegated to
scikit-learn, so each score can be recomputed by hand from the confusion
matrix or the residual vector. That is the point: a leaderboard you cannot
audit is just a ranking of opinions.

Conventions
-----------
* ``y_true`` / ``y_pred`` are 1-D arrays (or anything ``np.asarray`` accepts).
* Classification metrics take an explicit ``labels`` order so the confusion
  matrix rows/columns are stable across runs.
* Regression metrics return plain floats.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "accuracy",
    "confusion_matrix",
    "per_class_prf",
    "macro_f1",
    "weighted_f1",
    "roc_auc_binary",
    "log_loss",
    "rmse",
    "mae",
    "r2",
    "mean_baseline",
    "majority_baseline",
]


def _as_1d(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def accuracy(y_true, y_pred) -> float:
    """Fraction of exact matches. Raises on empty input rather than returning 0."""
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    if yt.size == 0:
        raise ValueError("accuracy: empty input")
    if yt.shape != yp.shape:
        raise ValueError(f"accuracy: shape mismatch {yt.shape} vs {yp.shape}")
    return float(np.mean(yt == yp))


def confusion_matrix(y_true, y_pred, labels: Sequence) -> np.ndarray:
    """Rows = actual, columns = predicted, ordered by ``labels``.

    Predictions outside ``labels`` are ignored rather than crashing the run —
    a model that emits an unseen class is a bug we want to surface in the
    metrics, not a 500.
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(yt, yp, strict=False):
        if actual in index and predicted in index:
            cm[index[actual], index[predicted]] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> list[dict]:
    """Precision / recall / F1 / support per class, derived from a confusion matrix.

    ``cm[i, i]`` is the true-positive count for class ``i``; the column sum
    minus that is false positives, the row sum minus that is false negatives.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("per_class_prf: expected a square confusion matrix")

    out: list[dict] = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

        out.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return out


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean F1 across classes — every class counts equally."""
    rows = per_class_prf(cm)
    if not rows:
        return 0.0
    return float(np.mean([r["f1"] for r in rows]))


def weighted_f1(cm: np.ndarray) -> float:
    """Support-weighted mean F1 — dominated by the common classes."""
    rows = per_class_prf(cm)
    total = sum(r["support"] for r in rows)
    if total == 0:
        return 0.0
    return float(sum(r["f1"] * r["support"] for r in rows) / total)


def roc_auc_binary(y_true, scores, positive_label) -> float | None:
    """Rank-based AUC for a binary problem, via the Mann-Whitney U statistic.

    Returns ``None`` when the fold contains only one class — AUC is undefined
    there, and inventing 0.5 would quietly flatter the model.
    """
    yt = _as_1d(y_true)
    s = _as_1d(scores).astype(float)
    if yt.shape != s.shape:
        raise ValueError("roc_auc_binary: shape mismatch")

    pos = s[yt == positive_label]
    neg = s[yt != positive_label]
    if pos.size == 0 or neg.size == 0:
        return None

    # Average ranks handle ties correctly (0.5 credit for a tied pair).
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    pos_ranks = ranks[yt == positive_label]
    u = pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def log_loss(y_true, probabilities: np.ndarray, labels: Sequence) -> float:
    """Multiclass cross-entropy. ``probabilities`` is (n_samples, n_classes)."""
    yt = _as_1d(y_true)
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (yt.size, len(labels)):
        raise ValueError("log_loss: probability matrix has the wrong shape")

    eps = 1e-15
    total = 0.0
    for row, actual in zip(probs, yt, strict=False):
        p = row[index[actual]] if actual in index else eps
        total += -np.log(min(max(p, eps), 1.0))
    return float(total / yt.size)


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def rmse(y_true, y_pred) -> float:
    yt = _as_1d(y_true).astype(float)
    yp = _as_1d(y_pred).astype(float)
    if yt.size == 0:
        raise ValueError("rmse: empty input")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt = _as_1d(y_true).astype(float)
    yp = _as_1d(y_pred).astype(float)
    if yt.size == 0:
        raise ValueError("mae: empty input")
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination. Can be negative — that is meaningful.

    A negative R² means the model is worse than predicting the mean, and we
    report it rather than clamping to zero.
    """
    yt = _as_1d(y_true).astype(float)
    yp = _as_1d(y_pred).astype(float)
    if yt.size == 0:
        raise ValueError("r2: empty input")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        # Constant target: perfect prediction is the only non-degenerate case.
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# --------------------------------------------------------------------------
# Baselines — the honest yardstick
# --------------------------------------------------------------------------


def majority_baseline(y_true) -> float:
    """Accuracy of always predicting the most common class.

    Shown next to every classification score so a 0.69 is not mistaken for
    skill when the majority class is 0.63.
    """
    yt = _as_1d(y_true)
    if yt.size == 0:
        raise ValueError("majority_baseline: empty input")
    _, counts = np.unique(yt, return_counts=True)
    return float(counts.max() / yt.size)


def mean_baseline(y_true) -> float:
    """RMSE of always predicting the training mean."""
    yt = _as_1d(y_true).astype(float)
    if yt.size == 0:
        raise ValueError("mean_baseline: empty input")
    return float(np.sqrt(np.mean((yt - np.mean(yt)) ** 2)))
