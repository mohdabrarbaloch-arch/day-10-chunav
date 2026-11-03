"""The AutoML search loop.

Given a :class:`~app.automl.data.Dataset`, this trains a fixed candidate pool
under cross-validation, scores each one, and ranks them into a leaderboard.

Design choices that matter:

* **The candidate pool is fixed and small.** A search that tries 200
  configurations and reports the best one is reporting the maximum of 200
  noisy estimates — the winner is partly luck. Twelve candidates with honest
  folds is a claim you can defend.
* **Every score carries its baseline.** A classification row shows the
  majority-class accuracy next to it; a regression row shows the mean-predictor
  RMSE. Without that column a 0.69 looks like a result instead of a coin flip.
* **The winner is chosen on mean CV score, ties broken by simplicity.** When
  two models are within a hair of each other, the simpler one wins — that is
  the only tie-break that generalises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .data import Dataset, kfold_indices
from .models import (
    KNN,
    DecisionTree,
    GradientBoosting,
    LogisticRegression,
    RandomForest,
    RidgeRegression,
    StandardScaler,
)

__all__ = ["Candidate", "LeaderboardRow", "SearchResult", "run_search", "candidate_pool"]

# Complexity rank used only to break near-ties. Lower = simpler.
_COMPLEXITY = {
    "ridge": 1,
    "logistic": 2,
    "knn": 3,
    "tree": 4,
    "gradient_boosting": 5,
    "random_forest": 6,
}


@dataclass
class Candidate:
    key: str
    label: str
    family: str
    params: dict
    needs_scaling: bool = True

    def build(self, task: str):
        """Instantiate the estimator for the given task."""
        p = self.params
        if self.key == "logistic":
            return LogisticRegression(**p)
        if self.key == "ridge":
            return RidgeRegression(**p)
        if self.key == "tree":
            return DecisionTree(task=task, **p)
        if self.key == "random_forest":
            return RandomForest(task=task, **p)
        if self.key == "gradient_boosting":
            return GradientBoosting(task=task, **p)
        if self.key == "knn":
            return KNN(task=task, **p)
        raise KeyError(f"unknown candidate '{self.key}'")


@dataclass
class LeaderboardRow:
    rank: int
    key: str
    label: str
    family: str
    params: dict
    primary_metric: str
    score: float
    score_std: float
    baseline: float
    fold_scores: list[float]
    fit_seconds: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # ``score`` is stored in a uniform "higher is better" direction, so for
        # regression it is -RMSE. Lift is reported in the metric's own units:
        # for RMSE that means how much error the model removed.
        if self.primary_metric == "rmse":
            lift = self.baseline - (-self.score)
            display_score = -self.score
        else:
            lift = self.score - self.baseline
            display_score = self.score
        return {
            "rank": self.rank,
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "params": self.params,
            "primary_metric": self.primary_metric,
            "score": round(display_score, 6),
            "score_std": round(self.score_std, 6),
            "baseline": round(self.baseline, 6),
            "lift": round(lift, 6),
            "fold_scores": [
                round(-s if self.primary_metric == "rmse" else s, 6) for s in self.fold_scores
            ],
            "fit_seconds": round(self.fit_seconds, 4),
            "detail": self.detail,
        }


@dataclass
class SearchResult:
    task: str
    primary_metric: str
    n_folds: int
    seed: int
    baseline: float
    leaderboard: list[LeaderboardRow]
    winner: LeaderboardRow
    total_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "primary_metric": self.primary_metric,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "baseline": round(self.baseline, 6),
            "total_seconds": round(self.total_seconds, 3),
            "notes": self.notes,
            "winner": self.winner.to_dict(),
            "leaderboard": [row.to_dict() for row in self.leaderboard],
        }


def candidate_pool(task: str) -> list[Candidate]:
    """The fixed pool. Classification and regression get different members."""
    if task == "classification":
        return [
            Candidate("logistic", "Logistic Regression", "linear", {"epochs": 300, "l2": 1e-3}),
            Candidate("knn", "k-Nearest Neighbours (k=5)", "instance", {"k": 5}),
            Candidate("knn", "k-Nearest Neighbours (k=11)", "instance", {"k": 11}),
            Candidate("tree", "Decision Tree (depth 4)", "tree", {"max_depth": 4}),
            Candidate("tree", "Decision Tree (depth 8)", "tree", {"max_depth": 8}),
            Candidate(
                "random_forest",
                "Random Forest (12 trees, depth 7)",
                "ensemble",
                {"n_estimators": 12, "max_depth": 7},
            ),
            Candidate(
                "random_forest",
                "Random Forest (24 trees, depth 10)",
                "ensemble",
                {"n_estimators": 24, "max_depth": 10},
            ),
            Candidate(
                "gradient_boosting",
                "Gradient Boosting (40 rounds)",
                "ensemble",
                {"n_estimators": 40, "learning_rate": 0.15},
            ),
        ]
    return [
        Candidate("ridge", "Ridge Regression (α=0.1)", "linear", {"alpha": 0.1}),
        Candidate("ridge", "Ridge Regression (α=1.0)", "linear", {"alpha": 1.0}),
        Candidate("ridge", "Ridge Regression (α=10)", "linear", {"alpha": 10.0}),
        Candidate("knn", "k-Nearest Neighbours (k=5)", "instance", {"k": 5}),
        Candidate("tree", "Decision Tree (depth 4)", "tree", {"max_depth": 4}),
        Candidate("tree", "Decision Tree (depth 8)", "tree", {"max_depth": 8}),
        Candidate(
            "random_forest",
            "Random Forest (12 trees, depth 7)",
            "ensemble",
            {"n_estimators": 12, "max_depth": 7},
        ),
        Candidate(
            "gradient_boosting",
            "Gradient Boosting (40 rounds)",
            "ensemble",
            {"n_estimators": 40, "learning_rate": 0.15},
        ),
    ]


def _score_fold(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    classes: np.ndarray | None,
) -> tuple[float, dict]:
    """Return (primary score, detail dict) for one fold.

    Higher is always better, so regression metrics are negated here and
    un-negated for display. Keeping one direction avoids a class of bug where
    a "best" model is actually the worst.
    """
    if task == "classification":
        pred = model.predict(X_test)
        cm = metrics.confusion_matrix(y_test, pred, classes)
        acc = metrics.accuracy(y_test, pred)
        detail = {
            "accuracy": acc,
            "macro_f1": metrics.macro_f1(cm),
            "weighted_f1": metrics.weighted_f1(cm),
            "confusion_matrix": cm.tolist(),
            "per_class": metrics.per_class_prf(cm),
        }
        if classes is not None and len(classes) == 2:
            try:
                proba = model.predict_proba(X_test)
                auc = metrics.roc_auc_binary(y_test, proba[:, 1], classes[1])
                detail["roc_auc"] = auc
            except (RuntimeError, AttributeError):
                detail["roc_auc"] = None
        return acc, detail

    pred = model.predict(X_test)
    return -metrics.rmse(y_test, pred), {
        "rmse": metrics.rmse(y_test, pred),
        "mae": metrics.mae(y_test, pred),
        "r2": metrics.r2(y_test, pred),
    }


def _merge_detail(details: list[dict], task: str) -> dict:
    """Average the per-fold detail dicts into one summary."""
    if not details:
        return {}
    merged: dict = {}
    numeric_keys = [
        k
        for k in details[0]
        if isinstance(details[0][k], (int, float)) and details[0][k] is not None
    ]
    for key in numeric_keys:
        values = [d[key] for d in details if isinstance(d.get(key), (int, float))]
        if values:
            merged[key] = float(np.mean(values))

    if task == "classification":
        # Sum the fold confusion matrices — a single matrix over all folds is
        # more useful than an average of matrices.
        total = np.zeros_like(np.array(details[0]["confusion_matrix"]))
        for d in details:
            total += np.array(d["confusion_matrix"])
        merged["confusion_matrix"] = total.tolist()
        merged["per_class"] = metrics.per_class_prf(total)
        merged["macro_f1"] = metrics.macro_f1(total)
        merged["weighted_f1"] = metrics.weighted_f1(total)
    return merged


def run_search(
    dataset: Dataset,
    n_folds: int = 5,
    seed: int = 42,
    max_candidates: int | None = None,
) -> SearchResult:
    """Train the candidate pool under CV and rank it.

    ``max_candidates`` truncates the pool — useful for a fast smoke run and
    for keeping a request inside a time budget on a large dataset.
    """
    started = time.perf_counter()
    task = dataset.task
    notes: list[str] = []

    folds = kfold_indices(dataset.y, n_folds, seed, task)
    n_folds = len(folds)

    if task == "classification":
        primary = "accuracy"
        baseline = metrics.majority_baseline(dataset.y)
        # Higher is better for accuracy, so a positive lift means the model
        # beat the baseline.
        higher_is_better = True
        notes.append(
            f"Baseline is majority-class accuracy ({baseline:.3f}) — "
            "always predicting the most common label."
        )
    else:
        primary = "rmse"
        baseline = metrics.mean_baseline(dataset.y)
        # RMSE is negated internally so the sort direction is uniform, which
        # means a *less negative* score is better. Comparing the raw numbers
        # against the baseline would invert the verdict, so the direction is
        # tracked explicitly rather than assumed.
        higher_is_better = False
        notes.append(
            f"Baseline is the mean-predictor RMSE ({baseline:.3f}) — "
            "predicting the training mean for every row."
        )

    if dataset.n_rows < 100:
        notes.append(
            f"Only {dataset.n_rows} rows: fold scores will vary noticeably. "
            "Treat the ranking as a hint, not a verdict."
        )

    pool = candidate_pool(task)
    # Gradient boosting here is binary-only. Rather than let it fail on every
    # fold and litter the notes with a stack of identical errors, drop it from
    # the pool when the target has more than two classes.
    if task == "classification" and dataset.classes is not None and len(dataset.classes) > 2:
        pool = [c for c in pool if c.key != "gradient_boosting"]
    if max_candidates is not None:
        pool = pool[:max_candidates]

    rows: list[LeaderboardRow] = []

    for candidate in pool:
        fold_scores: list[float] = []
        details: list[dict] = []
        fit_seconds = 0.0
        failed = False

        for test_idx in folds:
            train_mask = np.ones(dataset.n_rows, dtype=bool)
            train_mask[test_idx] = False
            X_train, y_train = dataset.X[train_mask], dataset.y[train_mask]
            X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

            if len(np.unique(y_train)) < 2 and task == "classification":
                failed = True
                break

            try:
                model = candidate.build(task)
                if candidate.needs_scaling:
                    scaler = StandardScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)

                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds += time.perf_counter() - t0

                score, detail = _score_fold(model, X_test, y_test, task, dataset.classes)
                fold_scores.append(score)
                details.append(detail)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                notes.append(f"{candidate.label} failed on a fold: {exc}")
                failed = True
                break

        if failed or not fold_scores:
            continue

        mean_score = float(np.mean(fold_scores))
        rows.append(
            LeaderboardRow(
                rank=0,
                key=candidate.key,
                label=candidate.label,
                family=candidate.family,
                params=candidate.params,
                primary_metric=primary,
                score=mean_score,
                score_std=float(np.std(fold_scores)),
                baseline=baseline,
                fold_scores=fold_scores,
                fit_seconds=fit_seconds,
                detail=_merge_detail(details, task),
            )
        )

    if not rows:
        raise RuntimeError("Every candidate failed — check the dataset and target")

    # Rank by mean score, then by stability, then by simplicity.
    rows.sort(
        key=lambda r: (
            -r.score,
            r.score_std,
            _COMPLEXITY.get(r.key, 99),
        )
    )
    for i, row in enumerate(rows, start=1):
        row.rank = i

    winner = rows[0]
    # For regression the stored score is -RMSE, so "beat the baseline" means
    # the model's RMSE is *smaller* than the mean predictor's.
    beat_baseline = winner.score > baseline if higher_is_better else -winner.score < baseline
    if not beat_baseline:
        notes.append(
            "No candidate beat the baseline. That is a real result: the "
            "features in this file do not carry signal for this target."
        )

    return SearchResult(
        task=task,
        primary_metric=primary,
        n_folds=n_folds,
        seed=seed,
        baseline=baseline,
        leaderboard=rows,
        winner=winner,
        total_seconds=time.perf_counter() - started,
        notes=notes,
    )
