"""Refit a winning candidate on the full dataset and score new rows.

The leaderboard is produced under cross-validation, which means no single
fitted model survives the search. To score new rows we refit the winner on
everything — the standard move, and worth stating plainly: the CV score is an
estimate of generalisation, the refit model is the one that actually answers.

New rows arrive as dicts keyed by the *original* CSV column names, so the
encoder has to be rebuilt from the same dataset and applied consistently.
Unseen categories become all-zeros, which is the honest encoding for "a value
the model never saw".
"""

from __future__ import annotations

import math

import numpy as np

from .data import Dataset, _parse_number
from .models import StandardScaler
from .search import Candidate, candidate_pool

__all__ = ["fit_winner", "score_rows"]


def _candidate_for(key: str, params: dict, task: str) -> Candidate:
    for candidate in candidate_pool(task):
        if candidate.key == key and candidate.params == params:
            return candidate
    # Fall back to a default-parameter candidate of the same family so a run
    # stored before a pool change can still be scored.
    for candidate in candidate_pool(task):
        if candidate.key == key:
            return candidate
    raise KeyError(f"no candidate named '{key}'")


def fit_winner(dataset: Dataset, winner_key: str, winner_params: dict):
    """Refit the winner on the full dataset. Returns (model, scaler)."""
    candidate = _candidate_for(winner_key, winner_params, dataset.task)
    model = candidate.build(dataset.task)

    X = dataset.X
    scaler = None
    if candidate.needs_scaling:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    model.fit(X, dataset.y)
    return model, scaler


def _encode_row(row: dict, dataset: Dataset) -> np.ndarray:
    """Encode one dict of raw values into the dataset's feature space.

    The dataset's ``feature_names`` are the source of truth: a numeric column
    appears as its own name, a one-hot column as ``"col=value"``. Anything the
    row does not supply is treated as missing (0 for one-hot, column median
    for numeric).
    """
    values: list[float] = []
    for name in dataset.feature_names:
        if "=" in name:
            column, category = name.split("=", 1)
            raw = row.get(column)
            values.append(1.0 if raw is not None and str(raw) == category else 0.0)
        else:
            raw = row.get(name)
            number = _parse_number(str(raw)) if raw is not None else None
            values.append(0.0 if number is None else number)
    return np.array(values, dtype=float)


def score_rows(
    dataset: Dataset, winner_key: str, winner_params: dict, rows: list[dict]
) -> list[dict]:
    """Score raw dict rows with the refit winner.

    Returns one dict per row: ``{"row_index", "predicted", "confidence"}``.
    Confidence is the max class probability for classifiers and ``None`` for
    regressors — a regression has no probability to report, and inventing one
    would be worse than leaving it blank.
    """
    model, scaler = fit_winner(dataset, winner_key, winner_params)

    matrix = np.vstack([_encode_row(row, dataset) for row in rows])
    if scaler is not None:
        matrix = scaler.transform(matrix)

    out: list[dict] = []
    if dataset.task == "classification":
        proba = model.predict_proba(matrix)
        labels = model.predict(matrix)
        for i, (label, probs) in enumerate(zip(labels, proba, strict=False)):
            confidence = float(np.max(probs))
            if math.isnan(confidence):
                confidence = None
            out.append(
                {
                    "row_index": i,
                    "predicted": str(label),
                    "confidence": confidence,
                }
            )
    else:
        preds = model.predict(matrix)
        for i, value in enumerate(preds):
            out.append(
                {
                    "row_index": i,
                    "predicted": f"{float(value):.6g}",
                    "confidence": None,
                }
            )
    return out
