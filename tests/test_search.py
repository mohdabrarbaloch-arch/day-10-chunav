"""Tests for the search loop and the leaderboard."""

from __future__ import annotations

import numpy as np
import pytest

from app.automl.data import load_dataset
from app.automl.search import candidate_pool, run_search


@pytest.fixture()
def churn_dataset(churn_csv):
    return load_dataset(churn_csv, "churned")


@pytest.fixture()
def exam_dataset(exam_csv):
    return load_dataset(exam_csv, "final_score")


class TestCandidatePool:
    def test_classification_pool_has_no_ridge(self):
        keys = {c.key for c in candidate_pool("classification")}
        assert "ridge" not in keys
        assert "logistic" in keys

    def test_regression_pool_has_no_logistic(self):
        keys = {c.key for c in candidate_pool("regression")}
        assert "logistic" not in keys
        assert "ridge" in keys

    def test_pool_is_small_and_fixed(self):
        # A huge pool makes the winner the max of many noisy estimates.
        assert len(candidate_pool("classification")) == 8
        assert len(candidate_pool("regression")) == 8


class TestSearch:
    def test_produces_a_ranked_leaderboard(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        assert len(result.leaderboard) >= 3
        ranks = [row.rank for row in result.leaderboard]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_leaderboard_is_sorted_by_score(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        scores = [row.score for row in result.leaderboard]
        assert scores == sorted(scores, reverse=True)

    def test_winner_is_the_first_row(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        assert result.winner is result.leaderboard[0]

    def test_baseline_is_reported(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        assert 0.0 < result.baseline < 1.0

    def test_same_seed_same_winner(self, churn_dataset):
        a = run_search(churn_dataset, n_folds=3, seed=7)
        b = run_search(churn_dataset, n_folds=3, seed=7)
        assert a.winner.key == b.winner.key
        assert a.winner.score == pytest.approx(b.winner.score)

    def test_max_candidates_truncates(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42, max_candidates=2)
        assert len(result.leaderboard) <= 2

    def test_regression_reports_rmse(self, exam_dataset):
        result = run_search(exam_dataset, n_folds=3, seed=42)
        assert result.primary_metric == "rmse"
        assert result.winner.score < 0  # stored negated

    def test_regression_lift_is_positive_when_model_wins(self, exam_dataset):
        # This is the bug the smoke test caught: comparing a negated score
        # against a positive baseline inverted the verdict.
        result = run_search(exam_dataset, n_folds=3, seed=42)
        row = result.winner.to_dict()
        assert row["lift"] > 0
        assert row["score"] > 0  # un-negated for display

    def test_multiclass_drops_gradient_boosting(self):
        rows = "\n".join(
            f"{i},{i % 3},{'a' if i % 3 == 0 else ('b' if i % 3 == 1 else 'c')}"
            for i in range(1, 61)
        )
        ds = load_dataset(f"x,y,label\n{rows}\n", "label")
        result = run_search(ds, n_folds=3, seed=1)
        assert "gradient_boosting" not in {row.key for row in result.leaderboard}

    def test_notes_mention_the_baseline(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        assert any("aseline" in note.lower() for note in result.notes)

    def test_to_dict_round_trips(self, churn_dataset):
        result = run_search(churn_dataset, n_folds=3, seed=42)
        payload = result.to_dict()
        assert payload["winner"]["rank"] == 1
        assert len(payload["leaderboard"]) == len(result.leaderboard)
        assert isinstance(payload["baseline"], float)
