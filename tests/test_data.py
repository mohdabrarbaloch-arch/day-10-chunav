"""Tests for data handling: parsing, type inference, folds."""

from __future__ import annotations

import numpy as np
import pytest

from app.automl.data import kfold_indices, load_dataset, profile_csv, stratified_folds


class TestProfile:
    def test_detects_numeric_and_categorical(self):
        csv = "age,city,score\n25,Karachi,88\n31,Lahore,92\n40,Karachi,75\n"
        p = profile_csv(csv)
        kinds = {c.name: c.kind for c in p.columns}
        assert kinds["age"] == "numeric"
        assert kinds["city"] == "categorical"
        assert kinds["score"] == "numeric"

    def test_mixed_column_is_categorical(self):
        # "N/A" means the column is not numeric, even though 2 of 3 parse.
        csv = "v\n12\nN/A\n8\n"
        p = profile_csv(csv)
        assert p.columns[0].kind == "categorical"

    def test_counts_missing(self):
        csv = "a,b\n1,x\n,y\n3,z\n"
        p = profile_csv(csv)
        assert p.columns[0].n_missing == 1

    def test_empty_csv_raises(self):
        with pytest.raises(ValueError, match="empty"):
            profile_csv("")

    def test_header_only_raises(self):
        with pytest.raises(ValueError, match="no data rows"):
            profile_csv("a,b\n")

    def test_suggests_last_column(self):
        csv = "x,y,label\n1,2,a\n3,4,b\n5,6,a\n"
        assert profile_csv(csv).suggested_target == "label"


class TestLoadDataset:
    def test_infers_classification(self):
        rows = "\n".join(f"{i},{i * 2},{'a' if i % 2 else 'b'}" for i in range(1, 21))
        ds = load_dataset(f"x,y,label\n{rows}\n", "label")
        assert ds.task == "classification"
        assert set(ds.classes) == {"a", "b"}

    def test_infers_regression(self):
        rows = "\n".join(f"{i},{i * 2},{i * 3.7 + 1}" for i in range(1, 30))
        ds = load_dataset(f"a,b,target\n{rows}\n", "target")
        assert ds.task == "regression"

    def test_one_hot_expands_features(self):
        rows = "\n".join(
            f"{'Karachi' if i % 2 else 'Lahore'},{'a' if i % 2 else 'b'}" for i in range(20)
        )
        ds = load_dataset(f"city,label\n{rows}\n", "label")
        assert ds.n_features == 2  # two cities
        assert "city=Karachi" in ds.feature_names

    def test_drops_rows_with_empty_target(self):
        csv = "x,label\n1,a\n2,\n3,b\n4,a\n5,b\n6,a\n7,b\n8,a\n9,b\n10,a\n11,b\n"
        ds = load_dataset(csv, "label")
        assert ds.n_rows == 10

    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="not in the CSV header"):
            load_dataset("a,b\n1,2\n3,4\n", "nope")

    def test_single_class_raises(self):
        csv = "x,label\n" + "\n".join(f"{i},same" for i in range(15))
        with pytest.raises(ValueError, match="only one distinct value"):
            load_dataset(csv, "label")

    def test_too_few_labelled_rows_raises(self):
        csv = "x,label\n1,a\n2,\n3,\n"
        with pytest.raises(ValueError, match="non-empty target"):
            load_dataset(csv, "label")

    def test_regression_on_text_target_raises(self):
        rows = "\n".join(f"{i},{'a' if i % 2 else 'b'}" for i in range(1, 21))
        with pytest.raises(ValueError, match="not numeric"):
            load_dataset(f"x,label\n{rows}\n", "label", task="regression")


class TestFolds:
    def test_stratified_keeps_class_balance(self):
        y = np.array(["a"] * 50 + ["b"] * 50, dtype=object)
        folds = stratified_folds(y, 5, seed=1)
        assert len(folds) == 5
        for fold in folds:
            labels = y[fold]
            assert (labels == "a").sum() == 10
            assert (labels == "b").sum() == 10

    def test_folds_cover_every_row_once(self):
        y = np.array(["a", "b"] * 30, dtype=object)
        folds = stratified_folds(y, 5, seed=2)
        combined = np.concatenate(folds)
        assert len(combined) == 60
        assert len(set(combined.tolist())) == 60

    def test_rare_class_falls_back_to_shuffle(self):
        # One class has 2 members, fewer than the 5 folds requested.
        y = np.array(["a"] * 20 + ["rare"] * 2, dtype=object)
        folds = stratified_folds(y, 5, seed=3)
        assert len(folds) == 5
        assert sum(len(f) for f in folds) == 22

    def test_kfold_regression_shuffles(self):
        y = np.arange(50, dtype=float)
        folds = kfold_indices(y, 5, seed=4, task="regression")
        assert len(folds) == 5
        assert sum(len(f) for f in folds) == 50

    def test_same_seed_same_folds(self):
        y = np.array(["a", "b"] * 20, dtype=object)
        a = stratified_folds(y, 4, seed=99)
        b = stratified_folds(y, 4, seed=99)
        for x, z in zip(a, b, strict=False):
            assert np.array_equal(x, z)
