"""Dataset ingestion, profiling and splitting.

The job here is to turn an arbitrary CSV into something the search loop can
trust: typed columns, a declared target, and folds that do not leak.

Two decisions worth defending:

1. **Type inference is conservative.** A column is numeric only if every
   non-empty value parses as a number. "12", "N/A", "8" is a categorical
   column, not a numeric one with a hole in it — guessing otherwise silently
   turns missing data into a mean.
2. **Splits are stratified for classification.** A 3-class target with a rare
   class will otherwise produce folds where that class never appears, and the
   fold's F1 becomes a coin flip.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "Dataset",
    "profile_csv",
    "load_dataset",
    "stratified_folds",
    "kfold_indices",
]

MAX_ROWS = 20_000
MAX_COLUMNS = 60
MAX_CELL_CHARS = 200


@dataclass
class ColumnProfile:
    name: str
    kind: str  # "numeric" | "categorical"
    n_missing: int
    n_unique: int
    # numeric only
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    # categorical only
    top_values: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "kind": self.kind,
            "n_missing": self.n_missing,
            "n_unique": self.n_unique,
        }
        if self.kind == "numeric":
            out.update(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "min": self.minimum,
                    "max": self.maximum,
                }
            )
        else:
            out["top_values"] = [{"value": v, "count": c} for v, c in self.top_values]
        return out


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    suggested_target: str | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": [c.to_dict() for c in self.columns],
            "suggested_target": self.suggested_target,
            "warnings": self.warnings,
        }


@dataclass
class Dataset:
    """A typed, ready-to-train dataset.

    ``X`` is a float matrix (categoricals one-hot encoded), ``y`` is the raw
    target vector, and ``feature_names`` lines up with ``X``'s columns so the
    UI can explain a split by name instead of by index.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    task: str  # "classification" | "regression"
    classes: np.ndarray | None
    profile: DatasetProfile
    n_rows: int
    n_features: int

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "classes": [str(c) for c in self.classes] if self.classes is not None else None,
            "profile": self.profile.to_dict(),
        }


def _parse_number(value: str) -> float | None:
    text = value.strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _read_rows(text: str) -> tuple[list[str], list[list[str]]]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV is empty — no header row found")

    header = [h.strip() for h in header]
    if not header or all(h == "" for h in header):
        raise ValueError("CSV header row is empty")
    if len(header) > MAX_COLUMNS:
        raise ValueError(f"CSV has {len(header)} columns; the limit is {MAX_COLUMNS}")

    rows: list[list[str]] = []
    for raw in reader:
        if not raw or all(cell.strip() == "" for cell in raw):
            continue
        # Pad short rows rather than dropping them — a trailing empty field is
        # common and usually means "missing", not "malformed".
        if len(raw) < len(header):
            raw = raw + [""] * (len(header) - len(raw))
        rows.append([cell[:MAX_CELL_CHARS] for cell in raw[: len(header)]])
        if len(rows) >= MAX_ROWS:
            break
    return header, rows


def profile_csv(text: str) -> DatasetProfile:
    """Infer column types and summarise a CSV without training anything."""
    header, rows = _read_rows(text)
    if not rows:
        raise ValueError("CSV has a header but no data rows")

    warnings: list[str] = []
    columns: list[ColumnProfile] = []

    for j, name in enumerate(header):
        values = [row[j] for row in rows]
        non_empty = [v for v in values if v.strip() != ""]
        n_missing = len(values) - len(non_empty)

        numbers = [_parse_number(v) for v in non_empty]
        is_numeric = bool(non_empty) and all(n is not None for n in numbers)

        if is_numeric:
            arr = np.array(numbers, dtype=float)
            columns.append(
                ColumnProfile(
                    name=name,
                    kind="numeric",
                    n_missing=n_missing,
                    n_unique=int(np.unique(arr).size),
                    mean=float(arr.mean()),
                    std=float(arr.std()),
                    minimum=float(arr.min()),
                    maximum=float(arr.max()),
                )
            )
        else:
            counts: dict[str, int] = {}
            for v in non_empty:
                counts[v] = counts.get(v, 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            columns.append(
                ColumnProfile(
                    name=name,
                    kind="categorical",
                    n_missing=n_missing,
                    n_unique=len(counts),
                    top_values=top,
                )
            )

    # Suggest a target: the last column is the convention, but a low-cardinality
    # column anywhere is a better guess than a 400-value ID column.
    suggested = header[-1]
    last = columns[-1]
    if last.kind == "numeric" and last.n_unique > max(20, len(rows) * 0.5):
        for col in reversed(columns[:-1]):
            if col.kind == "categorical" and 2 <= col.n_unique <= 20:
                suggested = col.name
                warnings.append(
                    f"Last column '{last.name}' looks like an identifier; "
                    f"suggested '{col.name}' as the target instead."
                )
                break

    for col in columns:
        if col.n_missing > 0:
            warnings.append(
                f"'{col.name}' has {col.n_missing} missing value(s); "
                "rows with a missing target are dropped, missing features are imputed."
            )
    if len(rows) < 50:
        warnings.append(f"Only {len(rows)} rows — cross-validation scores will be noisy.")

    return DatasetProfile(
        n_rows=len(rows),
        n_columns=len(header),
        columns=columns,
        suggested_target=suggested,
        warnings=warnings,
    )


def load_dataset(
    text: str,
    target: str,
    task: str | None = None,
    max_categories: int = 25,
) -> Dataset:
    """Build a typed :class:`Dataset` from CSV text.

    ``task`` may be forced to ``"classification"`` or ``"regression"``; when
    omitted it is inferred from the target's cardinality (a numeric target
    with few distinct values is treated as classification, which is what a
    human would do looking at the same column).
    """
    header, rows = _read_rows(text)
    if target not in header:
        raise ValueError(f"Target column '{target}' is not in the CSV header")
    if len(header) < 2:
        raise ValueError("Need at least one feature column besides the target")

    target_index = header.index(target)

    # Drop rows whose target is empty — you cannot learn from a missing label.
    kept = [row for row in rows if row[target_index].strip() != ""]
    if len(kept) < 10:
        raise ValueError(f"Only {len(kept)} rows have a non-empty target; need at least 10")

    raw_target = [row[target_index] for row in kept]
    target_numbers = [_parse_number(v) for v in raw_target]
    target_is_numeric = all(n is not None for n in target_numbers)

    if task is None:
        if target_is_numeric:
            distinct = len(set(target_numbers))
            task = "classification" if distinct <= 12 else "regression"
        else:
            task = "classification"
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if task == "regression" and not target_is_numeric:
        raise ValueError(f"Target '{target}' is not numeric, so it cannot be a regression target")

    # ---- features --------------------------------------------------------
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    for j, name in enumerate(header):
        if j == target_index:
            continue
        values = [row[j] for row in kept]
        numbers = [_parse_number(v) for v in values]

        if all(n is not None for n in numbers):
            arr = np.array(numbers, dtype=float)
            # Impute with the median: robust to the outliers that a mean would
            # chase, and it keeps the column's shape.
            if np.isnan(arr).any():
                arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
            blocks.append(arr.reshape(-1, 1))
            feature_names.append(name)
            continue

        # Categorical: one-hot, but only for categories that actually recur.
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        common = [c for c, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 2]
        common = common[:max_categories]
        if not common:
            continue

        index = {c: i for i, c in enumerate(common)}
        block = np.zeros((len(values), len(common)))
        for row_i, v in enumerate(values):
            if v in index:
                block[row_i, index[v]] = 1.0
        blocks.append(block)
        feature_names.extend([f"{name}={c}" for c in common])

    if not blocks:
        raise ValueError("No usable feature columns found")

    X = np.hstack(blocks)
    if X.shape[1] > 400:
        raise ValueError(
            f"One-hot encoding produced {X.shape[1]} features; "
            "reduce the number of categorical columns or their cardinality"
        )

    # ---- target ----------------------------------------------------------
    if task == "classification":
        y = np.array(raw_target, dtype=object)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Target has only one distinct value — nothing to learn")
        if len(classes) > 50:
            raise ValueError(
                f"Target has {len(classes)} distinct values; that is too many for classification"
            )
    else:
        y = np.array(target_numbers, dtype=float)
        classes = None

    profile = profile_csv(text)

    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target,
        task=task,
        classes=classes,
        profile=profile,
        n_rows=X.shape[0],
        n_features=X.shape[1],
    )


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return ``n_folds`` index arrays, each class spread evenly across folds.

    Falls back to a plain shuffle when any class has fewer members than folds —
    stratification is impossible there, and pretending otherwise would produce
    empty folds.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_folds = max(2, min(n_folds, n))

    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < n_folds:
        shuffled = rng.permutation(n)
        return [np.asarray(f) for f in np.array_split(shuffled, n_folds)]

    buckets: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in classes:
        members = np.where(y == cls)[0]
        members = rng.permutation(members)
        for i, idx in enumerate(members):
            buckets[i % n_folds].append(int(idx))
    return [np.array(sorted(b)) for b in buckets]


def kfold_indices(y: np.ndarray, n_folds: int, seed: int, task: str) -> list[np.ndarray]:
    """Stratify for classification, shuffle for regression."""
    if task == "classification":
        return stratified_folds(y, n_folds, seed)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_folds = max(2, min(n_folds, n))
    return [np.asarray(f) for f in np.array_split(rng.permutation(n), n_folds)]
