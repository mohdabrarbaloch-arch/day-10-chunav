# API reference

Base URL: `http://127.0.0.1:8000` in development.

All endpoints except `/api/health`, `/api/auth/register` and `/api/auth/login`
require `Authorization: Bearer <token>`.

Errors use FastAPI's standard shape:

```json
{ "detail": "Human-readable message" }
```

Validation errors (422) return `detail` as a list of field-level objects.

---

## Meta

### `GET /api/health`

No auth. Liveness probe.

```json
{ "status": "ok", "app": "Chunav", "environment": "development", "version": "1.0.0" }
```

---

## Auth

### `POST /api/auth/register` → `201`

```json
{ "email": "you@example.com", "display_name": "Ayesha Khan", "password": "at-least-8-chars" }
```

Returns:

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "user": { "id": 1, "email": "you@example.com", "display_name": "Ayesha Khan", "role": "analyst", "created_at": "2026-11-04T09:12:00Z" }
}
```

| Status | When |
|---|---|
| `201` | Created |
| `409` | Email already registered |
| `422` | Bad email, short password, blank name |

### `POST /api/auth/login` → `200`

```json
{ "email": "you@example.com", "password": "..." }
```

Same response shape as register. `401` for both "no such user" and "wrong
password" — deliberately identical, so the endpoint is not a user-enumeration
oracle.

### `GET /api/auth/me` → `200`

Returns the `user` object. `401` if the token is missing, malformed or expired.

---

## Datasets

### `POST /api/datasets/profile` → `200`

Infers types without persisting anything.

```json
{ "csv_text": "tenure,contract,churned\n12,monthly,yes\n" }
```

```json
{
  "n_rows": 400,
  "n_columns": 6,
  "columns": [
    { "name": "tenure_months", "kind": "numeric", "n_missing": 0, "n_unique": 72,
      "mean": 36.4, "std": 20.8, "min": 1.0, "max": 72.0 },
    { "name": "contract", "kind": "categorical", "n_missing": 0, "n_unique": 3,
      "top_values": [{ "value": "monthly", "count": 141 }] }
  ],
  "suggested_target": "churned",
  "warnings": ["'monthly_bill_pkr' has 2 missing value(s); ..."]
}
```

`422` if the CSV is empty, has no data rows, or exceeds the column cap.

### `POST /api/datasets` → `201`

```json
{ "name": "Q4 churn", "target_column": "churned", "task": "classification", "csv_text": "..." }
```

`task` may be `null` to infer it. Returns the dataset plus its full profile.

| Status | When |
|---|---|
| `201` | Saved |
| `409` | You already have a dataset with that name |
| `422` | Bad target, too few labelled rows, too many features |

### `GET /api/datasets` → `200`

Array of datasets, newest first. Profile omitted for brevity.

### `GET /api/datasets/{id}` → `200`

One dataset with its profile. `404` if it is not yours.

### `DELETE /api/datasets/{id}` → `204`

Cascades to the dataset's runs and their predictions. `404` if not yours.

---

## Runs

### `POST /api/runs` → `201`

```json
{ "dataset_id": 1, "n_folds": 5, "seed": 42, "max_candidates": 8 }
```

`n_folds` 2–10, `seed` 0–2³¹−1, `max_candidates` 1–20 (null uses the server
default).

**This call is synchronous** — it returns when the whole CV loop finishes.
Measured ~2 s on 150 rows, ~17 s on 400 rows with 8 candidates × 5 folds.

```json
{
  "id": 7,
  "dataset_id": 1,
  "task": "classification",
  "primary_metric": "accuracy",
  "n_folds": 5,
  "seed": 42,
  "baseline": 0.59,
  "winner_key": "logistic",
  "winner_label": "Logistic Regression",
  "winner_score": 0.6998,
  "duration_seconds": 16.78,
  "created_at": "2026-11-04T09:20:00Z",
  "leaderboard": [
    {
      "rank": 1,
      "key": "logistic",
      "label": "Logistic Regression",
      "family": "linear",
      "params": { "epochs": 300, "l2": 0.001 },
      "primary_metric": "accuracy",
      "score": 0.6998,
      "score_std": 0.0381,
      "baseline": 0.59,
      "lift": 0.1098,
      "fold_scores": [0.7125, 0.65, 0.7375, 0.7, 0.7],
      "fit_seconds": 0.0412,
      "detail": {
        "accuracy": 0.6998,
        "macro_f1": 0.6602,
        "weighted_f1": 0.6901,
        "roc_auc": 0.7412,
        "confusion_matrix": [[198, 38], [82, 82]],
        "per_class": [
          { "precision": 0.7071, "recall": 0.8390, "f1": 0.7674, "support": 236 },
          { "precision": 0.6833, "recall": 0.5000, "f1": 0.5775, "support": 164 }
        ]
      }
    }
  ],
  "notes": [
    "Baseline is majority-class accuracy (0.590) — always predicting the most common label."
  ]
}
```

For regression, `primary_metric` is `"rmse"`, `score` is the RMSE itself
(lower is better), and `lift` is `baseline − RMSE` — the error removed.

### `GET /api/runs` → `200`

Array of runs without the leaderboard.

### `GET /api/runs/{id}` → `200`

One run with the full leaderboard. `404` if not yours.

### `POST /api/runs/{id}/predict` → `200`

```json
{
  "rows": [
    { "tenure_months": 6, "monthly_bill_pkr": 3200, "support_calls": 5, "contract": "monthly", "has_internet": "yes" }
  ]
}
```

1–200 rows. Keys are your **original CSV column names**.

```json
{
  "run_id": 7,
  "task": "classification",
  "predictions": [
    { "row_index": 0, "predicted": "yes", "confidence": 0.7412 }
  ]
}
```

`confidence` is `null` for regression runs. The winner is refit on the full
dataset before scoring, so the first call after a restart is slower than the
rest.

---

## Rate limits

120 requests/minute per client IP by default (`RATE_LIMIT_PER_MINUTE`).
Exceeding it returns `429` with `Retry-After: 60`.

`/api/health` and static assets are exempt — a limiter that blocks the health
check turns a load spike into a false outage.

---

## Status codes used

| Code | Meaning here |
|---|---|
| `200` | OK |
| `201` | Created (register, dataset, run) |
| `204` | Deleted |
| `401` | Missing, malformed or expired token; bad credentials |
| `404` | Not found, or not yours |
| `409` | Duplicate email or dataset name |
| `413` | Request body over the size guard |
| `422` | Validation failure or unusable dataset |
| `429` | Rate limited |
