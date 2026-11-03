# Architecture

## The one-sentence version

A FastAPI service that turns a CSV into a ranked, baseline-annotated model
leaderboard using estimators written from scratch on NumPy, then refits the
winner to score new rows.

---

## System diagram

```
┌───────────────────────────────────────────────────────────────────┐
│  Browser — single-page app (vanilla JS, no build step)            │
│  auth · CSV upload · profile table · leaderboard · predict form   │
└───────────────────────────────┬───────────────────────────────────┘
                                │ JSON over HTTPS, Bearer JWT
┌───────────────────────────────▼───────────────────────────────────┐
│  FastAPI                                                          │
│                                                                   │
│  middleware:  CORS  →  rate limit  →  body-size guard             │
│                                                                   │
│  routers/                                                         │
│    auth.py       register · login · me                            │
│    datasets.py   profile · create · list · get · delete           │
│    runs.py       create · list · get · predict                    │
└───────────────────────────────┬───────────────────────────────────┘
                                │
┌───────────────────────────────▼───────────────────────────────────┐
│  app/automl/  — the engine, no ML framework                       │
│                                                                   │
│  data.py     CSV → typed matrix                                   │
│              type inference · one-hot · median impute             │
│              stratified k-fold                                    │
│                                                                   │
│  search.py   candidate pool → CV loop → leaderboard               │
│              mean score · std · lift over baseline                │
│                                                                   │
│  models.py   logistic · ridge · CART · random forest              │
│              gradient boosting · k-NN · StandardScaler            │
│                                                                   │
│  metrics.py  accuracy · confusion matrix · P/R/F1                 │
│              ROC-AUC (Mann-Whitney) · RMSE · MAE · R²             │
│              majority / mean baselines                            │
│                                                                   │
│  predict.py  refit winner on full data → encode rows → score      │
└───────────────────────────────┬───────────────────────────────────┘
                                │ SQLAlchemy 2.0
┌───────────────────────────────▼───────────────────────────────────┐
│  PostgreSQL 16 (production)  /  SQLite (development)              │
│  users · datasets · runs · predictions                            │
└───────────────────────────────────────────────────────────────────┘
```

---

## Data flow: one search

```
1. POST /api/datasets/profile
   csv_text ──► _read_rows ──► per-column type inference
                            ──► ColumnProfile[] + suggested target
   (nothing is persisted — this is the confirmation step)

2. POST /api/datasets
   csv_text ──► load_dataset
                  ├─ drop rows with an empty target
                  ├─ numeric columns  → float, median-imputed
                  ├─ categorical      → one-hot, categories seen ≥2 times
                  └─ target           → classification or regression
   ──► Dataset row (raw CSV kept for exact reproduction)

3. POST /api/runs
   Dataset ──► kfold_indices (stratified for classification)
           ──► for each candidate in pool:
                 for each fold:
                   StandardScaler.fit(train)      ← fit on train only
                   model.fit(train)
                   score(test)
                 mean, std, lift
           ──► sort by (score desc, std asc, complexity asc)
   ──► Run row (leaderboard + notes as JSON)

4. POST /api/runs/{id}/predict
   Run ──► winner key + params
       ──► refit on the FULL dataset
       ──► encode each request row into the dataset's feature space
       ──► predict / predict_proba
   ──► Prediction rows
```

---

## Decisions worth defending

### The scaler is fit on the training fold only

Fitting a scaler on the whole dataset before splitting leaks the test set's
mean and variance into training. It is a small leak and it inflates scores by a
point or two — which is exactly the kind of quiet dishonesty this project
exists to avoid.

### Ranking is by mean CV score, ties broken by simplicity

When two models are within noise of each other, the simpler one wins. A random
forest that beats logistic regression by 0.002 is not better; it is luckier.
The complexity rank (`ridge` < `logistic` < `knn` < `tree` < `boosting` <
`forest`) is the tie-break.

### Regression scores are negated internally

`_score_fold` returns `-RMSE` so that "higher is better" holds for every task
and the sort never has to branch. The negation is undone at the display layer
in `LeaderboardRow.to_dict()`, where `lift` is reported as
`baseline − RMSE` — the error the model removed, in the metric's own units.

This was a real bug during development: the first version compared the negated
score against a positive baseline and declared that no regression model ever
beat the baseline. The smoke test caught it because the numbers were
obviously wrong.

### Gradient boosting is dropped on multiclass targets

The implementation is binary (logit + `y − p` residuals). Rather than let it
fail on every fold and fill the notes with identical errors, the pool is
filtered before the loop.

### The raw CSV is stored, not just the profile

A profile is a summary; it cannot be retrained from. Keeping the original text
means a run is exactly reproducible months later, at the cost of a `TEXT`
column. For a tool whose entire value proposition is auditability, that trade
is not close.

---

## Scaling notes

**What breaks first: the synchronous search.** `POST /api/runs` holds the
request open for the whole CV loop. Measured: ~2 s on 150 rows, ~17 s on 400
rows with 8 candidates × 5 folds. At 20,000 rows this is minutes and the
request will time out behind most proxies.

The fix, in order of effort:

1. **Background job + polling.** `POST /api/runs` returns `202` with a run id in
   `queued` state; the client polls `GET /api/runs/{id}`. Needs a `status`
   column and a worker.
2. **A real queue.** Celery or RQ with Redis. The search is pure CPU and
   embarrassingly parallel across candidates — one worker per candidate turns
   a 17 s run into ~3 s.
3. **Cap the work, not the data.** `MAX_CANDIDATES` and `n_folds` already bound
   the loop; the remaining cost is per-fit, which scales with rows × features.

**Second: the rate limiter.** It is a `defaultdict(deque)` in process memory.
With N uvicorn workers the effective limit is N × the configured value, and a
restart clears it. Redis with a sliding window is the standard fix.

**Third: model persistence.** The winner is refit on every prediction request.
That is fine at 400 rows and wasteful at 20,000. Serialising the fitted model
(pickle or a portable format) into the `runs` row would make prediction O(1).

**What scales fine:** the database layer. Every query is indexed on `owner_id`
and the leaderboard is a JSON blob read whole — no N+1, no joins in the hot
path.

---

## Security posture

| Concern | Handling |
|---|---|
| Password storage | bcrypt, 12 rounds, 72-byte cap enforced |
| Session | JWT HS256, 12 h default, `sub` = user id |
| User enumeration | Login returns one message for both failure modes |
| Authorization | Every dataset/run query filters on `owner_id` |
| Upload size | `MAX_UPLOAD_BYTES` in the schema + a `Content-Length` guard |
| Row/column caps | 20,000 rows, 60 columns, 400 encoded features |
| Rate limiting | Per-IP, 120/min default, `Retry-After` on 429 |
| CORS | `*` in dev, explicit origins required in production |
| Secrets | Environment only; production startup fails on a default key |
| SQL injection | SQLAlchemy parameterised queries throughout |
| XSS | Frontend escapes every interpolated value via `esc()` |
