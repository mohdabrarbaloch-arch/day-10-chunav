# Usage

## The four steps

### 1 · Load a dataset

Paste a CSV or choose a file. Click **Inspect columns** first — it infers types
and suggests a target without saving anything, so you can confirm the guess
before committing.

What the profiler decides:

- A column is **numeric** only if every non-empty value parses as a number.
  `12`, `N/A`, `8` is categorical, not numeric-with-a-hole.
- A column is **categorical** otherwise. Categories appearing only once are
  dropped from the encoding — a value seen once cannot be learned from.
- The **suggested target** is the last column, unless that column looks like an
  identifier (high cardinality, numeric), in which case the last low-cardinality
  categorical column is suggested instead.

Then **Save dataset**. The raw CSV is stored so the run is reproducible.

### 2 · Run the search

| Control | Meaning |
|---|---|
| **Folds** | k in k-fold CV. 5 is a sane default; 10 is slower and noisier on small data |
| **Seed** | Controls the fold assignment. Same seed + same data = same leaderboard |
| **Max candidates** | Truncates the pool. Lower it for a fast look |

### 3 · Read the leaderboard

Every row shows:

- **Score** — mean across folds, in the task's own metric
- **± std** — spread across folds. A big std means the score is unstable
- **Baseline** — what you get with no model at all
- **Lift** — score minus baseline, in the metric's units
- **Relative bar** — the score's position between the worst model and the best

The winner is highlighted. Notes underneath explain the baseline and flag
anything that went wrong.

**How to read it honestly:**

- Lift under ~0.02 on a few hundred rows is inside the noise. Do not ship it.
- A high score with a high std is a coin flip that landed well once.
- If the notes say *"No candidate beat the baseline"*, believe them. The
  features do not carry signal for that target.

### 4 · Score new rows

One JSON object per line, keyed by your **original CSV column names**:

```json
{"tenure_months": 6, "monthly_bill_pkr": 3200, "support_calls": 5, "contract": "monthly", "has_internet": "yes"}
{"tenure_months": 60, "monthly_bill_pkr": 1200, "support_calls": 0, "contract": "two_year", "has_internet": "no"}
```

Missing fields are treated as absent: numeric columns fall back to 0, one-hot
columns to all-zeros. A category the model never saw encodes as all-zeros —
the honest representation of "unknown", not a guess.

Confidence is the max class probability for classifiers. For regression it is
blank, because a regression has no probability to report.

---

## Worked example: the churn sample

```bash
python scripts/make_samples.py
uvicorn app.main:app --reload
```

1. Click **Load sample (churn)** — 400 rows, target `churned`.
2. **Inspect columns** → 5 features, 1 target, 2 categorical.
3. **Save dataset** → task auto-detected as `classification`.
4. **Train & rank** with 5 folds, seed 42.

Expected result:

```
#1  Logistic Regression          0.6998 ±0.0381   lift +0.1098
#2  k-NN (k=11)                  0.6724 ±0.0351   lift +0.0824
#3  k-NN (k=5)                   0.6600 ±0.0141   lift +0.0700
#4  Random Forest (12 trees)     0.6251 ±0.0474   lift +0.0351
    baseline (majority class)    0.5900
```

Logistic regression wins, and the margin is thin. That is the correct answer
for this data — the generator was written with weak log-odds on purpose. A
tool reporting 0.95 here would be lying to you.

---

## Worked example: regression

Load `samples/exam_score.csv`, target `final_score`.

```
#1  Gradient Boosting (40 rounds)   RMSE 5.1189 ±0.3362   lift +5.3592
#2  Ridge (α=1.0)                   RMSE 5.2632 ±0.5631   lift +5.2149
    baseline (mean predictor)       RMSE 10.4781
```

Lift is `baseline − RMSE`: the model removed 5.36 points of error from a
10.48-point baseline. Roughly half the variance, which matches the generator's
noise level.

---

## Common errors

| Message | Cause |
|---|---|
| `Target column 'x' is not in the CSV header` | Typo, or the header has trailing spaces |
| `Only N rows have a non-empty target` | Too many blank labels; need at least 10 |
| `Target has only one distinct value` | Nothing to learn — check the column |
| `One-hot encoding produced N features` | Too many high-cardinality categoricals; drop some |
| `CSV exceeds the N byte upload limit` | Raise `MAX_UPLOAD_BYTES` or trim the file |
| `You already have a dataset named 'x'` | Names are unique per user; rename or delete |
| `429 Too many requests` | Rate limit; wait 60 s or raise `RATE_LIMIT_PER_MINUTE` |
