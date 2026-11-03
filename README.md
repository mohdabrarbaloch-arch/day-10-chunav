# Chunav (چناؤ)

**An AutoML platform where every score is auditable.**

Upload a CSV, pick a target, and Chunav trains a fixed pool of models under
cross-validation and ranks them. The difference from every other AutoML demo:
**the baseline sits next to every number.** A 0.69 accuracy on a 0.63 base rate
is not a result, and this tool says so out loud.

There is no scikit-learn, no XGBoost, no autograd underneath. Every estimator,
every metric and every split is written from scratch on NumPy, so any score on
the leaderboard can be recomputed by hand from the confusion matrix.

---

## Why this exists

AutoML tools are easy to demo and hard to trust. The usual failure is a
leaderboard of 200 configurations where the winner is the maximum of 200 noisy
estimates — it looks like a result and is mostly luck.

Chunav takes the opposite position:

- **A small, fixed candidate pool.** Eight configurations, not two hundred.
- **Honest folds.** Stratified for classification, so a rare class does not
  vanish from a fold and turn its F1 into a coin flip.
- **Baselines everywhere.** Majority-class accuracy for classification, the
  mean-predictor RMSE for regression.
- **A stated verdict when nothing works.** If no model beats the baseline, the
  run says so instead of quietly reporting the least-bad number.

---

## What it does

| Step | What happens |
|---|---|
| **Profile** | Infers column types, missing counts, cardinality, and suggests a target |
| **Search** | Trains 8 candidates under k-fold CV, scores each, ranks them |
| **Explain** | Shows per-fold scores, std, lift over baseline, confusion matrix, per-class P/R/F1 |
| **Predict** | Refits the winner on the full dataset and scores new rows |

### The candidate pool

**Classification:** Logistic Regression · k-NN (k=5, k=11) · Decision Tree
(depth 4, depth 8) · Random Forest (12 and 24 trees) · Gradient Boosting

**Regression:** Ridge (α = 0.1, 1, 10) · k-NN · Decision Tree (depth 4, 8) ·
Random Forest · Gradient Boosting

Gradient boosting is binary-only here, so it is dropped from the pool on
multiclass targets rather than failing on every fold.

---

## Verified results

These are real numbers from `scripts/smoke.py` against the shipped samples —
not aspirational figures.

| Dataset | Task | Winner | Score | Baseline | Lift |
|---|---|---|---|---|---|
| `iris.csv` (150 rows) | 3-class | Random Forest (12 trees) | **0.9867** | 0.3333 | +0.6533 |
| `churn.csv` (400 rows) | binary | Logistic Regression | **0.6998** | 0.5900 | +0.1098 |
| `exam_score.csv` (300 rows) | regression | Gradient Boosting | **RMSE 5.1189** | 10.4781 | +5.3592 |

The churn result is the interesting one. A 0.11 lift over a 0.59 base rate is a
*thin* margin, and that is the honest answer — the generator was written with
weak log-odds on purpose. A tool that reported 0.95 here would be lying.

---

## Screenshots

**The leaderboard** — baseline in its own column, lift in the metric's units, and a relative bar so a model that barely clears the base rate looks like one:

![Leaderboard](https://static.teamily.ai/sites/f5348921-52dc-48e1-bafd-82f54e96c932/documents/day-10-chunav-screenshots/05-leaderboard.jpg)

**Sign in** — the demo button fills the seeded credentials:

![Sign in](https://static.teamily.ai/sites/f5348921-52dc-48e1-bafd-82f54e96c932/documents/day-10-chunav-screenshots/01-login.jpg)

**Column profile** — types inferred before anything is saved:

![Profile](https://static.teamily.ai/sites/f5348921-52dc-48e1-bafd-82f54e96c932/documents/day-10-chunav-screenshots/03-profile.jpg)

**Scoring new rows** — the winner refit on the full dataset:

![Predict](https://static.teamily.ai/sites/f5348921-52dc-48e1-bafd-82f54e96c932/documents/day-10-chunav-screenshots/06-predict.jpg)

**Mobile (390px)** — no horizontal overflow, the email drops out of the header before the sign-out button does:

<img src="https://static.teamily.ai/sites/f5348921-52dc-48e1-bafd-82f54e96c932/documents/day-10-chunav-screenshots/07-mobile.jpg" width="300" alt="Mobile">

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_samples.py     # writes samples/*.csv
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. API docs at `/docs`.

**Demo account:** `demo@chunav.pk` / `chunav-demo-2026`

### With PostgreSQL

```bash
docker compose up --build
```

That brings up Postgres 16 and the app on port 8000. Without Docker, set
`DATABASE_URL` and run uvicorn directly — SQLite is the default.

---

## API

All endpoints except `/api/health` and the auth pair require a bearer token.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/auth/register` | Create an account, returns a JWT |
| `POST` | `/api/auth/login` | Exchange credentials for a JWT |
| `GET` | `/api/auth/me` | Current user |
| `POST` | `/api/datasets/profile` | Infer column types without saving |
| `POST` | `/api/datasets` | Save a dataset |
| `GET` | `/api/datasets` | List your datasets |
| `GET` | `/api/datasets/{id}` | One dataset with its profile |
| `DELETE` | `/api/datasets/{id}` | Delete a dataset and its runs |
| `POST` | `/api/runs` | Run the search, returns the leaderboard |
| `GET` | `/api/runs` | List your runs |
| `GET` | `/api/runs/{id}` | One run with the full leaderboard |
| `POST` | `/api/runs/{id}/predict` | Score new rows with the winner |

Full request/response shapes: [`docs/api.md`](docs/api.md).

---

## Stack

- **Python 3.11** · FastAPI 0.115 · SQLAlchemy 2.0 · Pydantic v2
- **NumPy** for the engine — no ML framework
- **JWT** (HS256) + **bcrypt** (12 rounds) for auth
- **SQLite** for development, **PostgreSQL 16** via docker-compose
- Vanilla JS single-page frontend, no build step

---

## Project layout

```
app/
  automl/
    metrics.py     accuracy, F1, AUC, RMSE, R², baselines — all from scratch
    models.py      logistic, ridge, CART, random forest, boosting, k-NN
    data.py        CSV parsing, type inference, stratified folds
    search.py      the CV loop and the leaderboard
    predict.py     refit the winner, encode new rows
  routers/         auth, datasets, runs
  static/          the SPA
  config.py        env-driven settings with production validation
  security.py      bcrypt + JWT
  models.py        ORM
scripts/
  make_samples.py  generates the three sample datasets
  smoke.py         end-to-end check against a live server
```

---

## Known limits

Stated rather than hidden:

- **Training is synchronous.** `POST /api/runs` blocks until the search finishes.
  On the 400-row churn sample that is ~17 seconds; on 20,000 rows it will be
  minutes. A worker queue is the right fix and is not built.
- **The rate limiter is in-process.** With N uvicorn workers the effective
  limit is N × `RATE_LIMIT_PER_MINUTE`. It does not coordinate across processes.
- **Gradient boosting is binary-only.** Multiclass targets drop it from the pool.
- **No model persistence.** The winner is refit on demand for each prediction
  request rather than serialised to disk.
- **Not deployed.** There is no live URL. The repo is deploy-ready
  (`vercel.json`, Dockerfile, compose) but no deployment has been performed or
  verified.

---

## License

MIT — see [LICENSE](LICENSE).
