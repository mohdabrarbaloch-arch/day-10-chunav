# Setup

## Requirements

- Python 3.11+
- pip
- Optional: Docker + Docker Compose (for the PostgreSQL path)

## Local, SQLite (fastest)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/make_samples.py   # writes samples/*.csv
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs at
<http://127.0.0.1:8000/docs>.

The database file `chunav.db` is created on first start. Tables are created by
`init_db()` at startup — there are no migrations to run.

## Local, PostgreSQL

```bash
docker compose up --build
```

App on <http://127.0.0.1:8000>, Postgres on `localhost:5432`.

To point at an existing Postgres instead:

```bash
export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/chunav"
uvicorn app.main:app --reload
```

## Configuration

Copy `.env.example` to `.env` and edit. Every variable is documented in that
file. The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` turns on strict validation |
| `SECRET_KEY` | dev placeholder | **Must** be 32+ chars in production |
| `DATABASE_URL` | `sqlite:///./chunav.db` | Must be Postgres in production |
| `MAX_UPLOAD_BYTES` | `5000000` | 5 MB CSV cap |
| `MAX_ROWS` | `20000` | Rows read per CSV |
| `DEFAULT_FOLDS` | `5` | 2–10 |
| `MAX_CANDIDATES` | `8` | Candidates trained per run |
| `CORS_ORIGINS` | `*` | Explicit list required in production |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per IP, in-process |

Generate a production key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Demo account

```
python scripts/seed.py
```

Creates `demo@chunav.pk` / `chunav-demo-2026` with the churn sample already
loaded. The frontend's "Use demo account" button fills these in.

## Tests

```bash
pytest -q
```

## Lint and format

```bash
ruff check .
black --check .
```

## Production notes

- Set `ENVIRONMENT=production`. The app **refuses to start** if `SECRET_KEY` is
the default, shorter than 32 characters, `DATABASE_URL` is SQLite, or
  `CORS_ORIGINS` contains `*`. That is deliberate: a misconfigured production
  deploy should fail loudly at boot, not quietly serve with a known key.
- Run behind a reverse proxy that terminates TLS.
- Use more than one uvicorn worker only after reading the rate-limiter caveat
  below.

## Known limits

Stated plainly rather than discovered in production:

1. **The search is synchronous.** `POST /api/runs` blocks for the whole CV
   loop. Measured ~2 s on 150 rows, ~17 s on 400 rows. At 20,000 rows it will
   exceed most proxy timeouts. A worker queue is the fix and is not built.
2. **The rate limiter is per-process.** With N workers the effective limit is
   N × `RATE_LIMIT_PER_MINUTE`. It does not coordinate across processes and
   resets on restart.
3. **`SEARCH_TIMEOUT_SECONDS` is declared but not enforced.** It is a
   documented intention, not a working guard. Do not rely on it.
4. **Gradient boosting is binary-only.** Multiclass targets drop it from the
   pool automatically.
5. **No model persistence.** The winner is refit on each prediction request.
6. **No deployment has been performed.** The repo ships `vercel.json`, a
   Dockerfile and compose, but there is no live URL and none has been verified.
