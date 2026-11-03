"""End-to-end check against a live server.

Starts uvicorn on a free port, drives the real HTTP API, and prints what came
back. Nothing is mocked — if this passes, the app works.

    python scripts/smoke.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(base: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/api/health", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.4)
    raise RuntimeError("server did not become healthy in time")


def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "DATABASE_URL": f"sqlite:///{ROOT / 'smoke.db'}",
        "SECRET_KEY": "smoke-test-key-not-for-production-use-32ch",
        "ENVIRONMENT": "development",
    }

    db_file = ROOT / "smoke.db"
    if db_file.exists():
        db_file.unlink()

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))

    try:
        wait_for(base)
        print(f"server up on {base}\n")

        with httpx.Client(base_url=base, timeout=180.0) as c:
            r = c.get("/api/health")
            check("health returns 200", r.status_code == 200, r.json().get("status", ""))

            r = c.get("/")
            check("index.html served", r.status_code == 200 and "Chunav" in r.text)

            r = c.get("/api/datasets")
            check("unauthenticated datasets -> 401", r.status_code == 401)

            r = c.post(
                "/api/auth/register",
                json={
                    "email": "smoke@chunav.pk",
                    "display_name": "Smoke",
                    "password": "smoke-pass-123",
                },
            )
            check("register -> 201", r.status_code == 201, str(r.status_code))
            token = r.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}

            r = c.post(
                "/api/auth/register",
                json={
                    "email": "smoke@chunav.pk",
                    "display_name": "Smoke",
                    "password": "smoke-pass-123",
                },
            )
            check("duplicate register -> 409", r.status_code == 409)

            r = c.post("/api/auth/login", json={"email": "smoke@chunav.pk", "password": "wrong"})
            check("bad password -> 401", r.status_code == 401)

            r = c.get("/api/auth/me", headers=headers)
            check(
                "me returns the user",
                r.status_code == 200 and r.json()["email"] == "smoke@chunav.pk",
            )

            churn = (ROOT / "samples" / "churn.csv").read_text(encoding="utf-8")
            r = c.post("/api/datasets/profile", json={"csv_text": churn}, headers=headers)
            prof = r.json()
            check("profile -> 200", r.status_code == 200)
            check("profile found 6 columns", prof.get("n_columns") == 6, str(prof.get("n_columns")))
            check("profile suggested churned", prof.get("suggested_target") == "churned")

            r = c.post(
                "/api/datasets",
                json={"name": "Smoke churn", "target_column": "churned", "csv_text": churn},
                headers=headers,
            )
            check("create dataset -> 201", r.status_code == 201, str(r.status_code))
            ds = r.json()
            check("task inferred as classification", ds.get("task") == "classification")

            r = c.post(
                "/api/runs",
                json={"dataset_id": ds["id"], "n_folds": 5, "seed": 42, "max_candidates": 8},
                headers=headers,
            )
            check("run search -> 201", r.status_code == 201, str(r.status_code))
            run = r.json()
            board = run["leaderboard"]
            check("leaderboard has rows", len(board) >= 4, f"{len(board)} models")
            check("winner beats baseline", board[0]["lift"] > 0, f"lift {board[0]['lift']:+.4f}")
            check("ranks are 1..n", [b["rank"] for b in board] == list(range(1, len(board) + 1)))
            check("baseline is 0.59", abs(run["baseline"] - 0.59) < 1e-9, f"{run['baseline']}")

            print(
                f"\n  winner: {run['winner_label']}  score {run['winner_score']:.4f}  "
                f"baseline {run['baseline']:.4f}  lift {board[0]['lift']:+.4f}"
            )
            for b in board[:4]:
                print(f"    #{b['rank']} {b['label']:38s} {b['score']:.4f} ±{b['score_std']:.4f}")

            r = c.post(
                f"/api/runs/{run['id']}/predict",
                json={
                    "rows": [
                        {
                            "tenure_months": 6,
                            "monthly_bill_pkr": 3200,
                            "support_calls": 5,
                            "contract": "monthly",
                            "has_internet": "yes",
                        },
                        {
                            "tenure_months": 60,
                            "monthly_bill_pkr": 1200,
                            "support_calls": 0,
                            "contract": "two_year",
                            "has_internet": "no",
                        },
                    ]
                },
                headers=headers,
            )
            check("predict -> 200", r.status_code == 200, str(r.status_code))
            preds = r.json()["predictions"]
            check("two predictions returned", len(preds) == 2)
            check("confidence in [0,1]", all(0.0 <= p["confidence"] <= 1.0 for p in preds))
            print(f"\n  predictions: {json.dumps(preds)}")

            r = c.get("/api/runs", headers=headers)
            check("list runs -> 200", r.status_code == 200 and len(r.json()) == 1)

            r = c.get("/api/runs/99999", headers=headers)
            check("missing run -> 404", r.status_code == 404)

            r = c.post("/api/datasets/profile", json={"csv_text": "a,b\n"}, headers=headers)
            check("header-only CSV -> 422", r.status_code == 422)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        if db_file.exists():
            db_file.unlink()

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for name, ok, detail in checks:
            if not ok:
                print(f"  - {name}  {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
