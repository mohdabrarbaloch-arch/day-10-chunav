"""Shared pytest fixtures.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point at an in-memory database before anything imports the engine.
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("SECRET_KEY", "test-key-not-for-production-use-32-chars")
os.environ.setdefault("ENVIRONMENT", "development")

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth(client):
    """Register a fresh user and return (headers, user_json)."""
    import uuid

    email = f"u{uuid.uuid4().hex[:10]}@test.pk"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "display_name": "Test User", "password": "test-pass-123"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]


@pytest.fixture()
def churn_csv() -> str:
    return (ROOT / "samples" / "churn.csv").read_text(encoding="utf-8")


@pytest.fixture()
def exam_csv() -> str:
    return (ROOT / "samples" / "exam_score.csv").read_text(encoding="utf-8")
