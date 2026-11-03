"""Database engine and session management.

SQLAlchemy 2.0 style. SQLite is the development default; PostgreSQL is what
docker-compose and production use. The only place the two differ is the
connect args, and that is handled here so nothing downstream has to care.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import settings


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        # check_same_thread=False is required because FastAPI hands requests to
        # a thread pool; SQLite otherwise refuses the cross-thread reuse.
        kwargs: dict = {"connect_args": {"check_same_thread": False}}
        # An in-memory SQLite database lives only as long as its connection.
        # Without a StaticPool every new connection gets a fresh, empty
        # database — which is why the test suite saw "no such table: users".
        if ":memory:" in url or url.rstrip("/") == "sqlite:":
            kwargs["poolclass"] = StaticPool
        return kwargs
    return {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 10}


engine = create_engine(settings.database_url, future=True, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that always closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. Import models first so they register on Base.metadata."""
    from . import models  # noqa: F401  (side effect: registers tables)

    Base.metadata.create_all(bind=engine)
