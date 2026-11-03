"""Application settings, loaded from the environment.

Every value has a safe development default so `uvicorn app.main:app` works on
a fresh clone with no .env file. Production values are never defaulted —
`SECRET_KEY` and `DATABASE_URL` are validated at startup when
``ENVIRONMENT=production``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    app_name: str = "Chunav"
    environment: str = field(
        default_factory=lambda: _env("ENVIRONMENT", "development") or "development"
    )

    # Auth
    secret_key: str = field(
        default_factory=lambda: _env("SECRET_KEY", "dev-only-insecure-key") or ""
    )
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = field(default_factory=lambda: _env_int("ACCESS_TOKEN_MINUTES", 720))

    # Database
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "sqlite:///./chunav.db") or ""
    )

    # Limits
    max_upload_bytes: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_BYTES", 5_000_000))
    max_rows: int = field(default_factory=lambda: _env_int("MAX_ROWS", 20_000))
    default_folds: int = field(default_factory=lambda: _env_int("DEFAULT_FOLDS", 5))
    max_candidates: int = field(default_factory=lambda: _env_int("MAX_CANDIDATES", 8))
    search_timeout_seconds: int = field(
        default_factory=lambda: _env_int("SEARCH_TIMEOUT_SECONDS", 120)
    )

    # Security
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in (_env("CORS_ORIGINS", "*") or "*").split(",") if o.strip()
        ]
    )
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 120)
    )

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    def validate(self) -> list[str]:
        """Return a list of fatal configuration problems (empty when healthy)."""
        problems: list[str] = []
        if self.is_production:
            if self.secret_key in ("", "dev-only-insecure-key"):
                problems.append("SECRET_KEY must be set to a strong random value in production")
            if len(self.secret_key) < 32:
                problems.append("SECRET_KEY must be at least 32 characters in production")
            if self.database_url.startswith("sqlite"):
                problems.append("DATABASE_URL must point at PostgreSQL in production")
            if "*" in self.cors_origins:
                problems.append("CORS_ORIGINS must list explicit origins in production")
        if self.default_folds < 2:
            problems.append("DEFAULT_FOLDS must be at least 2")
        return problems


settings = Settings()
