"""Chunav — an AutoML platform where every score is auditable.

FastAPI application factory. Middleware order matters and is deliberate:
CORS outermost so preflight requests are answered before anything else can
reject them, then the rate limiter, then the request-size guard.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .routers import auth, datasets, runs

logger = logging.getLogger("chunav")

STATIC_DIR = Path(__file__).parent / "static"


class RateLimiter:
    """Fixed-window-per-key limiter held in process memory.

    Honest about its limits: it does not coordinate across workers, so with N
    uvicorn workers the effective limit is N times the configured value. That
    is fine for a single-container deployment and is documented in
    docs/setup.md rather than papered over.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        return True


limiter = RateLimiter(settings.rate_limit_per_minute)


def create_app() -> FastAPI:
    problems = settings.validate()
    if problems:
        # Fail loudly at import time in production rather than serving with a
        # default secret key.
        raise RuntimeError("Invalid configuration: " + "; ".join(problems))

    app = FastAPI(
        title="Chunav API",
        description=(
            "An AutoML platform where every leaderboard score is computed from "
            "scratch and shown next to its baseline."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Health and static assets are exempt: a limiter that blocks the
        # health check turns a load spike into a false outage.
        path = request.url.path
        if path.startswith("/api/") and not path.startswith("/api/health"):
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Too many requests — slow down and retry shortly"},
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)

    @app.middleware("http")
    async def body_size_middleware(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > settings.max_upload_bytes * 2:
                        return JSONResponse(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            content={"detail": "Request body is too large"},
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Malformed Content-Length header"},
                    )
        return await call_next(request)

    @app.on_event("startup")
    def _startup() -> None:
        init_db()
        logger.info("Chunav started in %s mode", settings.environment)

    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(runs.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "app": settings.app_name,
            "environment": settings.environment,
            "version": "1.0.0",
        }

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()
