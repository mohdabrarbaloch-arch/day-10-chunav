"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI callable named ``app`` in a file
under ``api/``. The real application lives in ``app/main.py``; this module
exists only to expose it at the path the platform expects.
"""

import sys
from pathlib import Path

# The function bundle runs from ``api/``, so the project root has to be on the
# import path before ``app`` can be imported as a package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402  (path setup must run first)

__all__ = ["app"]
