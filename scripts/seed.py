"""Seed a demo account with the churn sample already loaded.

Idempotent: running it twice does not create a second user or dataset.

    python scripts/seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.automl.data import load_dataset  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Dataset, User  # noqa: E402
from app.security import hash_password  # noqa: E402

DEMO_EMAIL = "demo@chunav.pk"
DEMO_PASSWORD = "chunav-demo-2026"


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.email == DEMO_EMAIL)).scalar_one_or_none()
        if user is None:
            user = User(
                email=DEMO_EMAIL,
                display_name="Demo Analyst",
                password_hash=hash_password(DEMO_PASSWORD),
                role="analyst",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"created user {DEMO_EMAIL}")
        else:
            print(f"user {DEMO_EMAIL} already exists")

        sample = ROOT / "samples" / "churn.csv"
        if not sample.exists():
            print("samples/churn.csv missing — run scripts/make_samples.py first")
            return

        existing = db.execute(
            select(Dataset).where(
                Dataset.owner_id == user.id, Dataset.name == "Telecom churn (sample)"
            )
        ).scalar_one_or_none()
        if existing is not None:
            print("sample dataset already loaded")
            return

        text = sample.read_text(encoding="utf-8")
        parsed = load_dataset(text, "churned")
        import json

        db.add(
            Dataset(
                owner_id=user.id,
                name="Telecom churn (sample)",
                target_column="churned",
                task=parsed.task,
                n_rows=parsed.n_rows,
                n_columns=parsed.profile.n_columns,
                csv_text=text,
                profile_json=json.dumps(parsed.profile.to_dict()),
            )
        )
        db.commit()
        print(f"loaded sample dataset ({parsed.n_rows} rows)")
    finally:
        db.close()

    print(f"\nDemo login: {DEMO_EMAIL} / {DEMO_PASSWORD}")


if __name__ == "__main__":
    main()
