"""ORM models.

Four tables, and the shape is deliberate:

* ``User`` — owner of everything. Passwords are stored as a bcrypt hash, never
  as anything reversible.
* ``Dataset`` — an uploaded CSV plus its inferred profile. The raw text is
  kept so a run can be reproduced exactly; a profile alone is not enough to
  retrain.
* ``Run`` — one AutoML search: the leaderboard, the winner, the notes. Stored
  as JSON because the leaderboard is a report, not a queryable entity.
* ``Prediction`` — a single row scored by a saved winner, kept so the UI can
  show what the model actually said.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="analyst", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_dataset_owner_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    target_column: Mapped[str] = mapped_column(String(160), nullable=False)
    task: Mapped[str] = mapped_column(String(20), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    csv_text: Mapped[str] = mapped_column(Text, nullable=False)
    profile_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    owner: Mapped[User] = relationship(back_populates="datasets")
    runs: Mapped[list[Run]] = relationship(back_populates="dataset", cascade="all, delete-orphan")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    task: Mapped[str] = mapped_column(String(20), nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(30), nullable=False)
    n_folds: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    baseline: Mapped[float] = mapped_column(Float, nullable=False)
    winner_key: Mapped[str] = mapped_column(String(60), nullable=False)
    winner_label: Mapped[str] = mapped_column(String(160), nullable=False)
    winner_score: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    leaderboard_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    notes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    owner: Mapped[User] = relationship(back_populates="runs")
    dataset: Mapped[Dataset] = relationship(back_populates="runs")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="predictions")
