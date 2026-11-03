"""Pydantic request/response schemas.

Validation lives here so the routers stay thin and every 422 is produced by
the same mechanism. Field constraints are deliberately tight — an AutoML
endpoint that accepts a 50,000-row CSV and a 200-fold CV is a denial-of-service
vector, not a feature.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .config import settings

# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_not_blank(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("Password cannot be only whitespace")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    display_name: str
    role: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    target_column: str = Field(min_length=1, max_length=160)
    task: str | None = Field(default=None, pattern="^(classification|regression)$")
    csv_text: str = Field(min_length=1)

    @field_validator("csv_text")
    @classmethod
    def csv_within_limit(cls, v: str) -> str:
        if len(v.encode("utf-8")) > settings.max_upload_bytes:
            raise ValueError(f"CSV exceeds the {settings.max_upload_bytes} byte upload limit")
        return v


class ColumnProfileOut(BaseModel):
    name: str
    kind: str
    n_missing: int
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    top_values: list[dict] | None = None


class DatasetProfileOut(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[ColumnProfileOut]
    suggested_target: str | None
    warnings: list[str]


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    target_column: str
    task: str
    n_rows: int
    n_columns: int
    created_at: datetime


class DatasetDetailOut(DatasetOut):
    profile: DatasetProfileOut


class ProfileRequest(BaseModel):
    csv_text: str = Field(min_length=1)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


class RunCreate(BaseModel):
    dataset_id: int = Field(gt=0)
    n_folds: int = Field(default=5, ge=2, le=10)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    max_candidates: int | None = Field(default=None, ge=1, le=20)


class LeaderboardRowOut(BaseModel):
    rank: int
    key: str
    label: str
    family: str
    params: dict
    primary_metric: str
    score: float
    score_std: float
    baseline: float
    lift: float
    fold_scores: list[float]
    fit_seconds: float
    detail: dict


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    task: str
    primary_metric: str
    n_folds: int
    seed: int
    baseline: float
    winner_key: str
    winner_label: str
    winner_score: float
    duration_seconds: float
    created_at: datetime


class RunDetailOut(RunOut):
    leaderboard: list[LeaderboardRowOut]
    notes: list[str]


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


class PredictRequest(BaseModel):
    rows: list[dict] = Field(min_length=1, max_length=200)

    @field_validator("rows")
    @classmethod
    def rows_not_empty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("Provide at least one row to score")
        return v


class PredictionOut(BaseModel):
    row_index: int
    predicted: str
    confidence: float | None = None


class PredictResponse(BaseModel):
    run_id: int
    task: str
    predictions: list[PredictionOut]
