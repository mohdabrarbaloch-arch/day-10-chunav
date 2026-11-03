"""Run routes: launch a search, read the leaderboard, score new rows."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..automl.data import load_dataset
from ..automl.predict import score_rows
from ..automl.search import run_search
from ..config import settings
from ..database import get_db
from ..models import Dataset, Prediction, Run, User
from ..schemas import (
    LeaderboardRowOut,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    RunCreate,
    RunDetailOut,
    RunOut,
)
from ..security import current_user

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _load_owned_dataset(db: Session, dataset_id: int, user: User) -> Dataset:
    record = db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id)
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return record


@router.post("", response_model=RunDetailOut, status_code=status.HTTP_201_CREATED)
def create_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> RunDetailOut:
    record = _load_owned_dataset(db, payload.dataset_id, user)

    try:
        dataset = load_dataset(record.csv_text, record.target_column, record.task)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    max_candidates = payload.max_candidates or settings.max_candidates
    try:
        result = run_search(
            dataset,
            n_folds=payload.n_folds,
            seed=payload.seed,
            max_candidates=max_candidates,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    run = Run(
        owner_id=user.id,
        dataset_id=record.id,
        task=result.task,
        primary_metric=result.primary_metric,
        n_folds=result.n_folds,
        seed=result.seed,
        baseline=result.baseline,
        winner_key=result.winner.key,
        winner_label=result.winner.label,
        winner_score=result.winner.score,
        duration_seconds=result.total_seconds,
        leaderboard_json=json.dumps([row.to_dict() for row in result.leaderboard]),
        notes_json=json.dumps(result.notes),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return RunDetailOut(
        id=run.id,
        dataset_id=run.dataset_id,
        task=run.task,
        primary_metric=run.primary_metric,
        n_folds=run.n_folds,
        seed=run.seed,
        baseline=run.baseline,
        winner_key=run.winner_key,
        winner_label=run.winner_label,
        winner_score=run.winner_score,
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
        leaderboard=[LeaderboardRowOut(**row) for row in json.loads(run.leaderboard_json)],
        notes=json.loads(run.notes_json),
    )


@router.get("", response_model=list[RunOut])
def list_runs(db: Session = Depends(get_db), user: User = Depends(current_user)) -> list[RunOut]:
    rows = (
        db.execute(select(Run).where(Run.owner_id == user.id).order_by(Run.created_at.desc()))
        .scalars()
        .all()
    )
    return [RunOut.model_validate(r) for r in rows]


@router.get("/{run_id}", response_model=RunDetailOut)
def get_run(
    run_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> RunDetailOut:
    run = db.execute(
        select(Run).where(Run.id == run_id, Run.owner_id == user.id)
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    return RunDetailOut(
        id=run.id,
        dataset_id=run.dataset_id,
        task=run.task,
        primary_metric=run.primary_metric,
        n_folds=run.n_folds,
        seed=run.seed,
        baseline=run.baseline,
        winner_key=run.winner_key,
        winner_label=run.winner_label,
        winner_score=run.winner_score,
        duration_seconds=run.duration_seconds,
        created_at=run.created_at,
        leaderboard=[LeaderboardRowOut(**row) for row in json.loads(run.leaderboard_json)],
        notes=json.loads(run.notes_json),
    )


@router.post("/{run_id}/predict", response_model=PredictResponse)
def predict(
    run_id: int,
    payload: PredictRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> PredictResponse:
    run = db.execute(
        select(Run).where(Run.id == run_id, Run.owner_id == user.id)
    ).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    record = _load_owned_dataset(db, run.dataset_id, user)
    try:
        dataset = load_dataset(record.csv_text, record.target_column, record.task)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    winner_params = {}
    for row in json.loads(run.leaderboard_json):
        if row["key"] == run.winner_key and row["rank"] == 1:
            winner_params = row["params"]
            break

    try:
        scored = score_rows(dataset, run.winner_key, winner_params, payload.rows)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    for item in scored:
        db.add(
            Prediction(
                run_id=run.id,
                row_index=item["row_index"],
                predicted=item["predicted"],
                confidence=item["confidence"],
            )
        )
    db.commit()

    return PredictResponse(
        run_id=run.id,
        task=run.task,
        predictions=[PredictionOut(**item) for item in scored],
    )
