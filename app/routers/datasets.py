"""Dataset routes: profile a CSV, store it, list it, read it back."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..automl.data import load_dataset, profile_csv
from ..database import get_db
from ..models import Dataset, User
from ..schemas import (
    DatasetCreate,
    DatasetDetailOut,
    DatasetOut,
    DatasetProfileOut,
    ProfileRequest,
)
from ..security import current_user

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/profile", response_model=DatasetProfileOut)
def profile(payload: ProfileRequest, user: User = Depends(current_user)) -> DatasetProfileOut:
    """Infer column types and a suggested target without saving anything.

    This is the step the UI calls first, so the user can confirm the target
    before a dataset row is created.
    """
    try:
        result = profile_csv(payload.csv_text)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return DatasetProfileOut(**result.to_dict())


@router.post("", response_model=DatasetDetailOut, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DatasetDetailOut:
    try:
        dataset = load_dataset(payload.csv_text, payload.target_column, payload.task)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    record = Dataset(
        owner_id=user.id,
        name=payload.name.strip(),
        target_column=payload.target_column,
        task=dataset.task,
        n_rows=dataset.n_rows,
        n_columns=dataset.profile.n_columns,
        csv_text=payload.csv_text,
        profile_json=json.dumps(dataset.profile.to_dict()),
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You already have a dataset named '{payload.name.strip()}'",
        )
    db.refresh(record)

    return DatasetDetailOut(
        id=record.id,
        name=record.name,
        target_column=record.target_column,
        task=record.task,
        n_rows=record.n_rows,
        n_columns=record.n_columns,
        created_at=record.created_at,
        profile=DatasetProfileOut(**json.loads(record.profile_json)),
    )


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    db: Session = Depends(get_db), user: User = Depends(current_user)
) -> list[DatasetOut]:
    rows = (
        db.execute(
            select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [DatasetOut.model_validate(r) for r in rows]


@router.get("/{dataset_id}", response_model=DatasetDetailOut)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DatasetDetailOut:
    record = db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id)
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return DatasetDetailOut(
        id=record.id,
        name=record.name,
        target_column=record.target_column,
        task=record.task,
        n_rows=record.n_rows,
        n_columns=record.n_columns,
        created_at=record.created_at,
        profile=DatasetProfileOut(**json.loads(record.profile_json)),
    )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    record = db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id)
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    db.delete(record)
    db.commit()
