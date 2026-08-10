from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Device, FileRecord
from app.schemas.file import FileResponse, ManifestEntry, ManifestResponse
from app.services.auth import authenticate_device

router = APIRouter(tags=["sync"])


@router.get("/sync/manifest/{device_id}", response_model=ManifestResponse)
def manifest(
    device_id: str,
    _: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> ManifestResponse:
    if db.get(Device, device_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    files = db.scalars(
        select(FileRecord).where(FileRecord.device_id == device_id).order_by(FileRecord.uploaded_at.desc())
    )
    return ManifestResponse(device_id=device_id, files=[ManifestEntry.model_validate(file) for file in files])


@router.get("/devices/{device_id}/files", response_model=list[FileResponse])
def list_device_files(
    device_id: str,
    _: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> list[FileRecord]:
    if db.get(Device, device_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return list(
        db.scalars(
            select(FileRecord).where(FileRecord.device_id == device_id).order_by(FileRecord.uploaded_at.desc())
        )
    )
