from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Device, FileRecord, RecordingSession
from app.schemas.file import (
    CheckFileRequest,
    CheckFileResponse,
    FileResponse,
    ManifestEntry,
    ManifestResponse,
)
from app.services.auth import authenticate_device
from app.storage.filesystem import normalize_media_type, sanitize_filename

router = APIRouter(tags=["sync"])


def _validate_device_id(device_id: str, device: Device) -> None:
    if device_id != device.device_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")


def _validate_session_owner(session_id: str | None, device: Device, db: Session) -> None:
    if session_id is None:
        return
    session = db.get(RecordingSession, session_id)
    if session is None or session.device_id != device.device_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")


def _filtered_file_query(
    device_id: str,
    *,
    media_type: str | None,
    session_id: str | None,
    file_status: str | None,
) -> Select[tuple[FileRecord]]:
    query = select(FileRecord).where(FileRecord.device_id == device_id)
    if media_type is not None:
        query = query.where(FileRecord.media_type == normalize_media_type(media_type))
    if session_id is not None:
        query = query.where(FileRecord.session_id == session_id)
    if file_status is not None:
        query = query.where(FileRecord.status == file_status)
    return query.order_by(FileRecord.uploaded_at.desc())


@router.get("/sync/manifest/{device_id}", response_model=ManifestResponse)
def manifest(
    device_id: str,
    media_type: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> ManifestResponse:
    _validate_device_id(device_id, device)
    _validate_session_owner(session_id, device, db)
    files = db.scalars(
        _filtered_file_query(
            device_id,
            media_type=media_type,
            session_id=session_id,
            file_status=status_filter,
        )
    )
    return ManifestResponse(device_id=device_id, files=[ManifestEntry.model_validate(file) for file in files])


@router.post("/sync/check-file", response_model=CheckFileResponse)
def check_file(
    payload: CheckFileRequest,
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> CheckFileResponse:
    sanitize_filename(payload.filename)
    normalize_media_type(payload.media_type)
    _validate_session_owner(payload.session_id, device, db)

    record = db.scalar(
        select(FileRecord).where(
            FileRecord.device_id == device.device_id,
            FileRecord.sha256 == payload.sha256.lower(),
            FileRecord.size == payload.size,
        )
    )
    return CheckFileResponse(exists=record is not None, duplicate=record is not None, file=record)


@router.get("/devices/{device_id}/files", response_model=list[FileResponse])
def list_device_files(
    device_id: str,
    media_type: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> list[FileRecord]:
    _validate_device_id(device_id, device)
    _validate_session_owner(session_id, device, db)
    query = _filtered_file_query(
        device_id,
        media_type=media_type,
        session_id=session_id,
        file_status=status_filter,
    ).limit(limit).offset(offset)
    return list(db.scalars(query))
