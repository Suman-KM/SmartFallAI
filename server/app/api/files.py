from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse as DownloadResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Device, FileRecord, RecordingSession
from app.schemas.file import FileUploadResponse
from app.services.auth import authenticate_device
from app.storage.filesystem import (
    build_storage_relative_path,
    move_into_storage,
    normalize_media_type,
    resolve_storage_path,
    sanitize_filename,
    stage_upload,
)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    upload: UploadFile = File(...),
    media_type: str = Form(...),
    session_id: str | None = Form(default=None),
    filename: str | None = Form(default=None),
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileUploadResponse:
    clean_media_type = normalize_media_type(media_type)
    clean_filename = sanitize_filename(filename or upload.filename or "")

    if session_id is not None:
        session = db.get(RecordingSession, session_id)
        if session is None or session.device_id != device.device_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    temp_path, size, sha256 = await stage_upload(upload, settings)
    duplicate = db.scalar(
        select(FileRecord).where(
            FileRecord.device_id == device.device_id,
            FileRecord.sha256 == sha256,
            FileRecord.size == size,
        )
    )
    if duplicate is not None:
        temp_path.unlink(missing_ok=True)
        return FileUploadResponse(file=duplicate, duplicate=True)

    relative_path = build_storage_relative_path(
        device_id=device.device_id,
        filename=clean_filename,
        media_type=clean_media_type,
        session_id=session_id,
    )
    target_path = resolve_storage_path(settings, relative_path)
    if target_path.exists():
        relative_path = build_storage_relative_path(
            device_id=device.device_id,
            filename=f"{Path(clean_filename).stem}-{sha256[:12]}{Path(clean_filename).suffix}",
            media_type=clean_media_type,
            session_id=session_id,
        )
        target_path = resolve_storage_path(settings, relative_path)

    move_into_storage(temp_path, target_path)
    record = FileRecord(
        device_id=device.device_id,
        session_id=session_id,
        filename=clean_filename,
        relative_path=relative_path,
        media_type=clean_media_type,
        size=size,
        sha256=sha256,
        status="uploaded",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return FileUploadResponse(file=record, duplicate=False)


@router.get("/{file_id}/download")
def download_file(
    file_id: str,
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DownloadResponse:
    record = db.get(FileRecord, file_id)
    if record is None or record.device_id != device.device_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    path = resolve_storage_path(settings, record.relative_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file missing")
    return DownloadResponse(path=path, filename=record.filename, media_type="application/octet-stream")
