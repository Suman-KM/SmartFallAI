import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.config import Settings

ALLOWED_MEDIA_TYPES = {"sensor", "videos", "frames", "photos", "gallery", "other"}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    raw = filename.strip()
    if not raw or raw in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    original = PurePosixPath(raw.replace("\\", "/"))
    if original.is_absolute() or ".." in original.parts or len(original.parts) > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    name = SAFE_NAME_RE.sub("_", raw)
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
    return name[:255]


def normalize_media_type(media_type: str) -> str:
    normalized = media_type.strip().lower()
    if normalized not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported media type")
    return normalized


def safe_relative_path(*parts: str) -> str:
    path = PurePosixPath(*parts)
    if path.is_absolute() or ".." in path.parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")
    return path.as_posix()


def build_storage_relative_path(
    device_id: str,
    filename: str,
    media_type: str,
    session_id: str | None = None,
) -> str:
    device = sanitize_filename(device_id)
    media = normalize_media_type(media_type)
    name = sanitize_filename(filename)
    if session_id:
        session = sanitize_filename(session_id)
        return safe_relative_path("devices", device, "sessions", session, media, name)
    return safe_relative_path("devices", device, media, name)


def resolve_storage_path(settings: Settings, relative_path: str) -> Path:
    root = settings.storage_root.resolve()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid storage path")
    return target


async def stage_upload(upload: UploadFile, settings: Settings) -> tuple[Path, int, str]:
    hasher = hashlib.sha256()
    size = 0
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="upload-", dir=settings.storage_root)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as buffer:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Upload exceeds configured size limit",
                    )
                hasher.update(chunk)
                buffer.write(chunk)
        return temp_path, size, hasher.hexdigest()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def move_into_storage(temp_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path = target_path.with_name(f"{target_path.stem}-{uuid4().hex}{target_path.suffix}")
    shutil.move(str(temp_path), str(target_path))
