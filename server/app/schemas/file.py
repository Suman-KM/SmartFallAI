from datetime import datetime

from pydantic import BaseModel


class FileResponse(BaseModel):
    file_id: str
    device_id: str
    session_id: str | None
    filename: str
    relative_path: str
    media_type: str
    size: int
    sha256: str
    created_at: datetime
    uploaded_at: datetime
    status: str

    model_config = {"from_attributes": True}


class FileUploadResponse(BaseModel):
    file: FileResponse
    duplicate: bool


class ManifestEntry(BaseModel):
    file_id: str
    session_id: str | None
    filename: str
    relative_path: str
    media_type: str
    size: int
    sha256: str
    status: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ManifestResponse(BaseModel):
    device_id: str
    files: list[ManifestEntry]
