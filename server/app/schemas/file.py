from datetime import datetime

from pydantic import BaseModel, Field


ALLOWED_FILE_STATUSES = {"uploaded", "verified", "archived"}


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


class FileStatusUpdateRequest(BaseModel):
    status: str = Field(min_length=1, max_length=64)


class CheckFileRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=64)
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[A-Fa-f0-9]{64}$")
    session_id: str | None = Field(default=None, max_length=36)


class CheckFileResponse(BaseModel):
    exists: bool
    duplicate: bool
    file: FileResponse | None = None


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
