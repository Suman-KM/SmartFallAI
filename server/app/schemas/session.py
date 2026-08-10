from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=36)
    activity: str | None = Field(default=None, max_length=128)
    started_at: datetime
    ended_at: datetime | None = None
    sample_count: int = Field(default=0, ge=0)
    status: str = Field(default="recording", max_length=64)


class SessionResponse(BaseModel):
    session_id: str
    device_id: str
    activity: str | None
    started_at: datetime
    ended_at: datetime | None
    sample_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
