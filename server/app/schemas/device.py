from datetime import datetime

from pydantic import BaseModel, Field


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    device_name: str = Field(min_length=1, max_length=255)
    device_type: str = Field(min_length=1, max_length=64)
    model: str | None = Field(default=None, max_length=255)
    app_version: str | None = Field(default=None, max_length=64)


class DeviceResponse(BaseModel):
    device_id: str
    device_name: str
    device_type: str
    model: str | None
    app_version: str | None
    created_at: datetime
    last_seen: datetime | None
    status: str

    model_config = {"from_attributes": True}


class DeviceRegisterResponse(BaseModel):
    device: DeviceResponse
    token: str | None = None
