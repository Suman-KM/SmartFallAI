from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("created_at", "last_seen")
    @classmethod
    def assume_utc_for_naive_datetimes(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class DeviceRegisterResponse(BaseModel):
    device: DeviceResponse
    token: str | None = None


class DeviceHeartbeatRequest(BaseModel):
    status: str = Field(default="active", min_length=1, max_length=64)


class DeviceHeartbeatResponse(BaseModel):
    device: DeviceResponse
