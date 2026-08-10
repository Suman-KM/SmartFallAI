from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Device
from app.schemas.device import DeviceRegisterRequest, DeviceRegisterResponse, DeviceResponse
from app.services.auth import authenticate_device, generate_device_token, hash_token

router = APIRouter(prefix="/devices", tags=["devices"])


@router.post("/register", response_model=DeviceRegisterResponse)
def register_device(
    payload: DeviceRegisterRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DeviceRegisterResponse:
    device = db.get(Device, payload.device_id)
    issued_token: str | None = None

    if device is None:
        issued_token = generate_device_token()
        device = Device(
            device_id=payload.device_id,
            device_name=payload.device_name,
            device_type=payload.device_type,
            model=payload.model,
            app_version=payload.app_version,
            token_hash=hash_token(issued_token, settings),
            last_seen=datetime.now(timezone.utc),
            status="active",
        )
    else:
        device.device_name = payload.device_name
        device.device_type = payload.device_type
        device.model = payload.model
        device.app_version = payload.app_version
        device.last_seen = datetime.now(timezone.utc)
        device.status = "active"

    db.add(device)
    db.commit()
    db.refresh(device)
    return DeviceRegisterResponse(device=device, token=issued_token)


@router.get("", response_model=list[DeviceResponse])
def list_devices(
    _: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> list[Device]:
    return list(db.scalars(select(Device).order_by(Device.created_at.desc())))
