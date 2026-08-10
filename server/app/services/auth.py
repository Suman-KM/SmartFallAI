import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Device

bearer_scheme = HTTPBearer(auto_error=False)


def generate_device_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(token: str, settings: Settings) -> str:
    return hmac.new(
        settings.secret_key.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_token(token: str, token_hash: str, settings: Settings) -> bool:
    return hmac.compare_digest(hash_token(token, settings), token_hash)


def authenticate_device(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Device:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required",
        )

    presented_hash = hash_token(credentials.credentials, settings)
    device = db.scalar(select(Device).where(Device.token_hash == presented_hash))
    if device is None or not verify_token(credentials.credentials, device.token_hash, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )

    device.last_seen = datetime.now(timezone.utc)
    db.add(device)
    db.commit()
    db.refresh(device)
    return device
