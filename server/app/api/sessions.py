from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Device, RecordingSession
from app.schemas.session import SessionCreateRequest, SessionResponse
from app.services.auth import authenticate_device

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionCreateRequest,
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> RecordingSession:
    session = RecordingSession(
        session_id=payload.session_id,
        device_id=device.device_id,
        activity=payload.activity,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
        sample_count=payload.sample_count,
        status=payload.status,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("", response_model=list[SessionResponse])
def list_sessions(
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> list[RecordingSession]:
    return list(
        db.scalars(
            select(RecordingSession)
            .where(RecordingSession.device_id == device.device_id)
            .order_by(RecordingSession.started_at.desc())
        )
    )


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    device: Device = Depends(authenticate_device),
    db: Session = Depends(get_db),
) -> RecordingSession:
    session = db.get(RecordingSession, session_id)
    if session is None or session.device_id != device.device_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session
