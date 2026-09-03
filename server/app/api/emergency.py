"""
FastAPI router for Emergency Alerts.
Endpoint: POST /api/v1/emergency
"""

import os
import sys
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Ensure server root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from emergency_service import format_emergency_email, send_smtp_email, get_smtp_config

router = APIRouter(prefix="/emergency", tags=["emergency"])


class EmergencyRequest(BaseModel):
    event: str = Field(default="FALL_CONFIRMED")
    deviceSource: str = Field(default="Unknown Device")
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    timeString: Optional[str] = None
    heartRate: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[float] = None
    eventId: Optional[str] = None
    recipients: Optional[list[str]] = None


class EmergencyResponse(BaseModel):
    status: str
    mode: Optional[str] = None
    recipients: list[str]
    eventId: Optional[str] = None
    timestamp: int
    message: Optional[str] = None


@router.post("", response_model=EmergencyResponse)
@router.post("/", response_model=EmergencyResponse)
def dispatch_emergency_alert(payload: EmergencyRequest):
    data = payload.model_dump()
    config = get_smtp_config()
    recipients = data.get("recipients") or [config["recipient"]]
    
    subject, body = format_emergency_email(data)
    result = send_smtp_email(recipients, subject, body)
    
    if not result.get("success"):
        status_code = 503 if result.get("mode") == "SMTP_UNCONFIGURED" else 500
        raise HTTPException(
            status_code=status_code,
            detail=result.get("error", "Emergency email dispatch failed")
        )

    return EmergencyResponse(
        status="SENT",
        mode=result.get("mode"),
        recipients=recipients,
        eventId=data.get("eventId"),
        timestamp=data.get("timestamp"),
        message=result.get("message")
    )
