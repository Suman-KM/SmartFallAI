from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    database_url = f"sqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    settings = Settings(
        database_url=database_url,
        secret_key="test-secret-key-that-is-long-enough",
        storage_root=tmp_path / "storage",
        cors_origins=[],
        max_upload_size_bytes=1024 * 1024,
    )

    def override_settings() -> Settings:
        return settings

    def override_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_settings] = override_settings
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def registered_device(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/api/v1/devices/register",
        json={
            "device_id": "watch-001",
            "device_name": "Galaxy Watch",
            "device_type": "watch",
            "model": "Galaxy Watch",
            "app_version": "1.0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    return data["device"]["device_id"], data["token"]
