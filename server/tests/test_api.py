import hashlib

from fastapi.testclient import TestClient


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_device_registration_returns_token_once(client: TestClient) -> None:
    payload = {
        "device_id": "watch-001",
        "device_name": "Galaxy Watch",
        "device_type": "watch",
        "model": "SM-R900",
        "app_version": "1.0",
    }
    first = client.post("/api/v1/devices/register", json=payload)
    assert first.status_code == 200
    assert first.json()["token"]

    second = client.post("/api/v1/devices/register", json=payload)
    assert second.status_code == 200
    assert second.json()["token"] is None


def test_authentication_required(client: TestClient) -> None:
    response = client.get("/api/v1/devices")
    assert response.status_code == 401


def test_authenticated_session_creation(
    client: TestClient, registered_device: tuple[str, str]
) -> None:
    _, token = registered_device
    response = client.post(
        "/api/v1/sessions",
        headers=auth_header(token),
        json={
            "activity": "Walking",
            "started_at": "2026-08-10T10:00:00Z",
            "sample_count": 12,
            "status": "saved",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["device_id"] == "watch-001"
    assert data["activity"] == "Walking"
    assert data["sample_count"] == 12


def test_file_upload_sha256_manifest_and_duplicate_detection(
    client: TestClient, registered_device: tuple[str, str]
) -> None:
    device_id, token = registered_device
    session = client.post(
        "/api/v1/sessions",
        headers=auth_header(token),
        json={
            "session_id": "session-001",
            "activity": "Fall Test",
            "started_at": "2026-08-10T10:00:00Z",
        },
    )
    assert session.status_code == 201

    content = b"timestamp,accX\n1,2\n"
    expected_hash = hashlib.sha256(content).hexdigest()
    first = client.post(
        "/api/v1/files/upload",
        headers=auth_header(token),
        data={"media_type": "sensor", "session_id": "session-001"},
        files={"upload": ("fall.csv", content, "text/csv")},
    )
    assert first.status_code == 201
    first_data = first.json()
    assert first_data["duplicate"] is False
    assert first_data["file"]["sha256"] == expected_hash
    assert first_data["file"]["relative_path"].endswith(
        "devices/watch-001/sessions/session-001/sensor/fall.csv"
    )

    duplicate = client.post(
        "/api/v1/files/upload",
        headers=auth_header(token),
        data={"media_type": "sensor", "session_id": "session-001"},
        files={"upload": ("fall-again.csv", content, "text/csv")},
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["file"]["file_id"] == first_data["file"]["file_id"]

    manifest = client.get(f"/api/v1/sync/manifest/{device_id}", headers=auth_header(token))
    assert manifest.status_code == 200
    files = manifest.json()["files"]
    assert len(files) == 1
    assert files[0]["sha256"] == expected_hash


def test_path_traversal_rejected(
    client: TestClient, registered_device: tuple[str, str]
) -> None:
    _, token = registered_device
    response = client.post(
        "/api/v1/files/upload",
        headers=auth_header(token),
        data={"media_type": "sensor", "filename": "../escape.csv"},
        files={"upload": ("escape.csv", b"bad", "text/csv")},
    )
    assert response.status_code == 400
