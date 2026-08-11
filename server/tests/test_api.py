import hashlib
from collections.abc import Iterator
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Device


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def db_session(client: TestClient) -> Iterator[Session]:
    override = client.app.dependency_overrides[get_db]
    yield from override()


def register_device(
    client: TestClient,
    device_id: str,
    device_name: str = "Galaxy Watch",
) -> tuple[str, str]:
    response = client.post(
        "/api/v1/devices/register",
        json={
            "device_id": device_id,
            "device_name": device_name,
            "device_type": "watch",
            "model": "SM-R900",
            "app_version": "1.0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["token"]
    return data["device"]["device_id"], data["token"]


def create_session(client: TestClient, token: str, session_id: str) -> dict[str, object]:
    response = client.post(
        "/api/v1/sessions",
        headers=auth_header(token),
        json={
            "session_id": session_id,
            "activity": "Fall Test",
            "started_at": "2026-08-10T10:00:00Z",
        },
    )
    assert response.status_code == 201
    return response.json()


def upload_sensor_file(
    client: TestClient, token: str, session_id: str, content: bytes = b"timestamp,accX\n1,2\n"
) -> dict[str, object]:
    response = client.post(
        "/api/v1/files/upload",
        headers=auth_header(token),
        data={"media_type": "sensor", "session_id": session_id},
        files={"upload": (f"{session_id}.csv", content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()


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


def test_device_registration_stores_hash_not_raw_token(client: TestClient) -> None:
    device_id, token = register_device(client, "watch-token-check")

    with next(db_session(client)) as db:
        device = db.get(Device, device_id)
        assert device is not None
        assert device.token_hash
        assert device.token_hash != token
        assert len(device.token_hash) == 64


def test_repeated_registration_does_not_rotate_token_hash(client: TestClient) -> None:
    device_id, _ = register_device(client, "watch-repeat")
    with next(db_session(client)) as db:
        original_hash = db.get(Device, device_id).token_hash

    response = client.post(
        "/api/v1/devices/register",
        json={
            "device_id": device_id,
            "device_name": "Renamed Watch",
            "device_type": "watch",
            "model": "SM-R900",
            "app_version": "1.1",
        },
    )
    assert response.status_code == 200
    assert response.json()["token"] is None

    with next(db_session(client)) as db:
        updated = db.get(Device, device_id)
        assert updated.token_hash == original_hash
        assert updated.device_name == "Renamed Watch"
        assert updated.app_version == "1.1"


def test_invalid_bearer_token_rejected(client: TestClient) -> None:
    response = client.get("/api/v1/devices/me", headers=auth_header("not-a-real-token"))
    assert response.status_code == 401


def test_current_device_endpoint_returns_authenticated_identity(
    client: TestClient, registered_device: tuple[str, str]
) -> None:
    device_id, token = registered_device
    response = client.get("/api/v1/devices/me", headers=auth_header(token))
    assert response.status_code == 200
    assert response.json()["device_id"] == device_id


def test_heartbeat_updates_status_and_last_seen(
    client: TestClient, registered_device: tuple[str, str]
) -> None:
    device_id, token = registered_device
    old_last_seen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with next(db_session(client)) as db:
        device = db.get(Device, device_id)
        device.last_seen = old_last_seen
        device.status = "inactive"
        db.add(device)
        db.commit()

    response = client.post(
        "/api/v1/devices/heartbeat",
        headers=auth_header(token),
        json={"status": "online"},
    )
    assert response.status_code == 200
    data = response.json()["device"]
    assert data["device_id"] == device_id
    assert data["status"] == "online"
    assert datetime.fromisoformat(data["last_seen"]) > old_last_seen


def test_sessions_are_limited_to_authenticated_device(client: TestClient) -> None:
    _, token_a = register_device(client, "watch-a")
    _, token_b = register_device(client, "watch-b")
    create_session(client, token_a, "session-a")
    create_session(client, token_b, "session-b")

    list_response = client.get("/api/v1/sessions", headers=auth_header(token_a))
    assert list_response.status_code == 200
    sessions = list_response.json()
    assert [session["session_id"] for session in sessions] == ["session-a"]

    cross_lookup = client.get("/api/v1/sessions/session-b", headers=auth_header(token_a))
    assert cross_lookup.status_code == 404


def test_cross_device_file_listing_and_manifest_are_rejected(client: TestClient) -> None:
    device_a, token_a = register_device(client, "watch-a")
    device_b, token_b = register_device(client, "watch-b")
    create_session(client, token_b, "session-b")
    upload_sensor_file(client, token_b, "session-b")

    own_manifest = client.get(f"/api/v1/sync/manifest/{device_a}", headers=auth_header(token_a))
    assert own_manifest.status_code == 200

    cross_manifest = client.get(f"/api/v1/sync/manifest/{device_b}", headers=auth_header(token_a))
    assert cross_manifest.status_code == 404

    cross_files = client.get(f"/api/v1/devices/{device_b}/files", headers=auth_header(token_a))
    assert cross_files.status_code == 404


def test_cross_device_file_download_is_rejected(client: TestClient) -> None:
    _, token_a = register_device(client, "watch-a")
    _, token_b = register_device(client, "watch-b")
    create_session(client, token_b, "session-b")
    upload = upload_sensor_file(client, token_b, "session-b")
    file_id = upload["file"]["file_id"]

    cross_download = client.get(f"/api/v1/files/{file_id}/download", headers=auth_header(token_a))
    assert cross_download.status_code == 404

    own_download = client.get(f"/api/v1/files/{file_id}/download", headers=auth_header(token_b))
    assert own_download.status_code == 200
