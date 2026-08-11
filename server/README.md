# SmartFallAI Server

Phase 2 adds a Linux server/client architecture so normal SmartFallAI data access does not depend on ADB. ADB remains useful for development, APK installation, debugging, and manual recovery.

## Prerequisites

- Python 3.11 or newer
- PostgreSQL 14 or newer
- `pip` and `venv`

## Python Setup

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## PostgreSQL Setup

Create a database and application user:

```sql
CREATE USER smartfallai WITH PASSWORD 'smartfallai';
CREATE DATABASE smartfallai OWNER smartfallai;
```

Use stronger credentials outside local development.

## Environment

Copy the example file and edit it:

```bash
cp .env.example .env
```

Important settings:

- `SMARTFALLAI_DATABASE_URL`: PostgreSQL SQLAlchemy URL.
- `SMARTFALLAI_SECRET_KEY`: long random server secret used to hash device tokens.
- `SMARTFALLAI_STORAGE_ROOT`: filesystem root for uploaded data.
- `SMARTFALLAI_MAX_UPLOAD_SIZE_BYTES`: upload limit.
- `SMARTFALLAI_CORS_ORIGINS`: comma-separated trusted client origins.

Do not commit a real `.env`.

## Database Migrations

```bash
cd server
alembic upgrade head
```

The initial migration creates:

- `devices`
- `recording_sessions`
- `files`

The schema leaves room for future `fall_events`, `emergency_events`, `users`, and `sync_jobs` tables.

## Start Server

```bash
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## API Overview

- `GET /health`
- `POST /api/v1/devices/register`
- `GET /api/v1/devices`
- `GET /api/v1/devices/me`
- `POST /api/v1/devices/heartbeat`
- `POST /api/v1/sessions`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/files/upload`
- `GET /api/v1/files/{file_id}`
- `PATCH /api/v1/files/{file_id}/status`
- `GET /api/v1/files/{file_id}/download`
- `GET /api/v1/sync/manifest/{device_id}`
- `POST /api/v1/sync/check-file`
- `GET /api/v1/devices/{device_id}/files`

Device registration returns a bearer token only when the device is first created. Store that token on the device or client. All normal device, sync, upload, and download APIs require:

```http
Authorization: Bearer <token>
```

Tokens are never stored in plaintext. The server stores HMAC-SHA256 hashes using `SMARTFALLAI_SECRET_KEY`.

## Storage Structure

Large binaries are stored on disk. PostgreSQL stores metadata only.

```text
storage/
  devices/
    <device_id>/
      sessions/
        <session_id>/
          sensor/
          videos/
          frames/
          photos/
          gallery/
```

Files not tied to a recording session are stored under `storage/devices/<device_id>/<media_type>/`.

Supported media types are:

- `sensor`
- `videos`
- `frames`
- `photos`
- `gallery`
- `other`

## File Synchronization API

All file synchronization endpoints require bearer authentication. A device can only read, update, upload, download, or check files that belong to its own `device_id`. The API returns metadata paths relative to `SMARTFALLAI_STORAGE_ROOT`; it never returns absolute filesystem paths.

### Upload

```http
POST /api/v1/files/upload
```

Multipart form fields:

- `upload`: required file part.
- `media_type`: required; one of the supported media types below.
- `session_id`: optional recording session id owned by the authenticated device.
- `filename`: optional server filename override.
- `sha256`: optional client-calculated SHA-256.
- `size`: optional client-calculated byte size.

The server always calculates its own SHA-256 and size. If client-provided `sha256` or `size` does not match the server calculation, the request is rejected and the staged temporary file is removed. Duplicate detection is based on `device_id`, `sha256`, and `size`. Actual file bytes are stored on disk; PostgreSQL stores metadata only.

### Metadata

```http
GET /api/v1/files/{file_id}
```

Returns metadata for a single owned file. It does not download bytes and does not expose absolute filesystem paths.

```http
GET /api/v1/devices/{device_id}/files
```

Lists owned file metadata newest upload first. Query filters:

- `media_type`
- `session_id`
- `status`
- `limit`, default `100`, range `1..500`
- `offset`, default `0`

### Manifest

```http
GET /api/v1/sync/manifest/{device_id}
```

Returns the sync manifest for the authenticated device. Query filters:

- `media_type`
- `session_id`
- `status`

The manifest includes `file_id`, `relative_path`, `media_type`, `size`, `sha256`, `status`, and upload timestamps.

### Preflight Duplicate Check

```http
POST /api/v1/sync/check-file
```

JSON body:

```json
{
  "filename": "Walking_20260811_101500.csv",
  "media_type": "sensor",
  "size": 12345,
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "session_id": "optional-session-id"
}
```

The endpoint validates media type and optional session ownership, checks for an existing file by `device_id`, `sha256`, and `size`, and returns matching metadata when present. It does not create a database row and does not write a file.

### Status

```http
PATCH /api/v1/files/{file_id}/status
```

Allowed statuses in Phase 2.2:

- `uploaded`
- `verified`
- `archived`

This is metadata-only. It does not delete physical files.

### Download

```http
GET /api/v1/files/{file_id}/download
```

Downloads bytes for an owned file. Cross-device access returns 404.

## Client Synchronization Flow

Clients should keep a local pending-upload queue. For each local file, calculate byte size and SHA-256 locally, then call `POST /api/v1/sync/check-file` or fetch `GET /api/v1/sync/manifest/{device_id}` with filters. If the server already has the file, mark the local item complete. If it is missing, upload it with `POST /api/v1/files/upload` and include the client-calculated `sha256` and `size` fields. Failed uploads should remain pending locally and be retried later.

## Development Workflow

Run tests:

```bash
cd server
pytest
```

The tests use SQLite with dependency overrides so they do not require PostgreSQL. Production remains configured for PostgreSQL through `SMARTFALLAI_DATABASE_URL`.
