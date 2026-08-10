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
- `POST /api/v1/sessions`
- `GET /api/v1/sessions`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/files/upload`
- `GET /api/v1/sync/manifest/{device_id}`
- `GET /api/v1/devices/{device_id}/files`
- `GET /api/v1/files/{file_id}/download`

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

## Sync Protocol

Clients should keep a local pending-upload queue. For each local file, calculate SHA-256 and compare it against:

```bash
GET /api/v1/sync/manifest/<device_id>
```

The manifest includes `file_id`, `relative_path`, `media_type`, `size`, `sha256`, `status`, and timestamps. Clients can classify local files as already uploaded, missing, changed, or duplicate before uploading. Failed uploads should stay pending locally and be retried later. The server deduplicates uploads by `device_id`, `sha256`, and `size`.

## Development Workflow

Run tests:

```bash
cd server
pytest
```

The tests use SQLite with dependency overrides so they do not require PostgreSQL. Production remains configured for PostgreSQL through `SMARTFALLAI_DATABASE_URL`.
