# SmartFallAI — Project Context

## Current Checkpoint

Current Git commit:

e226149 — Build Linux server foundation

Repository:

https://github.com/Suman-KM/SmartFallAI

Current branch:

main

The working tree was clean at this checkpoint.

---

# 1. Project Purpose

SmartFallAI is an elderly fall-detection system built around a Galaxy Watch as the primary sensing device.

The long-term system combines:

- Wear OS sensor collection
- Motion sensors
- GPS
- Health sensors
- Linux server infrastructure
- PostgreSQL
- File synchronization
- Video/media collection
- Machine-learning based fall detection
- Eventually an emergency alert workflow

The Galaxy Watch should eventually be the primary data-collection device.

ADB is for development/debugging and manual inspection only. It is NOT the final data-transfer architecture.

---

# 2. Development Roadmap

## Phase 1 — Wear OS UI

Clean and lightweight Wear OS interface.

Planned/implemented UI areas include:

- Recording state
- Live sensor status
- GPS status
- GPS coordinates
- GPS accuracy/speed
- Health sensor availability
- Device/server connection status placeholder
- Sample/recording information

The UI reads existing sensor/GPS manager flows rather than using fake values.

Known UI issue:

- Some text/background colors currently have insufficient contrast.
- Color cleanup should be done in a future UI build.

Health values may currently be unavailable because the health data manager/flow is not yet fully implemented.

---

# 3. Phase 2 — Linux Server

We are currently in Phase 2.

The Linux server is located at:

server/

Current stack:

- Python 3.13.x
- FastAPI
- Uvicorn
- SQLAlchemy
- PostgreSQL
- Alembic
- Pydantic
- pydantic-settings
- psycopg
- pytest
- uv

The server has already been created and its initial PostgreSQL schema has been migrated successfully.

The server currently has:

- device model
- file model
- recording session model
- API modules
- authentication service foundation
- filesystem storage foundation
- Alembic migration
- automated tests

---

# 4. Current Database

PostgreSQL database:

smartfallai

Database owner:

smartfallai

The initial Alembic migration has successfully run.

Current tables:

- alembic_version
- devices
- files
- recording_sessions

## devices

Important fields include:

- device_id
- device_name
- device_type
- model
- app_version
- token_hash
- created_at
- last_seen
- status

The device_id is the primary key.

Files and recording sessions reference devices.

## files

Important fields include:

- file_id
- device_id
- session_id
- filename
- relative_path
- media_type
- size
- sha256
- created_at
- uploaded_at
- status

There is a unique constraint on:

device_id + sha256 + size

This is intended to provide duplicate protection during future synchronization.

Files reference devices and may optionally reference recording sessions.

## recording_sessions

Recording sessions belong to devices and are intended to associate future data/media with specific recording events.

---

# 5. Server Verification Already Completed

The Linux server has successfully passed:

- PostgreSQL connection
- Alembic migration
- FastAPI startup
- /health endpoint
- /docs endpoint
- automated API tests

Previously verified:

PYTHONPATH=. uv run pytest

Result:

6 tests passed.

The health endpoint returned:

{"status":"ok"}

The FastAPI documentation endpoint returned HTTP 200.

Uvicorn successfully ran on:

0.0.0.0:8000

---

# 6. Linux Environment

The Linux development machine contains:

~/SmartFallAI

Server:

~/SmartFallAI/server

The server uses a Python 3.13 virtual environment.

uv is installed and is used to manage the Python environment and packages.

Example server setup:

cd ~/SmartFallAI/server

source .venv/bin/activate

Dependencies are installed with:

uv pip install -r requirements.txt

Tests:

PYTHONPATH=. uv run pytest

Alembic:

PYTHONPATH=. uv run alembic upgrade head

Server:

PYTHONPATH=. uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

---

# 7. Environment / Secrets

The following files/directories must NOT be committed:

server/.env
server/.venv/
server/storage/

Python cache files are also ignored.

server/.env.example is safe to commit.

The actual .env contains local configuration and secrets.

Never commit database passwords, authentication tokens, or secret keys.

---

# 8. Wear OS Device

Primary development device:

Samsung Galaxy Watch 4

Model:

SM-R870

The Wear OS application package is:

com.suman.smartfallai.wear

ADB has been successfully used to:

- connect to the watch
- install the debug APK
- launch the application
- inspect generated CSV files
- pull dataset files for development/testing

However:

ADB is NOT the final SmartFallAI data-transfer architecture.

The intended architecture is:

Galaxy Watch
    ↓
SmartFallAI server API
    ↓
Linux filesystem + PostgreSQL
    ↓
Mac/PC access

ADB should remain primarily a development/debugging tool.

---

# 9. Existing Watch Data

The Watch has already generated CSV recordings.

The CSV structure currently includes fields such as:

- timestamp
- accX
- accY
- accZ
- gyroX
- gyroY
- gyroZ
- pitch
- roll
- yaw
- latitude
- longitude
- altitude
- speed
- accuracy
- heartRate
- spo2
- pressure
- activity

Walking recordings have already been manually collected and inspected.

Timestamp ordering was checked and no timestamp reversals were found in the inspected recordings.

Official large-scale data collection has NOT started yet.

Infrastructure must be completed before the official dataset collection phase.

---

# 10. Media / Gallery Requirement

A major future requirement is media synchronization.

The user plans to record fall events as videos and later work with frames for the ML dataset.

The system must eventually support synchronization of device media including:

- videos
- photographs
- extracted frames
- CSV sensor recordings
- other relevant device files

The user specifically wants the server to be capable of accessing/synchronizing the complete device gallery/media collection.

The current gallery is empty, but the architecture must support future gallery contents.

Do NOT implement uncontrolled full-gallery copying into PostgreSQL.

PostgreSQL should store metadata.

Actual media should live in the Linux filesystem/storage layer.

Conceptually:

PostgreSQL
    ↓
metadata

Linux filesystem
    ↓
actual videos/photos/frames/CSV files

The existing files table is intended to support this architecture.

Media synchronization belongs to Phase 2 file management/sync work.

---

# 11. Phase 2 Planned Architecture

Phase 2 consists of:

## 2.1 Device Registration + Authentication

NEXT TASK.

The Watch/Phone must register with the Linux server.

Requirements:

- unique device identity
- device registration
- secure authentication token
- token hashing
- Bearer authentication
- authenticated device identification
- /me endpoint
- heartbeat
- last_seen
- status
- authentication tests

Raw authentication tokens must never be stored in PostgreSQL.

Only secure token hashes should be stored.

## 2.2 File Management

Future APIs should support:

- listing device files
- uploading files
- downloading files
- file metadata
- file status
- SHA-256 verification
- duplicate detection
- session association

## 2.3 Synchronization

Future synchronization should support:

- new-file detection
- upload queue
- offline queue
- retry
- resumable/reliable transfers where appropriate
- SHA-256 verification
- duplicate protection
- server-side metadata

## 2.4 Mac/PC Access

Mac and PC should eventually access synchronized device data through the Linux server rather than relying on ADB.

## 2.5 Media/Gallery Synchronization

The synchronization architecture must eventually handle:

- videos
- photos
- frames
- sensor CSV files

The system should scale to large numbers of media files.

---

# 12. Phase 3 — Data Collection

Only begin official data collection after Phase 2 infrastructure is stable.

Initial activities include:

- Walking
- Sitting
- Lying down
- Running
- Standing
- Falling
- Other activities required by the ML model

The Galaxy Watch is the primary data-collection device.

The dataset should NOT be designed around ADB.

ADB may only be used for development/testing/manual extraction.

---

# 13. Phase 4 — ML Pipeline

After infrastructure and data collection:

- dataset validation
- preprocessing
- windowing
- feature extraction
- training
- validation
- fall/non-fall model
- eventual on-device inference

Do not begin this phase prematurely.

---

# 14. Phase 5 — Emergency System

Eventually:

Fall detected
    ↓
confirmation/countdown
    ↓
GPS acquisition
    ↓
emergency event
    ↓
server synchronization
    ↓
alert workflow

This comes after the ML pipeline is sufficiently reliable.

---

# 15. Current Next Task

## Phase 2.1 — Device Registration + Authentication

The next implementation should:

1. Inspect the existing server architecture first.

2. Use the existing devices table.

3. Implement device registration.

4. Generate a secure authentication token.

5. Store only the token hash.

6. Implement Bearer-token authentication.

7. Implement authenticated:

GET /api/devices/me

8. Implement:

POST /api/devices/heartbeat

9. Update:

- last_seen
- status

10. Handle duplicate device registration cleanly.

11. Reject missing/invalid authentication with HTTP 401.

12. Add tests for:

- successful registration
- duplicate registration
- invalid registration input
- authenticated /me
- missing Authorization
- invalid token
- successful heartbeat
- last_seen/status updates

13. Preserve the existing SQLite test override.

14. Run all existing tests after implementation.

15. Inspect git diff for unrelated changes.

16. Do not introduce ADB as a dependency.

17. Do not begin gallery synchronization in this task.

---

# 16. Development Rules

Before modifying code:

- inspect the existing implementation
- understand current models/routes/services
- preserve the existing architecture
- avoid unnecessary database changes
- do not duplicate existing functionality

After modifying code:

- run tests
- inspect git diff
- inspect git status
- verify no secrets are staged
- verify no .venv/storage/cache files are staged

Do not commit or push automatically unless explicitly requested.

Do not reset or force-push the repository.

Do not delete working functionality without discussing it first.

Do not redesign the entire architecture unnecessarily.

---

# 17. Git Checkpoint

Current important commits:

e226149 — Build Linux server foundation

6c8768f — Integrated the watch gps and health sensors

514f836 — Sprint 2: Phone sensors and GPS integration

The repository is hosted on GitHub:

https://github.com/Suman-KM/SmartFallAI

At the current checkpoint, the working tree was clean and main was synchronized with origin/main.

---

# 18. Important Project Principle

The final system should NOT depend on:

- ADB
- manual adb pull
- manually copying files from the Watch
- designing the dataset around desktop extraction

The intended production flow is:

Galaxy Watch
    ↓
SmartFallAI Watch client
    ↓
Authenticated server API
    ↓
Linux server
    ├── PostgreSQL metadata
    └── filesystem media/data
          ↓
      Mac / PC access

This architecture should support offline operation and later synchronization.

---

# CURRENT STATUS

Phase 1:
Wear OS UI and sensor/GPS plumbing implemented, with some UI polish remaining.

Phase 2:
Linux server foundation implemented.

PostgreSQL:
Initialized and migrated.

FastAPI:
Running and verified.

Tests:
6/6 previously passed.

GitHub:
Clean checkpoint at e226149.

Official data collection:
Not started.

ML:
Not started.

Emergency system:
Not started.

---

# NEXT TASK

Implement:

Phase 2.1 — Device Registration + Authentication

Do not skip directly to ML or official data collection.

---

# IMPORTANT CONSTRAINTS

- Preserve existing architecture.
- Use the Galaxy Watch as the primary sensing/data device.
- ADB is development/debugging only.
- Never commit .env.
- Never commit .venv.
- Never commit storage/media data.
- Never store raw authentication tokens in PostgreSQL.
- Keep PostgreSQL for metadata and Linux filesystem/storage for actual media.
- Test every server change.
- Inspect git diff before committing.
- Do not push unless explicitly requested.
