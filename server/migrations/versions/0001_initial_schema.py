"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-10
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("device_id", sa.String(length=128), nullable=False),
        sa.Column("device_name", sa.String(length=255), nullable=False),
        sa.Column("device_type", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column("token_hash", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("device_id"),
    )
    op.create_table(
        "recording_sessions",
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("device_id", sa.String(length=128), nullable=False),
        sa.Column("activity", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["device_id"], ["devices.device_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_index("ix_recording_sessions_device_id", "recording_sessions", ["device_id"])
    op.create_table(
        "files",
        sa.Column("file_id", sa.String(length=36), nullable=False),
        sa.Column("device_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("media_type", sa.String(length=64), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["device_id"], ["devices.device_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["recording_sessions.session_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("file_id"),
        sa.UniqueConstraint("device_id", "sha256", "size", name="uq_files_device_sha256_size"),
    )
    op.create_index("ix_files_device_id", "files", ["device_id"])
    op.create_index("ix_files_session_id", "files", ["session_id"])
    op.create_index("ix_files_sha256", "files", ["sha256"])


def downgrade() -> None:
    op.drop_index("ix_files_sha256", table_name="files")
    op.drop_index("ix_files_session_id", table_name="files")
    op.drop_index("ix_files_device_id", table_name="files")
    op.drop_table("files")
    op.drop_index("ix_recording_sessions_device_id", table_name="recording_sessions")
    op.drop_table("recording_sessions")
    op.drop_table("devices")
