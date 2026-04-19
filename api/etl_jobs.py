"""In-memory ETL job status (single-process API server)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class EtlJobRecord:
    job_id: str
    kind: str
    status: str  # queued | running | completed | failed
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


_lock = threading.Lock()
_jobs: dict[str, EtlJobRecord] = {}


def create_job(kind: str) -> str:
    import uuid

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = EtlJobRecord(job_id=job_id, kind=kind, status="queued", message="queued")
    return job_id


def get_job(job_id: str) -> EtlJobRecord | None:
    with _lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return None
        return EtlJobRecord(
            job_id=rec.job_id,
            kind=rec.kind,
            status=rec.status,
            message=rec.message,
            result=dict(rec.result) if rec.result is not None else None,
            error=rec.error,
            created_at=rec.created_at,
            updated_at=rec.updated_at,
        )


def _touch(rec: EtlJobRecord) -> None:
    rec.updated_at = datetime.now(UTC)


def mark_running(job_id: str, message: str = "") -> None:
    with _lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return
        rec.status = "running"
        rec.message = message or "running"
        _touch(rec)


def complete(job_id: str, result: dict[str, Any]) -> None:
    with _lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return
        rec.status = "completed"
        rec.message = "completed"
        rec.result = result
        rec.error = None
        _touch(rec)


def fail(job_id: str, error: str) -> None:
    with _lock:
        rec = _jobs.get(job_id)
        if rec is None:
            return
        rec.status = "failed"
        rec.message = "failed"
        rec.error = error
        _touch(rec)
