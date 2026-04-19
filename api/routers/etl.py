"""ETL HTTP API: seed / backfill / incremental as background jobs."""

from __future__ import annotations

import os

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.etl_jobs import complete, create_job, fail, get_job, mark_running
from api.schemas import (
    EtlBackfillRequest,
    EtlIncrementalRequest,
    EtlJobAcceptedResponse,
    EtlJobStatusResponse,
)
from etl.pipeline import _get_db_today, parse_symbols_list, run_backfill, run_incremental, run_seed_small_dataset

router = APIRouter()


def _database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def _run_seed_job(job_id: str) -> None:
    try:
        mark_running(job_id, "loading fixture")
        res = run_seed_small_dataset(database_url=_database_url())
        complete(job_id, res)
    except Exception as e:  # noqa: BLE001 — surface to job status
        fail(job_id, str(e))


def _run_backfill_job(job_id: str, payload: EtlBackfillRequest) -> None:
    try:
        mark_running(job_id, "backfill")
        symbols = parse_symbols_list(payload.symbols, None)
        res = run_backfill(
            _database_url(),
            symbols,
            payload.start,
            payload.end,
            chunk_days=payload.chunk_days,
            concurrency=payload.concurrency,
            requests_per_minute=payload.requests_per_minute,
            rate_limit_burst=payload.rate_limit_burst,
        )
        complete(job_id, res)
    except Exception as e:  # noqa: BLE001
        fail(job_id, str(e))


def _run_incremental_job(job_id: str, payload: EtlIncrementalRequest) -> None:
    try:
        mark_running(job_id, "incremental")
        symbols = parse_symbols_list(payload.symbols, None)
        db_url = _database_url()
        end = payload.end if payload.end is not None else _get_db_today(db_url)
        res = run_incremental(
            db_url,
            symbols,
            end,
            chunk_days=payload.chunk_days,
            concurrency=payload.concurrency,
            requests_per_minute=payload.requests_per_minute,
            rate_limit_burst=payload.rate_limit_burst,
        )
        complete(job_id, res)
    except Exception as e:  # noqa: BLE001
        fail(job_id, str(e))


@router.post("/etl/seed", response_model=EtlJobAcceptedResponse)
async def etl_seed(background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        _database_url()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    job_id = create_job("seed")
    background_tasks.add_task(_run_seed_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@router.post("/etl/backfill", response_model=EtlJobAcceptedResponse)
async def etl_backfill(background_tasks: BackgroundTasks, req: EtlBackfillRequest) -> dict[str, str]:
    try:
        _database_url()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    job_id = create_job("backfill")
    background_tasks.add_task(_run_backfill_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.post("/etl/incremental", response_model=EtlJobAcceptedResponse)
async def etl_incremental(background_tasks: BackgroundTasks, req: EtlIncrementalRequest) -> dict[str, str]:
    try:
        _database_url()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    job_id = create_job("incremental")
    background_tasks.add_task(_run_incremental_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@router.get("/etl/status/{job_id}", response_model=EtlJobStatusResponse)
async def etl_status(job_id: str) -> dict[str, object]:
    rec = get_job(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return {
        "job_id": rec.job_id,
        "kind": rec.kind,
        "status": rec.status,
        "message": rec.message,
        "result": rec.result,
        "error": rec.error,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
    }
