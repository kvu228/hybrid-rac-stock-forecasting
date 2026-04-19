"""FastAPI application entry point.

Creates the app, initializes the async PostgreSQL connection pool, and mounts
routers for OHLCV, metadata (S/R zones), and RAC stored-procedure calls.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool


if sys.platform == "win32":
    # psycopg async doesn't support ProactorEventLoop on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: open/close the async DB connection pool."""
    load_dotenv(override=False)
    db_url = _database_url()
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")

    pool = AsyncConnectionPool(conninfo=db_url, min_size=2, max_size=10, open=False)
    await pool.open()
    app.state.db_pool = pool
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="Hybrid-RAC-Stock API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register routers ---
from api.routers.benchmark import router as benchmark_router  # noqa: E402
from api.routers.etl import router as etl_router  # noqa: E402
from api.routers.metadata import router as metadata_router  # noqa: E402
from api.routers.ohlcv import router as ohlcv_router  # noqa: E402
from api.routers.rac import router as rac_router  # noqa: E402

app.include_router(ohlcv_router, prefix="/api", tags=["OHLCV"])
app.include_router(metadata_router, prefix="/api", tags=["Metadata"])
app.include_router(rac_router, prefix="/api", tags=["RAC"])
app.include_router(etl_router, prefix="/api", tags=["ETL"])
app.include_router(benchmark_router, prefix="/api", tags=["Benchmark"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}