"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import AsyncConnectionPool


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
from api.routers.ohlcv import router as ohlcv_router  # noqa: E402
from api.routers.metadata import router as metadata_router  # noqa: E402

app.include_router(ohlcv_router, prefix="/api", tags=["OHLCV"])
app.include_router(metadata_router, prefix="/api", tags=["Metadata"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}