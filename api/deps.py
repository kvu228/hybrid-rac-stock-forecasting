"""Shared FastAPI dependencies.

Currently exposes `DbConn`, an async psycopg connection acquired from the
application-level pool created in `api.main`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

import psycopg
from fastapi import Depends, Request


async def get_db_conn(request: Request) -> AsyncIterator[psycopg.AsyncConnection[tuple[object, ...]]]:
    """Yield an async DB connection from the app pool.

    Note: prefer passing `DbConn` into endpoints instead of creating new pools.
    """
    pool = request.app.state.db_pool
    async with pool.connection() as conn:
        yield conn


DbConn = Annotated[psycopg.AsyncConnection[tuple[object, ...]], Depends(get_db_conn)]