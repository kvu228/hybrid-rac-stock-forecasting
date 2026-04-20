"""Shared API wiring: sidebar base-URL input, HTTP client, symbol picker."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import httpx
import streamlit as st


def api_base_url_input(*, label: str = "API base URL") -> str:
    """Render the sidebar text_input and return a trimmed base URL."""
    value = st.sidebar.text_input(
        label,
        value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"),
    )
    return value.rstrip("/")


@contextmanager
def http_client(api: str, *, timeout: float = 120.0) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=api, timeout=timeout) as client:
        yield client


def pick_symbol(client: httpx.Client, *, label: str = "Symbol") -> str:
    """Fetch ``/api/symbols`` and render a selectbox; stop the page if empty/failed."""
    resp = client.get("/api/symbols")
    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()
    symbols = [r["symbol"] for r in resp.json()]
    if not symbols:
        st.warning("No symbols in database.")
        st.stop()
    return st.selectbox(label, symbols)
