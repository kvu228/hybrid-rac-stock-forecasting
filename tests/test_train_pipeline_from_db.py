"""CLI tests for ml.train_pipeline --from-db (no real DB required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from ml.train_pipeline import main


def _dense_bday_ohlcv(*, symbol: str = "VCB", n: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range(start="2020-01-02", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    open_ = close - rng.uniform(0, 0.3, size=n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.2, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.2, size=n)
    vol = rng.integers(1_000, 5_000, size=n, dtype=np.int64)
    return pd.DataFrame(
        {
            "time": dates,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


def test_from_db_errors_without_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("ml.train_pipeline.load_dotenv", lambda **_k: None)
    with pytest.raises(SystemExit, match="DATABASE_URL"):
        main(["--from-db", "--symbols", "VCB"])


def test_from_db_errors_without_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    with pytest.raises(SystemExit, match="--symbols"):
        main(["--from-db"])


def test_synthetic_and_from_db_conflict() -> None:
    with pytest.raises(SystemExit):
        main(["--synthetic", "--from-db", "--symbols", "VCB"])


def test_device_cuda_fails_fast_on_cpu_only_torch(tmp_path: Path) -> None:
    if getattr(torch.version, "cuda", None) is not None and torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is available")
    out = tmp_path / "unused.pt"
    with pytest.raises(SystemExit, match="CPU-only|cuda"):
        main(["--synthetic", "--epochs", "1", "--device", "cuda", "--batch-size", "8", "--out", str(out)])


def test_from_db_trains_with_mock_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/db")
    out = tmp_path / "enc.pt"
    df = _dense_bday_ohlcv()
    with patch("ml.train_pipeline._fetch_ohlcv_from_db", return_value=df):
        rc = main(
            [
                "--from-db",
                "--symbols",
                "VCB",
                "--epochs",
                "1",
                "--batch-size",
                "32",
                "--out",
                str(out),
                "--device",
                "cpu",
            ]
        )
    assert rc == 0
    assert out.is_file()
