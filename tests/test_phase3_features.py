"""Tests for Phase 3: feature engineering, sliding windows, S/R detection, train/test split."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import sqlalchemy as sa

from etl.feature_engineer import (
    forward_fill_trading_days,
    generate_windows,
    relative_returns,
    train_test_split_by_time,
    zscore_normalize_window,
)
from etl.sr_detector import (
    SRZone,
    detect_pivot_points,
    detect_sr_zones,
    ingest_sr_zones,
    purge_inactive_sr_zones,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 50, symbol: str = "VCB", start: str = "2024-01-02") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with ``n`` business-day rows."""
    dates = pd.bdate_range(start=start, periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "time": dates,
            "symbol": symbol,
            "open": close - rng.uniform(0, 1, n),
            "high": close + rng.uniform(0, 2, n),
            "low": close - rng.uniform(0, 2, n),
            "close": close,
            "volume": rng.integers(1000, 5000, size=n),
        }
    )


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        import pytest

        pytest.skip("DATABASE_URL not set; skipping DB integration tests")
    return url


def _engine() -> sa.Engine:
    return sa.create_engine(_database_url(), pool_pre_ping=True)


# ---------------------------------------------------------------------------
# Feature engineering unit tests (no DB required)
# ---------------------------------------------------------------------------


class TestZscoreNormalize:
    def test_output_shape(self) -> None:
        w = np.random.default_rng(0).normal(size=(30, 5))
        normed = zscore_normalize_window(w)
        assert normed.shape == (30, 5)

    def test_mean_near_zero(self) -> None:
        w = np.random.default_rng(1).normal(loc=50, scale=10, size=(30, 5))
        normed = zscore_normalize_window(w)
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=1e-10)

    def test_constant_channel_zeroed(self) -> None:
        w = np.ones((30, 5))
        normed = zscore_normalize_window(w)
        np.testing.assert_array_equal(normed, np.zeros((30, 5)))


class TestForwardFill:
    def test_fills_missing_business_days(self) -> None:
        # Create data with a gap (skip a business day)
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-02", "2024-01-04"], utc=True),
                "symbol": "VCB",
                "open": [100.0, 102.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [104.0, 105.0],
                "volume": [1000, 1100],
            }
        )
        result = forward_fill_trading_days(df)
        # 2024-01-02 (Tue), 2024-01-03 (Wed, filled), 2024-01-04 (Thu)
        assert len(result) == 3
        filled_row = result[result["time"].dt.day == 3].iloc[0]
        assert filled_row["close"] == 104.0  # forward-filled from Jan 2
        assert filled_row["volume"] == 0  # filled with 0

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
        result = forward_fill_trading_days(df)
        assert result.empty


class TestRelativeReturns:
    def test_adds_close_ret(self) -> None:
        df = _make_ohlcv(10)
        result = relative_returns(df)
        assert "close_ret" in result.columns
        assert result["close_ret"].iloc[0] == 0.0
        # Second row should be (close[1] - close[0]) / close[0]
        expected = (df["close"].iloc[1] - df["close"].iloc[0]) / df["close"].iloc[0]
        np.testing.assert_almost_equal(result["close_ret"].iloc[1], expected)


class TestGenerateWindows:
    def test_basic_window_generation(self) -> None:
        df = _make_ohlcv(50, symbol="VCB")
        records = generate_windows(df, window_size=30, horizon=5, stride=1)
        # With 50 rows, max_start = 50 - 30 - 5 = 15 → 16 windows (indices 0..15)
        assert len(records) == 16
        for r in records:
            assert r.symbol == "VCB"
            assert r.data.shape == (30, 5)
            assert r.label in (0, 1, 2)

    def test_stride_reduces_count(self) -> None:
        df = _make_ohlcv(50)
        recs_s1 = generate_windows(df, window_size=30, horizon=5, stride=1)
        recs_s5 = generate_windows(df, window_size=30, horizon=5, stride=5)
        assert len(recs_s5) < len(recs_s1)

    def test_too_short_returns_empty(self) -> None:
        df = _make_ohlcv(10)
        records = generate_windows(df, window_size=30, horizon=5)
        assert records == []

    def test_multi_symbol(self) -> None:
        df1 = _make_ohlcv(50, symbol="VCB")
        df2 = _make_ohlcv(50, symbol="FPT", start="2024-01-02")
        df = pd.concat([df1, df2], ignore_index=True)
        records = generate_windows(df, window_size=30, horizon=5)
        symbols = {r.symbol for r in records}
        assert symbols == {"VCB", "FPT"}


class TestTrainTestSplit:
    def test_split_ratio(self) -> None:
        df = _make_ohlcv(80)
        records = generate_windows(df, window_size=30, horizon=5)
        train, test = train_test_split_by_time(records, train_ratio=0.8)
        assert len(train) + len(test) == len(records)
        assert len(train) == int(len(records) * 0.8)

    def test_chronological_order(self) -> None:
        df = _make_ohlcv(80)
        records = generate_windows(df, window_size=30, horizon=5)
        train, test = train_test_split_by_time(records, train_ratio=0.8)
        if train and test:
            assert max(r.window_end for r in train) <= min(r.window_end for r in test)

    def test_empty_input(self) -> None:
        train, test = train_test_split_by_time([])
        assert train == []
        assert test == []


# ---------------------------------------------------------------------------
# S/R detection unit tests (no DB required)
# ---------------------------------------------------------------------------


class TestSRDetection:
    def test_detect_pivot_points_basic(self) -> None:
        df = _make_ohlcv(100, symbol="VCB")
        zones = detect_pivot_points(df, order=5)
        assert len(zones) > 0
        for z in zones:
            assert z.symbol == "VCB"
            assert z.zone_type in ("SUPPORT", "RESISTANCE")
            assert z.price_level > 0
            assert z.strength >= 1.0

    def test_too_short_returns_empty(self) -> None:
        df = _make_ohlcv(5)
        zones = detect_pivot_points(df, order=5)
        assert zones == []

    def test_detect_sr_zones_multi_symbol(self) -> None:
        df1 = _make_ohlcv(100, symbol="VCB")
        df2 = _make_ohlcv(100, symbol="FPT", start="2024-01-02")
        df = pd.concat([df1, df2], ignore_index=True)
        zones = detect_sr_zones(df, order=5)
        symbols = {z.symbol for z in zones}
        assert "VCB" in symbols
        assert "FPT" in symbols


# ---------------------------------------------------------------------------
# Integration tests (require live DB)
# ---------------------------------------------------------------------------


def test_ingest_sr_zones_roundtrip() -> None:
    """Insert S/R zones into DB and verify they are stored correctly."""
    db_url = _database_url()
    symbol = "__TEST_SR__"

    # Clean up any prior test data
    with _engine().begin() as conn:
        conn.execute(
            sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"),
            {"sym": symbol},
        )

    zones = [
        SRZone(symbol=symbol, zone_type="SUPPORT", price_level=95.0, strength=3.0),
        SRZone(symbol=symbol, zone_type="RESISTANCE", price_level=110.0, strength=5.0),
    ]
    inserted = ingest_sr_zones(zones, db_url)
    assert inserted == 2

    with _engine().connect() as conn:
        rows = conn.execute(
            sa.text(
                "SELECT zone_type, price_level, strength, is_active "
                "FROM support_resistance_zones WHERE symbol = :sym ORDER BY price_level"
            ),
            {"sym": symbol},
        ).fetchall()

    assert len(rows) == 2
    assert rows[0][0] == "SUPPORT"
    assert float(rows[0][1]) == 95.0
    assert rows[0][3] is True
    assert rows[1][0] == "RESISTANCE"

    # Re-run should deactivate old and insert new
    inserted2 = ingest_sr_zones(zones, db_url)
    assert inserted2 == 2

    with _engine().connect() as conn:
        active = conn.execute(
            sa.text(
                "SELECT COUNT(*) FROM support_resistance_zones WHERE symbol = :sym AND is_active = TRUE"
            ),
            {"sym": symbol},
        ).scalar_one()
        assert int(active) == 2

    # Cleanup
    with _engine().begin() as conn:
        conn.execute(
            sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"),
            {"sym": symbol},
        )


def test_purge_inactive_sr_zones() -> None:
    """Inactive S/R rows are removed; active rows for the same symbol remain."""
    db_url = _database_url()
    symbol = "__TEST_PURGE_SR__"

    with _engine().begin() as conn:
        conn.execute(sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"), {"sym": symbol})
        conn.execute(
            sa.text(
                "INSERT INTO support_resistance_zones (symbol, zone_type, price_level, strength, is_active) VALUES "
                "(:sym, 'SUPPORT', 1.0, 1.0, FALSE), (:sym, 'SUPPORT', 2.0, 1.0, FALSE), "
                "(:sym, 'RESISTANCE', 3.0, 1.0, TRUE)"
            ),
            {"sym": symbol},
        )

    deleted = purge_inactive_sr_zones(db_url, symbols=[symbol])
    assert deleted == 2

    with _engine().connect() as conn:
        total = conn.execute(
            sa.text("SELECT COUNT(*) FROM support_resistance_zones WHERE symbol = :sym"),
            {"sym": symbol},
        ).scalar_one()
        assert int(total) == 1
        inactive = conn.execute(
            sa.text("SELECT COUNT(*) FROM support_resistance_zones WHERE symbol = :sym AND is_active = FALSE"),
            {"sym": symbol},
        ).scalar_one()
        assert int(inactive) == 0

    with _engine().begin() as conn:
        conn.execute(sa.text("DELETE FROM support_resistance_zones WHERE symbol = :sym"), {"sym": symbol})